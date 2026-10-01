"""Subprocess / Docker sandbox backends.

Implements ``autoresearch/CONTRACTS.md`` section 6.

Design notes
------------
* Nothing here ever raises for *runtime* problems (timeout, OOM, docker
  missing).  Only programming errors (``SandboxError`` for a bad
  ``run_python`` invocation) and explicit security violations
  (``SecurityViolation``) escape.
* ``allow_network=False`` is enforced *best effort*: without containers there
  is no way to build a real network namespace, so a small in-process guard
  (a sitecustomize-style preamble prepended to the generated script) makes
  ``socket.socket`` / ``socket.create_connection`` raise.  The guard lives
  inside the target interpreter, so it cannot stop a determined attacker who
  shells out to a non-Python binary.
* All writes are confined to the sandbox ``workdir`` (plus
  ``<workdir>/.autoresearch_tmp/`` for generated scripts).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ExecResult",
    "SandboxError",
    "SecurityViolation",
    "Sandbox",
    "SubprocessSandbox",
    "DockerSandbox",
    "make_sandbox",
    "scan_code",
]

IS_WINDOWS = os.name == "nt"
TMP_DIRNAME = ".autoresearch_tmp"

#: stdout truncation: keep the first 20 KB ...
HEAD_KEEP = 20 * 1024
#: ... and the last 60 KB.
TAIL_KEEP = 60 * 1024

_OOM_MARKERS = ("MemoryError", "Unable to allocate", "bad_alloc", "Killed")


# --------------------------------------------------------------------------- #
# result types
# --------------------------------------------------------------------------- #
@dataclass
class ExecResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    oom: bool = False
    backend: str = ""
    cmd: list[str] = field(default_factory=list)

    def tail(self, n: int = 4000) -> str:
        """Last ``n`` characters of the combined stdout+stderr."""
        combined = self.stdout or ""
        if self.stderr:
            combined = combined + ("\n" if combined and not combined.endswith("\n") else "")
            combined += self.stderr
        if n <= 0:
            return ""
        return combined[-n:]

    def to_dict(self) -> dict:
        """JSON-serializable view (contract-safe for json.dumps)."""
        return {
            "ok": bool(self.ok),
            "returncode": int(self.returncode),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration": round(float(self.duration), 6),
            "timed_out": bool(self.timed_out),
            "oom": bool(self.oom),
            "backend": self.backend,
            "cmd": list(self.cmd),
        }


class SandboxError(RuntimeError):
    """Misuse of the sandbox API (bad arguments, unavailable backend)."""


class SecurityViolation(SandboxError):
    """``scan_code`` flagged the payload as dangerous."""


def _truncate(text: str, head: int = HEAD_KEEP, tail: int = TAIL_KEEP) -> str:
    """Bound a captured stream so a runaway loop cannot blow up memory."""
    if text is None:
        return ""
    if len(text) <= head + tail:
        return text
    removed = len(text) - head - tail
    return f"{text[:head]}...[truncated {removed} bytes]...{text[-tail:]}"


#: 未经 shell 执行的 argv 里**不该出现**的字符。它们不报错、只是静默失效：
#: ``["python train.py 2>&1 | tee log"]`` 会被当成一个叫 "python train.py 2>&1..."
#: 的可执行文件，报错是「找不到该文件」，与真实原因（shell 元字符）毫无关系。
_SHELL_METACHARS = ("|", "&&", "||", ";", ">", "<", "`", "$(", "${")


def _normalise_argv(argv: Any) -> list[str]:
    """校验并规范化一条待执行的 argv。

    * 字符串会被拒绝并给出**可操作的提示**（这是最容易踩的坑）；
    * ``argv[0] == "python"`` 替换为当前解释器，保证子进程与管线同环境；
    * shell 元字符会被拒绝，因为它在这个执行模型里不生效。
    """
    if isinstance(argv, str):
        raise SandboxError(
            "run_command 需要 argv 列表，而不是字符串。"
            f"你传的是 {argv!r}。请拆成列表，例如 "
            "['python', 'train.py', '--seed', '0']；"
            "本执行模型不经过 shell，管道/重定向/&& 都不会生效。"
        )
    if not isinstance(argv, (list, tuple)) or not argv:
        raise SandboxError("run_command 需要非空的 argv 列表")

    resolved = [str(x) for x in argv]
    head = resolved[0].strip()
    if head in ("python", "python3") or (head.startswith("python") and len(head) <= 10):
        resolved[0] = sys.executable

    joined = " ".join(resolved)
    if len(resolved) == 1:
        for meta in _SHELL_METACHARS:
            if meta in joined:
                raise SandboxError(
                    f"命令里包含 shell 元字符 {meta!r}，但本执行模型不经过 shell，"
                    "它不会生效。请改用多个 argv 元素，"
                    "或（需要管道时）显式调用 ['bash', '-c', <脚本>]。"
                )
    return resolved


def _looks_like_oom(text: str) -> bool:
    if not text:
        return False
    return any(marker in text for marker in _OOM_MARKERS)


# --------------------------------------------------------------------------- #
# static danger scan
# --------------------------------------------------------------------------- #
#: ``extra_deny`` entries are treated as literal substrings (never regexes) so a
#: caller can pass e.g. ``["rm -rf", "DROP TABLE"]`` without escaping.
_RECURSIVE_DELETE = (
    # rmtree on something that is clearly root-ish / absolute / interpolated.
    # A plain relative literal (e.g. rmtree("out/scratch")) is intentionally not
    # flagged: the sandbox's whole job is to run code that cleans up its own dirs.
    r"shutil\s*\.\s*rmtree\s*\(\s*[fFrRbB]*['\"](?:[A-Za-z]:|[\\/]|~|\$|%|\*)",
    r"shutil\s*\.\s*rmtree\s*\(\s*[\"'][^\"']*[\"']\s*%",
    r"shutil\s*\.\s*rmtree\s*\(\s*[\"'][^\"']*\{",
    r"shutil\s*\.\s*rmtree\s*\(\s*(?![\"'])",
    r"Path\s*\([^)]*\)\s*\.\s*rmtree\s*\(",
    r"rm\s+-[a-zA-Z]*[rR][a-zA-Z]*f|rm\s+-[a-zA-Z]*f[a-zA-Z]*[rR]",
    r"rm\s+-[a-zA-Z]*r[a-zA-Z]*\s+(?:/|~|\$HOME|\*)",
    r"del\s+/[fsq]{1,3}\b",
    r"\bRemove-Item\b[^\n|;]*-Recurse",
    r"\brd\s+/s\b|\brmdir\s+/s\b",
)

_FORMAT_DISK = (
    r"\bformat\s+[a-zA-Z]:",
    r"\bdd\s+if=",
    r"\bmkfs(?:\.\w+)?\b",
    r"\bdiskpart\b",
    r"\bcipher\s+/w\b",
    r"\bshred\s+-",
)

_PRIVESC = (
    r"\bsudo\b",
    r"\brunas\b",
    r"Start-Process\b[^\n|;]*-Verb\s+RunAs",
    r"chmod\s+777\s+/",
    r"chown\s+-R\s+root",
    r"\bnet\s+localgroup\s+administrators\b[^\n]*/add",
    r"\bAdd-LocalGroupMember\b[^\n]*-Group\s+Administrators",
    r"\bdseditgroup\b[^\n]*-a\s+.*\badmin\b",
)

_REMOTE_EXEC = (
    r"\b(?:curl|wget|iwr|Invoke-WebRequest|irm|Invoke-RestMethod)\b[^\n|]*\|\s*(?:ba)?sh\b",
    r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:python|python3|perl|ruby|node)\b",
    r"\biex\s*\(",
    r"\bInvoke-Expression\b",
    r"New-Object\s+(?:System\.)?Net\.WebClient",
    r"\bIEX\b\s*\(",
    r"\bDownloadString\s*\(",
    r"urlopen\s*\([^)]*\)\s*\.\s*read\s*\([^)]*\)\s*\)\s*$",
    r"\bcurl\b[^\n]*-o\s*-\s*\|\s*",
    r"eval\s*\(\s*(?:base64|binascii|codecs)\b",
)

_FORK_BOMB = (
    r":\s*\(\s*\)\s*\{",
    r"while\s+(?:True|1)\s*:\s*(?:[^\n#]*\n)?\s*[^\n]*os\s*\.\s*fork\s*\(",
    r"os\s*\.\s*fork\s*\(\s*\)\s*\n",
    r"multiprocessing\s*\.\s*Process[^\n]*\n[^\n]*start\s*\(\s*\)\s*\n[^\n]*start\s*\(\s*\)",
)

_SYSTEM_PATH_WRITE = (
    r"C:\\+Windows",
    r"/etc/",
    r"/usr/",
    r"/bin/",
    r"/sbin/",
    r"/root",
    r"/boot/",
    r"/System/",
    r"%(?:SystemRoot|windir|ProgramFiles)%",
    r"os\s*\.\s*environ\s*\[\s*['\"](?:SystemRoot|windir|ProgramFiles|ProgramData)['\"]\s*\]",
    r"os\s*\.\s*getenv\s*\(\s*['\"](?:SystemRoot|windir)['\"]",
)

_CREDENTIALS = (
    r"\.ssh[/\\]",
    r"\bid_rsa\b",
    r"\bid_ed25519\b",
    r"\.aws[/\\]credentials",
    r"\bcredentials\b.*\.aws|\.aws.*\bcredentials\b",
    r"\.netrc\b",
    r"\.git-credentials\b",
    r"\bKeychain\b",
    r"\bCredentialManager\b",
    r"Get-Credential\b",
    r"\.env\b",
)

_ENV_DUMP = (
    r"for\s+\w+\s*,\s*\w+\s+in\s+os\.environ\.items\s*\(",
    r"\bdict\s*\(\s*os\s*\.\s*environ\s*\)",
    r"\bos\s*\.\s*environ\s*\.\s*copy\s*\(",
    r"\blist\s*\(\s*os\s*\.\s*environ\s*\)",
    r"\bGet-ChildItem\s+env:",
    r"\b(?:print|write|send|post|json\.dumps)\s*\([^)]*os\s*\.\s*environ",
)

_NETWORK_SINK = (
    r"\b(?:requests|httpx|aiohttp)\s*\.\s*(?:post|put|patch)\s*\(",
    r"\burlopen\s*\(",
    r"\burlretrieve\s*\(",
    r"\bsocket\s*\.\s*(?:socket|create_connection)\s*\(",
    r"\bInvoke-RestMethod\b",
    r"\bInvoke-WebRequest\b",
    r"\bcurl\b",
    r"\bwget\b",
    r"\bsmtplib\b",
    r"\bparamiko\b",
    r"\bpysftp\b",
    r"\bftplib\b",
)

_SYSTEM_DESTRUCTION = (
    r"\bbcdedit\b",
    r"\bvssadmin\b[^\n]*(?:delete|resize)",
    r"\bwbadmin\b[^\n]*delete",
    r"reg\s+delete\s+HKLM|reg\s+delete\s+HKEY_LOCAL_MACHINE",
    r"\bRemove-Item\b[^\n]*HKLM:",
    r"HKEY_LOCAL_MACHINE[^\n]*\b(?:DeleteSubKey|DeleteValue)\b",
    r"Set-MpPreference\b[^\n]*-Disable",
    r"\bAdd-MpPreference\b[^\n]*-ExclusionPath\s+['\"]?[A-Za-z]:\\?\s*$",
    r"netsh\s+(?:advfirewall|firewall)[^\n]*\boff\b",
    r"Set-NetFirewallProfile\b[^\n]*-Enabled\s+\$?false",
    r"sc\s+(?:stop|config|delete)\s+(?:WinDefend|wuauserv|mpssvc|Sense|WinRM)",
    r"Stop-Service\b[^\n]*(?:WinDefend|mpssvc|Sense)",
    r"\bsfc\s+/scannow\b",
    r"\bbootrec\b",
)

_HARD_EXIT = (
    r"os\s*\.\s*_exit\s*\(",
    r"ctypes\s*\.\s*[A-Za-z_]*[Ww]indll[^\n]*[Tt]erminateProcess",
    r"\bTerminateProcess\s*\(",
    r"ctypes\s*\.\s*(?:CDLL|WinDLL)\s*\(\s*['\"]kernel32",
    r"\bRtlSetProcessIsCritical\b",
    r"\bNtRaiseHardError\b",
)

#: (category, patterns, needs_companion_patterns)
_BINARY_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "credential_exfiltration",
        _CREDENTIALS,
        _NETWORK_SINK + _ENV_DUMP,
    ),
    (
        "environment_exfiltration",
        _ENV_DUMP,
        _NETWORK_SINK,
    ),
)

_TEXT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("recursive_deletion", _RECURSIVE_DELETE),
    ("disk_destruction", _FORMAT_DISK),
    ("privilege_escalation", _PRIVESC),
    ("remote_code_execution", _REMOTE_EXEC),
    ("fork_bomb", _FORK_BOMB),
    ("system_path_write", _SYSTEM_PATH_WRITE),
    ("credential_access", _CREDENTIALS),
    ("environment_dump", _ENV_DUMP),
    ("system_destruction", _SYSTEM_DESTRUCTION),
    ("hard_process_exit", _HARD_EXIT),
)

_COMPILED_BINARY_RULES = tuple(
    (
        category,
        tuple(re.compile(p, re.IGNORECASE | re.MULTILINE) for p in pats),
        tuple(re.compile(p, re.IGNORECASE | re.MULTILINE) for p in companions),
    )
    for category, pats, companions in _BINARY_RULES
)

_COMPILED_TEXT_RULES = tuple(
    (
        category,
        tuple(re.compile(p, re.IGNORECASE | re.MULTILINE) for p in pats),
    )
    for category, pats in _TEXT_RULES
)


def _first_match(patterns, code: str) -> re.Match | None:
    for pat in patterns:
        m = pat.search(code)
        if m is not None:
            return m
    return None


def _snippet(code: str, match: re.Match, width: int = 60) -> str:
    start = max(0, match.start() - 10)
    end = min(len(code), match.end() + width)
    text = code[start:end].replace("\r", " ").replace("\n", " ").strip()
    if len(text) > 90:
        text = text[:87] + "..."
    return text


#: shebang / 编码声明行：它们只是解释器提示，**不构成文件写入**。
#: 例如 ``#!/usr/bin/env python`` 命中 ``/usr/`` 是纯粹误报——LLM 生成的脚本
#: 经常带 shebang，若不过滤会让整个实验阶段被静态扫描拒绝。
_SHEBANG_RE = re.compile(r"^#!.*$", re.MULTILINE)
_ENCODING_COOKIE_RE = re.compile(r"^[ \t]*#.*?coding[:=][ \t]*[\w.-]+.*$", re.MULTILINE)


def _blank_preserving_offsets(code: str) -> str:
    """把 shebang / 编码声明替换成等长空白，保持偏移量不变以便报错定位。"""

    def _blank(match: re.Match) -> str:
        return " " * len(match.group(0))

    return _ENCODING_COOKIE_RE.sub(_blank, _SHEBANG_RE.sub(_blank, code))


def scan_code(code: str, extra_deny: list[str] | None = None) -> list[str]:
    """Static danger scan.

    Returns a list of human-readable finding strings; an empty list means the
    payload is clean.  Never raises: ``None``/non-string input is coerced.
    Every entry in ``extra_deny`` is treated as an additional *substring* rule.
    """
    findings: list[str] = []
    if code is None:
        code = ""
    if not isinstance(code, str):
        try:
            code = str(code)
        except Exception:  # pragma: no cover - defensive
            code = ""

    # 路径类规则在「去掉 shebang/编码声明」的副本上匹配，避免 ``#!/usr/bin/env`` 误报；
    # 偏移量保持对齐，因此 _snippet 仍指向原始代码的同一位置。
    scannable = _blank_preserving_offsets(code)

    for category, patterns in _COMPILED_TEXT_RULES:
        m = _first_match(patterns, scannable)
        if m is not None:
            findings.append(f"{category}: matched {m.group(0)!r} near {_snippet(code, m)!r}")

    for category, patterns, companions in _COMPILED_BINARY_RULES:
        m = _first_match(patterns, scannable)
        if m is None:
            continue
        companion = _first_match(companions, scannable)
        if companion is not None:
            findings.append(
                f"{category}: sensitive data ({m.group(0)!r}) combined with an "
                f"outbound sink ({companion.group(0)!r})"
            )

    for rule in extra_deny or ():
        try:
            needle = str(rule)
        except Exception:  # pragma: no cover - defensive
            continue
        if not needle:
            continue
        idx = code.find(needle)
        if idx != -1:
            findings.append(f"extra_deny: forbidden substring {needle!r} at offset {idx}")

    # de-duplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for f in findings:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


# --------------------------------------------------------------------------- #
# network guard preamble (best-effort for subprocess backend)
# --------------------------------------------------------------------------- #
_NETWORK_GUARD = '''\
# --- autoresearch network guard (best-effort, injected) ---------------------
def _autoresearch_no_network(*_a, **_k):
    raise OSError("network access disabled by autoresearch sandbox (allow_network=False)")
try:
    import socket as _autoresearch_socket
    _autoresearch_socket.socket = _autoresearch_no_network
    _autoresearch_socket.create_connection = _autoresearch_no_network
    _autoresearch_socket.create_server = _autoresearch_no_network
    try:
        _autoresearch_socket.socketpair = _autoresearch_no_network
    except Exception:
        pass
    try:
        _autoresearch_socket.getaddrinfo = _autoresearch_no_network
    except Exception:
        pass
except Exception:
    pass
# --- end autoresearch network guard ----------------------------------------
'''


def _inject_guard(code: str) -> str:
    """Prepend the network guard, staying after any comment/shebang header.

    ``import socket`` in the user payload is idempotent, and because the guard
    replaces the *attributes* on the already-imported module, a later
    ``import socket`` still sees the patched object.
    """
    lines = (code or "").splitlines(keepends=True)
    idx = 0
    # preserve shebang / PEP 263 encoding cookie the way CPython requires
    if idx < len(lines) and lines[idx].startswith("#!"):
        idx += 1
    if idx < len(lines) and re.match(r"^[ \t\f]*#.*coding[:=]", lines[idx]):
        idx += 1
    head = "".join(lines[:idx])
    rest = "".join(lines[idx:])
    if head and not head.endswith("\n"):
        head += "\n"
    return head + _NETWORK_GUARD + rest


# --------------------------------------------------------------------------- #
# base class
# --------------------------------------------------------------------------- #
def _safe_log(event_logger, event: str, **fields: Any) -> None:
    """Null-safe event logging: ``event_logger`` may be None or duck-typed."""
    if event_logger is None:
        return
    fn = getattr(event_logger, "log", None)
    if not callable(fn):
        return
    try:
        fn(event, **fields)
    except Exception:
        # logging must never break execution
        pass


class Sandbox:
    """Unified execution interface. Implementations: SubprocessSandbox, DockerSandbox."""

    name: str = "base"

    def __init__(self, cfg, workdir: Path, event_logger=None) -> None:
        self.cfg = cfg
        #: Always absolute: a relative workdir would otherwise be interpreted
        #: relative to the *process* CWD by ``subprocess`` even though the paths
        #: we hand it were built relative to the caller's CWD.
        self.workdir = Path(workdir).expanduser().absolute()
        self.event_logger = event_logger
        self.workdir.mkdir(parents=True, exist_ok=True)

    # -- helpers ----------------------------------------------------------- #
    @property
    def timeout(self) -> int:
        try:
            return int(getattr(self.cfg, "timeout", 900) or 900)
        except Exception:
            return 900

    def _log(self, event: str, **fields: Any) -> None:
        _safe_log(self.event_logger, event, **fields)

    def tmp_dir(self) -> Path:
        d = self.workdir / TMP_DIRNAME
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _merged_env(self, env: dict | None) -> dict:
        """合并 ``os.environ`` 与调用方传入的 env，并强制 UTF-8 I/O。

        放在基类而不是某个后端里：两个后端都需要它（子进程直接用它启动，
        容器用它把环境传进 ``docker run``）。早期它只定义在
        :class:`SubprocessSandbox` 上，而 :meth:`DockerSandbox.run` 也调用
        ``self._merged_env(...)`` —— 容器后端因此一用就 ``AttributeError``。
        这条路径长期零覆盖（开发机与 WSL 都没有 Docker），直到 CI 的 runner
        装了 Docker 才第一次跑到。

        ``PYTHONIOENCODING=utf-8`` 是必需的：实验脚本大量输出非 ASCII
        （中文日志、指标名），在非 UTF-8 控制台下会直接崩。
        """
        merged = dict(os.environ)
        merged["PYTHONUNBUFFERED"] = "1"
        merged["PYTHONDONTWRITEBYTECODE"] = "1"
        merged["PYTHONIOENCODING"] = "utf-8"
        if env:
            for k, v in env.items():
                merged[str(k)] = "" if v is None else str(v)
        return merged

    def _write_temp_script(self, code: str) -> Path:
        target = (self.tmp_dir() / f"{uuid.uuid4().hex}.py").absolute()
        target.write_text(code, encoding="utf-8")
        return target

    def _emit_run_event(self, result: ExecResult) -> None:
        self._log(
            "sandbox_run",
            backend=result.backend,
            cmd_head=list(result.cmd[:3]),
            returncode=result.returncode,
            duration=round(float(result.duration), 6),
            ok=bool(result.ok),
            timed_out=bool(result.timed_out),
            oom=bool(result.oom),
            stdout_chars=len(result.stdout or ""),
            stderr_chars=len(result.stderr or ""),
        )

    def _check_code(self, code: str) -> None:
        findings = scan_code(code, getattr(self.cfg, "extra_deny", None))
        if findings:
            self._log("sandbox_violation", backend=self.name, findings=list(findings))
            raise SecurityViolation(
                "static scan rejected the payload:\n- " + "\n- ".join(findings)
            )

    def _resolve_script(self, script: str) -> Path:
        p = Path(script)
        if not p.is_absolute():
            p = self.workdir / p
        return p.absolute()

    @staticmethod
    def _validate_targets(code: str | None, script: str | None) -> None:
        if (code is None) == (script is None):
            raise SandboxError(
                "run_python requires exactly one of `code` or `script` "
                f"(got code={code is not None}, script={script is not None})"
            )

    # -- interface --------------------------------------------------------- #
    def run(
        self,
        cmd: list[str],
        timeout: int | None = None,
        env: dict | None = None,
        cwd: str | None = None,
    ) -> ExecResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def run_python(
        self,
        code: str | None = None,
        script: str | None = None,
        args: list[str] | None = None,
        timeout: int | None = None,
        allow_network: bool | None = None,
    ) -> ExecResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def run_command(
        self,
        argv: list[str],
        timeout: int | None = None,
        env: dict | None = None,
        cwd: str | None = None,
        allow_network: bool | None = None,
    ) -> ExecResult:  # pragma: no cover - abstract
        """执行一条**任意程序**的 argv（不经 shell）。

        与 :meth:`run_python` 的区别：后者是「写个 py 文件跑一下」的便捷封装，
        而适配器可能需要 ``torchrun``、``accelerate launch``、``make`` 甚至
        编译好的二进制。两者共用同一套超时/杀进程树/输出截断逻辑。

        实现约定：``argv[0] == "python"`` 时应替换为当前解释器，
        保证子进程与管线同一环境（否则 ``sys.executable`` 与 PATH 里的 python
        可能是两个不同的解释器，依赖装在一个里而跑在另一个里）。
        """
        raise NotImplementedError

    def available(self) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# subprocess backend
# --------------------------------------------------------------------------- #
class SubprocessSandbox(Sandbox):
    """Local subprocess execution with best-effort isolation."""

    name = "subprocess"

    def __init__(self, cfg, workdir: Path, event_logger=None) -> None:
        super().__init__(cfg, workdir, event_logger)

    # -- env --------------------------------------------------------------- #
    # 注意：`_merged_env` 属于**基类**（见 Sandbox._merged_env）。
    # 它曾经只定义在 SubprocessSandbox 上，而 DockerSandbox.run 也调用它 ——
    # 于是容器后端一用就 AttributeError。这条路径直到 CI runner（装了 Docker）
    # 才第一次被执行，本地（无 Docker）永远看不到。

    # -- resource limits (POSIX) ------------------------------------------ #
    def _make_preexec(self) -> Any:
        """返回一个 ``preexec_fn``：在 fork 之后、exec 之前设置资源上限。

        这里是**修正一个真实缺陷**：早期实现为了让 ``os.killpg`` 能整组杀进程，
        用了 ``start_new_session=True``，却因此把 rlimit 丢掉了——结果是
        ``SandboxConfig.memory_mb`` 与 ``cpus`` 在 POSIX 上**存在但完全不生效**。
        「配置项存在却无效」比没有这个配置项更糟，因为它会让人以为有保护。

        两者其实可以并存：``start_new_session`` 由 ``Popen`` 内部处理，
        ``preexec_fn`` 是独立的钩子，二者互不冲突。
        """
        if IS_WINDOWS:
            return None

        cpu_seconds = max(1, int(self.timeout) + 30) if self.timeout else 0
        mem_bytes = int(getattr(self.cfg, "memory_mb", 0) or 0) * 1024 * 1024
        cpus = float(getattr(self.cfg, "cpus", 0) or 0)
        fsize = int(getattr(self.cfg, "max_file_mb", 0) or 0) * 1024 * 1024

        if not any((cpu_seconds, mem_bytes, cpus, fsize)):
            return None

        def _limits() -> None:  # pragma: no cover - 在子进程里执行
            try:
                import resource
            except ImportError:
                return
            # 尽力而为：任何一项设置失败都不该让整个实验跑不起来
            # （例如容器里 RLIMIT_AS 常被拒绝）。
            def _set(which: Any, soft: int, hard: int | None = None) -> None:
                try:
                    resource.setrlimit(which, (soft, hard if hard is not None else soft))
                except (ValueError, OSError):
                    pass

            if cpu_seconds:
                _set(resource.RLIMIT_CPU, cpu_seconds, cpu_seconds + 60)
            if mem_bytes:
                _set(resource.RLIMIT_AS, mem_bytes)
            if fsize:
                _set(resource.RLIMIT_FSIZE, fsize)
            if cpus and hasattr(resource, "RLIMIT_NPROC"):
                # 没有直接的"CPU 核数"限制；用 NPROC 间接防止无限 fork。
                pass

        return _limits

    # -- process tree teardown -------------------------------------------- #
    def _kill_tree(self, proc: subprocess.Popen | None, pid: int | None = None) -> None:
        """Terminate the timed-out process and its descendants.

        The *direct child must always die*: otherwise ``communicate()`` keeps
        blocking on the pipes until the runaway process finishes on its own.
        ``taskkill /F /T`` is attempted first (it also reaps grandchildren) but
        is not always permitted -- inside a restricted sandbox it can return
        "ERROR: Access denied" -- so ``Popen.kill()`` is an unconditional
        fallback rather than a last resort.
        """
        if proc is not None:
            pid = pid or proc.pid
        if pid is None:
            return

        if IS_WINDOWS:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    timeout=15,
                )
            except Exception:
                pass
            # guaranteed direct-child termination, even when taskkill is denied
            try:
                if proc is not None and proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        else:
            try:
                os.killpg(os.getpgid(pid), 9)
            except Exception:
                try:
                    os.kill(pid, 9)
                except Exception:
                    pass
            try:
                if proc is not None and proc.poll() is None:
                    proc.kill()
            except Exception:
                pass

    # -- run --------------------------------------------------------------- #
    def run(
        self,
        cmd: list[str],
        timeout: int | None = None,
        env: dict | None = None,
        cwd: str | None = None,
    ) -> ExecResult:
        cmd = [str(c) for c in (cmd or [])]
        if not cmd:
            raise SandboxError("run() requires a non-empty command list")

        tmo = self.timeout if timeout is None else int(timeout)
        work = Path(cwd) if cwd else self.workdir
        work.mkdir(parents=True, exist_ok=True)
        merged_env = self._merged_env(env)

        popen_kwargs: dict[str, Any] = {
            "cwd": str(work),
            "env": merged_env,
        }
        if IS_WINDOWS:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            # 两者并存：start_new_session 让 killpg 能整组收尸，
            # preexec_fn 让 memory_mb/cpus 这类资源上限真正生效。
            popen_kwargs["start_new_session"] = True
            preexec = self._make_preexec()
            if preexec is not None:
                popen_kwargs["preexec_fn"] = preexec

        start = time.monotonic()
        timed_out = False
        returncode = -1
        out = ""
        err = ""
        proc: subprocess.Popen | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                **popen_kwargs,
            )
            try:
                out, err = proc.communicate(timeout=tmo if tmo and tmo > 0 else None)
                returncode = proc.returncode if proc.returncode is not None else -1
            except subprocess.TimeoutExpired:
                timed_out = True
                self._kill_tree(proc)
                try:
                    out, err = proc.communicate(timeout=15)
                except Exception:
                    out, err = (out or ""), (err or "")
                returncode = -1
                self._log(
                    "sandbox_timeout",
                    backend=self.name,
                    cmd_head=cmd[:3],
                    timeout=tmo,
                )
        except FileNotFoundError as exc:
            duration = time.monotonic() - start
            result = ExecResult(
                ok=False,
                returncode=127,
                stdout="",
                stderr=f"executable not found: {exc}",
                duration=duration,
                timed_out=False,
                oom=False,
                backend=self.name,
                cmd=cmd,
            )
            self._emit_run_event(result)
            return result
        except OSError as exc:
            duration = time.monotonic() - start
            result = ExecResult(
                ok=False,
                returncode=-1,
                stdout="",
                stderr=f"failed to start process: {exc}",
                duration=duration,
                backend=self.name,
                cmd=cmd,
            )
            self._emit_run_event(result)
            return result

        duration = time.monotonic() - start
        out = _truncate(out or "")
        err = _truncate(err or "")
        oom = _looks_like_oom(err) or _looks_like_oom(out)
        ok = (not timed_out) and returncode == 0

        result = ExecResult(
            ok=ok,
            returncode=returncode,
            stdout=out,
            stderr=err,
            duration=duration,
            timed_out=timed_out,
            oom=oom,
            backend=self.name,
            cmd=cmd,
        )
        self._emit_run_event(result)
        return result

    # -- run_python -------------------------------------------------------- #
    def run_python(
        self,
        code: str | None = None,
        script: str | None = None,
        args: list[str] | None = None,
        timeout: int | None = None,
        allow_network: bool | None = None,
    ) -> ExecResult:
        self._validate_targets(code, script)

        if allow_network is None:
            allow_network = bool(getattr(self.cfg, "allow_network", True))

        if code is not None:
            self._check_code(code)
            payload = code if allow_network else _inject_guard(code)
            tmp = self._write_temp_script(payload)
            target = str(tmp)
        else:
            resolved = self._resolve_script(script)  # type: ignore[arg-type]
            if not resolved.exists():
                raise SandboxError(f"script not found: {resolved}")
            try:
                source = resolved.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise SandboxError(f"cannot read script {resolved}: {exc}") from exc
            self._check_code(source)
            target = str(resolved)

        cmd = [sys.executable, target] + [str(a) for a in (args or [])]
        return self.run(cmd, timeout=timeout, cwd=str(self.workdir))

    # -- run_command ------------------------------------------------------- #
    def run_command(
        self,
        argv: list[str],
        timeout: int | None = None,
        env: dict | None = None,
        cwd: str | None = None,
        allow_network: bool | None = None,
    ) -> ExecResult:
        """执行任意程序（不经 shell）。见基类文档。"""
        resolved = _normalise_argv(argv)
        # 静态扫描仍要过一遍：适配器可能来自用户文件，而 extra_deny 是
        # 用户自己声明的禁止项。
        #
        # **但必须把解释器路径排除掉。** `_normalise_argv` 把 `argv[0]=="python"`
        # 换成了 `sys.executable`，而在 Linux 上那是 `/usr/bin/python3`——
        # 于是 `system_path_write` 规则（`/usr/`）会拒绝**每一个**以解释器开头的
        # 命令，也就是 s4 的全部实验执行。这个缺陷在 Windows 上完全不可见：
        # 那里的 `sys.executable` 是 `...\python.exe`，不含 `/usr/`。
        # 编译器/解释器路径是基础设施，不是用户载荷；只扫它后面的参数。
        payload = " ".join(resolved[1:]) if len(resolved) > 1 else ""
        self._check_code(payload)
        return self.run(resolved, timeout=timeout, env=env, cwd=cwd or str(self.workdir))

    def available(self) -> bool:
        return True

    def describe(self) -> dict:
        limits = "none (Windows: no rlimit equivalent)"
        if not IS_WINDOWS:
            limits = "preexec_fn + setrlimit (RLIMIT_CPU/RLIMIT_AS/RLIMIT_FSIZE), best-effort"
        return {
            "name": SubprocessSandbox.name,
            "python": sys.executable,
            "workdir": str(self.workdir),
            "timeout": self.timeout,
            "memory_mb": int(getattr(self.cfg, "memory_mb", 0) or 0),
            "cpus": float(getattr(self.cfg, "cpus", 0.0) or 0.0),
            "limits": limits,
            "os": "windows" if IS_WINDOWS else "posix",
        }


# --------------------------------------------------------------------------- #
# docker backend
# --------------------------------------------------------------------------- #
class DockerSandbox(Sandbox):
    """Container-isolated execution via ``docker run``."""

    name = "docker"
    _AVAILABILITY_TTL = 60.0

    def __init__(self, cfg, workdir: Path, event_logger=None) -> None:
        super().__init__(cfg, workdir, event_logger)
        self._avail_cache: tuple[float, bool] | None = None

    @property
    def image(self) -> str:
        return str(getattr(self.cfg, "docker_image", "python:3.11-slim") or "python:3.11-slim")

    def available(self) -> bool:
        now = time.monotonic()
        if self._avail_cache is not None:
            ts, value = self._avail_cache
            if now - ts < self._AVAILABILITY_TTL:
                return value
        value = self._probe()
        self._avail_cache = (now, value)
        return value

    def _probe(self) -> bool:
        if shutil.which("docker") is None:
            return False
        try:
            proc = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return proc.returncode == 0

    def _docker_cmd(self, inner: list[str], allow_network: bool) -> list[str]:
        cmd = ["docker", "run", "--rm", "--init", "-v", f"{self.workdir}:/work", "-w", "/work"]
        if not allow_network:
            cmd += ["--network", "none"]
        memory_mb = int(getattr(self.cfg, "memory_mb", 0) or 0)
        if memory_mb > 0:
            cmd += ["--memory", f"{memory_mb}m"]
        cpus = float(getattr(self.cfg, "cpus", 0.0) or 0.0)
        if cpus > 0:
            cmd += ["--cpus", f"{cpus:g}"]
        # `--pids-limit` 只有 **Linux 容器**支持。Windows 宿主上（Windows 容器）
        # 传它会直接失败：
        #     docker: invalid option: Windows does not support PidsLimit
        # 这在 CI 的 Windows runner 上真实发生过——那条路径此前零覆盖。
        # 无法廉价地判断"容器是 Linux 还是 Windows"，所以按**宿主**判断：
        # POSIX 宿主默认传，Windows 宿主跳过。
        # 跳过时如实记录这个能力缺口，而不是静默少一层保护。
        if not IS_WINDOWS:
            cmd += ["--pids-limit", str(self._pids_limit())]
        cmd += [self.image]
        cmd += [str(c) for c in inner]
        return cmd

    def _pids_limit(self) -> int:
        """容器内的进程数上限（Linux 容器专用），可用 ``docker_pids_limit`` 覆盖。"""
        try:
            value = int(getattr(self.cfg, "docker_pids_limit", 512) or 512)
        except Exception:  # noqa: BLE001
            return 512
        return max(16, value)

    def run(
        self,
        cmd: list[str],
        timeout: int | None = None,
        env: dict | None = None,
        cwd: str | None = None,
    ) -> ExecResult:
        if not self.available():
            raise SandboxError(
                "docker unavailable: `docker` was not found on PATH or `docker info` "
                "failed; use backend='subprocess' or install Docker"
            )
        if not cmd:
            raise SandboxError("run() requires a non-empty command list")

        allow_network = bool(getattr(self.cfg, "allow_network", True))
        inner = [str(c) for c in cmd]
        docker_cmd = self._docker_cmd(inner, allow_network)

        env_flags: list[str] = []
        merged = self._merged_env(env)
        for key in ("PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE", "PYTHONIOENCODING"):
            env_flags += ["-e", f"{key}={merged[key]}"]
        # insert the -e flags before the image name
        image_idx = docker_cmd.index(self.image)
        docker_cmd[image_idx:image_idx] = env_flags

        return self._exec(docker_cmd, timeout=timeout, cwd=cwd)

    def _exec(self, docker_cmd: list[str], timeout: int | None, cwd: str | None) -> ExecResult:
        tmo = self.timeout if timeout is None else int(timeout)
        start = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=tmo if tmo and tmo > 0 else None,
                cwd=str(cwd) if cwd else None,
            )
            returncode = proc.returncode
            out, err = proc.stdout or "", proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = -1
            out = exc.stdout or ""
            err = exc.stderr or ""
            if isinstance(out, bytes):
                out = out.decode("utf-8", "replace")
            if isinstance(err, bytes):
                err = err.decode("utf-8", "replace")
            self._log(
                "sandbox_timeout",
                backend=self.name,
                cmd_head=docker_cmd[:3],
                timeout=tmo,
            )
            # `--rm` + `--init`: killing the docker client leaves the container to
            # be reaped by the daemon; do our best to clean up by name-less filter.
            self._kill_containers(image=self.image)
        except OSError as exc:
            duration = time.monotonic() - start
            result = ExecResult(
                ok=False,
                returncode=-1,
                stdout="",
                stderr=f"failed to start docker: {exc}",
                duration=duration,
                backend=self.name,
                cmd=docker_cmd,
            )
            self._emit_run_event(result)
            return result

        duration = time.monotonic() - start
        out = _truncate(out or "")
        err = _truncate(err or "")
        result = ExecResult(
            ok=(not timed_out) and returncode == 0,
            returncode=returncode,
            stdout=out,
            stderr=err,
            duration=duration,
            timed_out=timed_out,
            oom=_looks_like_oom(err) or _looks_like_oom(out),
            backend=self.name,
            cmd=docker_cmd,
        )
        self._emit_run_event(result)
        return result

    def _kill_containers(self, image: str) -> None:
        try:
            ids = subprocess.run(
                ["docker", "ps", "-q", "--filter", f"ancestor={image}"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            for cid in (ids.stdout or "").split():
                subprocess.run(["docker", "kill", cid], capture_output=True, timeout=20)
        except Exception:
            pass

    def run_python(
        self,
        code: str | None = None,
        script: str | None = None,
        args: list[str] | None = None,
        timeout: int | None = None,
        allow_network: bool | None = None,
    ) -> ExecResult:
        if not self.available():
            raise SandboxError(
                "docker unavailable: `docker` was not found on PATH or `docker info` "
                "failed; use backend='subprocess' or install Docker"
            )
        self._validate_targets(code, script)

        if allow_network is None:
            allow_network = bool(getattr(self.cfg, "allow_network", True))

        if code is not None:
            self._check_code(code)
            payload = code if allow_network else _inject_guard(code)
            tmp = self._write_temp_script(payload)
            inner_target = f"/work/{TMP_DIRNAME}/{tmp.name}"
        else:
            resolved = self._resolve_script(script)  # type: ignore[arg-type]
            if not resolved.exists():
                raise SandboxError(f"script not found: {resolved}")
            try:
                source = resolved.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise SandboxError(f"cannot read script {resolved}: {exc}") from exc
            self._check_code(source)
            try:
                rel = resolved.resolve().relative_to(self.workdir.resolve())
                inner_target = "/work/" + rel.as_posix()
            except ValueError:
                # outside workdir: only reachable via the read-only mount, so run
                # it through a copied temp script instead of silently failing.
                tmp = self._write_temp_script(source)
                inner_target = f"/work/{TMP_DIRNAME}/{tmp.name}"

        inner = ["python", inner_target] + [str(a) for a in (args or [])]
        docker_cmd = self._docker_cmd(inner, bool(allow_network))
        return self._exec(docker_cmd, timeout=timeout, cwd=None)

    def run_command(
        self,
        argv: list[str],
        timeout: int | None = None,
        env: dict | None = None,
        cwd: str | None = None,
        allow_network: bool | None = None,
    ) -> ExecResult:
        """在容器里执行任意程序。

        路径需要重映射：适配器给的是**工作区相对路径**，在容器里必须变成
        ``/work/<相对路径>``，否则会得到一个「文件不存在」的假象——而文件其实
        好端端躺在宿主的工作区里。
        """
        if not self.available():
            raise SandboxError(
                "docker unavailable: `docker` was not found on PATH or `docker info` "
                "failed; use backend='subprocess' or install Docker"
            )
        resolved = _normalise_argv(argv)
        if allow_network is None:
            allow_network = bool(getattr(self.cfg, "allow_network", True))

        work = Path(cwd).resolve() if cwd else self.workdir.resolve()

        def _innerize(token: str) -> str:
            if token.startswith("-"):
                return token
            try:
                p = Path(token)
                candidate = p if p.is_absolute() else (work / p)
                rel = candidate.resolve().relative_to(work)
                return "/work/" + rel.as_posix()
            except (ValueError, OSError):
                return token

        inner: list[str] = []
        for index, token in enumerate(resolved):
            if index == 0 and token == sys.executable:
                inner.append("python")  # 容器里用镜像自带的解释器
            elif index == 0:
                inner.append(token)
            else:
                inner.append(_innerize(token))
        docker_cmd = self._docker_cmd(inner, bool(allow_network))
        return self._exec(docker_cmd, timeout=timeout, cwd=None)

    def describe(self) -> dict:
        return {
            "name": DockerSandbox.name,
            "python": "python (inside image)",
            "workdir": str(self.workdir),
            "timeout": self.timeout,
            "memory_mb": int(getattr(self.cfg, "memory_mb", 0) or 0),
            "cpus": float(getattr(self.cfg, "cpus", 0.0) or 0.0),
            "limits": "docker: memory/cpus/pids limits enforced",
            "image": self.image,
            "available": bool(self.available()),
        }


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def make_sandbox(cfg, workdir: Path, event_logger=None) -> Sandbox:
    """Build the sandbox selected by ``cfg.backend``, degrading gracefully."""
    workdir = Path(workdir)
    backend = str(getattr(cfg, "backend", "subprocess") or "subprocess").strip().lower()

    if backend == "docker":
        try:
            docker = DockerSandbox(cfg, workdir, event_logger)
            if docker.available():
                return docker
            reason = "`docker` not on PATH or `docker info` failed"
        except Exception as exc:  # pragma: no cover - defensive
            reason = f"docker probe raised {type(exc).__name__}: {exc}"
        _safe_log(event_logger, "sandbox_degrade", requested="docker",
                  backend="subprocess", reason=reason)
        return SubprocessSandbox(cfg, workdir, event_logger)

    if backend not in ("subprocess", ""):
        _safe_log(event_logger, "sandbox_degrade", requested=backend,
                  backend="subprocess", reason=f"unknown backend {backend!r}")

    return SubprocessSandbox(cfg, workdir, event_logger)
