"""LaTeX compilation with automatic standalone-tectonic provisioning.

Implements ``autoresearch/CONTRACTS.md`` section 7.

* ``detect()`` probes ``tectonic`` (PATH *and* the vendored copy),
  ``pdflatex``, ``xelatex`` and ``latexmk``.
* ``install_tectonic()`` downloads the single-file tectonic release into
  ``PROJECT_ROOT/vendor/tectonic/`` with safe archive extraction.  It never
  raises: every failure is logged as ``tectonic_install`` with
  ``status="failed"`` and returns ``None``.
* ``compile()`` degrades gracefully: with no usable engine it returns a
  ``CompileResult(ok=False, ...)`` rather than raising.  Only a *missing
  .tex file* raises ``LatexError`` (per contract).
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "LatexError",
    "CompileResult",
    "LatexCompiler",
    "PROJECT_ROOT",
    "VENDOR_DIR",
    "TECTONIC_RELEASE_BASE",
]

#: ``autoresearch/`` package directory, resolved from this file so the tree can
#: be copied anywhere without editing a hardcoded absolute path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor" / "tectonic"

TECTONIC_RELEASE_BASE = "https://github.com/tectonic-typesetting/tectonic/releases/download"

_DOWNLOAD_TIMEOUT = 300.0
_PROBE_TIMEOUT = 10
_USER_AGENT = "autoresearch-tectonic-installer/1.0 (+https://example.org/autoresearch)"

#: engine probe order (contract order: tectonic, pdflatex, xelatex, latexmk)
ENGINE_ORDER: tuple[str, ...] = ("tectonic", "pdflatex", "xelatex", "latexmk")

_MAX_ISSUES = 50


class LatexError(RuntimeError):
    """Raised for programming-level LaTeX problems (e.g. missing .tex file)."""


@dataclass
class CompileResult:
    ok: bool
    pdf: Path | None
    log: str
    errors: list[str]
    warnings: list[str]
    engine: str = ""
    duration: float = 0.0

    def summary(self) -> str:
        status = "ok" if self.ok else "failed"
        parts = [
            f"compile {status}",
            f"engine={self.engine or 'none'}",
            f"pdf={self.pdf}" if self.pdf else "pdf=<none>",
            f"duration={self.duration:.2f}s",
            f"errors={len(self.errors)}",
            f"warnings={len(self.warnings)}",
        ]
        line = " | ".join(parts)
        if self.errors:
            line += "\n  first error: " + self.errors[0].strip().replace("\n", " ")[:200]
        return line


def _safe_log(event_logger, event: str, **fields) -> None:
    if event_logger is None:
        return
    fn = getattr(event_logger, "log", None)
    if not callable(fn):
        return
    try:
        fn(event, **fields)
    except Exception:
        pass


def _run_quiet(cmd: list[str], timeout: int = _PROBE_TIMEOUT, cwd: str | None = None):
    """Run a probe command, returning ``(returncode, combined_output)``.

    Never raises: a missing binary or timeout yields a non-zero code.
    """
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=cwd,
            env=_child_env(),
        )
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"
    except OSError as exc:
        return 126, str(exc)
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return proc.returncode, out


def _child_env(cache_dir: str | os.PathLike[str] | None = None) -> dict:
    r"""Environment for LaTeX child processes.

    Forces UTF-8 on the child's stdio so non-ASCII engine messages (e.g. the
    localized Windows "access denied" text) come back readable instead of
    mojibake, and points the temp vars at the current drive so no child has to
    reach a per-user temp path.

    ``cache_dir``：**必须传**。``tectonic`` 把宏包缓存放在
    ``TECTONIC_CACHE_DIR``，不设时它用默认位置（Windows 是
    ``%LOCALAPPDATA%\Tectonic``）——那个位置在本项目的运行环境里**不可写**，
    于是 tectonic 既读不到已填充的缓存、也写不了新缓存，**而且不会去下载缺失宏包**，
    直接在 TeX 层报「File `size11.clo' not found」。

    这个错误指向完全错误的方向（看起来像模板缺宏包，实际是缓存目录没配对），
    而它曾经让 s7 的编译闭环**从未收敛**。教训值得写在这里：
    **预检必须验证生产中真正使用的那套配置**——早期预检自己设了缓存目录，
    真实编译却没设，于是预检通过而生产失败，是个典型的虚假信心检查。
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTHONUTF8", "1")
    if cache_dir is not None:
        env["TECTONIC_CACHE_DIR"] = str(cache_dir)
    return env


def _decode(data) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        data = data.decode("utf-8", "replace")
    text = str(data)
    # Repair UTF-8 bytes that a non-UTF-8 console code page decoded as CP936/
    # CP1252 (e.g. the localized Windows message becomes readable again).
    try:
        repaired = text.encode("cp936", "strict").decode("utf-8", "strict")
        if repaired:
            text = repaired
    except (UnicodeError, LookupError):
        pass
    return text


#: Patterns that make a `tectonic`/`pdflatex` invocation look like an interface
#: mismatch rather than a genuine TeX failure; only these justify retrying with
#: a different argument form.
_INTERFACE_MISMATCH = (
    "unrecognized subcommand",
    "unexpected argument",
    "unrecognized option",
    "error: Found argument",
    "unknown option",
    "invalid subcommand",
)


def _looks_like_interface_mismatch(output: str) -> bool:
    low = (output or "").lower()
    return any(needle.lower() in low for needle in _INTERFACE_MISMATCH)


def _console_safe(text: str) -> str:
    """Make a string printable on a legacy console code page.

    Engine errors can be localized (e.g. Chinese "access denied"); embedding them
    verbatim in a message that a Windows console then fails to encode would turn
    a compile failure into a ``UnicodeEncodeError``.  Keep ASCII, replace the rest.
    """
    try:
        text.encode("ascii")
        return text
    except UnicodeEncodeError:
        return text.encode("ascii", "replace").decode("ascii")


def _last_output_line(text: str, limit: int = 150) -> str:
    for raw in reversed((text or "").splitlines()):
        line = raw.strip()
        if line:
            return _console_safe(line[:limit])
    return ""


# --------------------------------------------------------------------------- #
# log parsing (pure, standalone-usable)
# --------------------------------------------------------------------------- #
_ERROR_LINE = re.compile(r"^!.*$")
_FILE_LINE_ERROR = re.compile(r"^[^\s:][^:\n]*?:\d+:\s*.+$")
_WARNING_LINE = re.compile(
    r"LaTeX Warning|Overfull|Underfull|Package\s+\S+\s+Warning|Class\s+\S+\s+Warning",
    re.IGNORECASE,
)


def extract_errors(log: str) -> list[str]:
    """Pull error lines out of a TeX log.

    Recognises ``! ...`` lines (with the following continuation line, which is
    where TeX usually prints what it was doing) and ``file:line: message``
    matches produced by ``-file-line-error``.  De-duplicated, capped at 50.
    """
    if not log:
        return []
    lines = _decode(log).splitlines()
    found: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        text = text.rstrip()
        if not text or text in seen:
            return
        seen.add(text)
        found.append(text)

    i = 0
    n = len(lines)
    while i < n and len(found) < _MAX_ISSUES:
        raw = lines[i]
        if _ERROR_LINE.match(raw):
            entry = raw.rstrip()
            # the continuation line carries the offending context
            if i + 1 < n and lines[i + 1].strip() and not _ERROR_LINE.match(lines[i + 1]):
                entry += "\n" + lines[i + 1].rstrip()
            add(entry)
            i += 2
            continue
        if _FILE_LINE_ERROR.match(raw):
            add(raw)
        i += 1

    return found[:_MAX_ISSUES]


def extract_warnings(log: str) -> list[str]:
    """Pull warning-ish lines out of a TeX log (de-duped, capped at 50)."""
    if not log:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for raw in _decode(log).splitlines():
        if not _WARNING_LINE.search(raw):
            continue
        text = raw.rstrip()
        if not text or text in seen:
            continue
        seen.add(text)
        found.append(text)
        if len(found) >= _MAX_ISSUES:
            break
    return found


# --------------------------------------------------------------------------- #
# safe archive extraction
# --------------------------------------------------------------------------- #
def _is_within(base: Path, target: Path) -> bool:
    try:
        base_r = base.resolve()
        target_r = target.resolve()
    except OSError:  # pragma: no cover - defensive
        return False
    if target_r == base_r:
        return True
    try:
        target_r.relative_to(base_r)
        return True
    except ValueError:
        return False


def _member_is_unsafe(name: str) -> bool:
    if not name:
        return True
    if name.startswith(("/", "\\")):
        return True
    # Windows drive-absolute or UNC
    if re.match(r"^[A-Za-z]:", name):
        return True
    parts = re.split(r"[\\/]+", name)
    return ".." in parts


def _safe_extract_zip(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if _member_is_unsafe(info.filename):
                raise LatexError(f"unsafe zip member rejected: {info.filename!r}")
            target = dest / info.filename
            if not _is_within(dest, target):
                raise LatexError(f"zip member escapes target dir: {info.filename!r}")
            # symlink entries: refuse (we only need a regular binary)
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode and stat.S_ISLNK(mode):
                raise LatexError(f"symlink zip member rejected: {info.filename!r}")
        zf.extractall(dest)


def _safe_extract_tar(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:*") as tf:
        members = tf.getmembers()
        for member in members:
            if _member_is_unsafe(member.name):
                raise LatexError(f"unsafe tar member rejected: {member.name!r}")
            if member.issym() or member.islnk():
                raise LatexError(f"link tar member rejected: {member.name!r}")
            target = dest / member.name
            if not _is_within(dest, target):
                raise LatexError(f"tar member escapes target dir: {member.name!r}")
        tf.extractall(dest)


# --------------------------------------------------------------------------- #
# compiler
# --------------------------------------------------------------------------- #
class LatexCompiler:
    def __init__(self, cfg, workdir: Path, event_logger=None) -> None:
        self.cfg = cfg
        self.workdir = Path(workdir)
        self.event_logger = event_logger
        try:
            self.workdir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self._detected: str | None = None
        self._detect_done = False
        self._tectonic_supports_x = False
        #: 已安装但无法使用的具体原因（如 bundle 缓存写入被拒），供 s7/doctor 展示。
        self._tectonic_cache_error = ""

    # -- logging ----------------------------------------------------------- #
    def _log(self, event: str, **fields) -> None:
        _safe_log(self.event_logger, event, **fields)

    # -- config helpers ---------------------------------------------------- #
    @property
    def cfg_engine(self) -> str:
        return str(getattr(self.cfg, "engine", "tectonic") or "").strip().lower()

    @property
    def _auto_install(self) -> bool:
        return bool(getattr(self.cfg, "auto_install_tectonic", False))

    @property
    def _tectonic_version(self) -> str:
        return str(getattr(self.cfg, "tectonic_version", "0.15.0") or "0.15.0")

    # -- engine discovery -------------------------------------------------- #
    @staticmethod
    def vendor_dir() -> Path:
        """Vendored-binary directory, resolved at call time from the module."""
        return VENDOR_DIR

    @classmethod
    def vendored_tectonic_path(cls) -> Path | None:
        """Return the vendored tectonic binary path if it exists."""
        names = ["tectonic.exe", "tectonic"] if os.name == "nt" else ["tectonic", "tectonic.exe"]
        for name in names:
            candidate = cls.vendor_dir() / name
            if candidate.is_file():
                return candidate
        return None

    def _probe_tectonic(self, executable: str) -> tuple[bool, bool]:
        """Return ``(usable, supports_-X_compile)``."""
        code, out = _run_quiet([executable, "--version"], timeout=_PROBE_TIMEOUT)
        if code != 0:
            code, out = _run_quiet([executable, "-V"], timeout=_PROBE_TIMEOUT)
        if code != 0:
            return False, False
        supports_x = False
        m = re.search(r"tectonic\s+(\d+)\.(\d+)", out or "")
        if m:
            major, minor = int(m.group(1)), int(m.group(2))
            supports_x = (major, minor) >= (0, 15)
        else:
            # unknown version string: ask the binary itself
            code2, help_out = _run_quiet([executable, "-X", "--help"], timeout=_PROBE_TIMEOUT)
            supports_x = code2 == 0 or "compile" in (help_out or "")
        return True, supports_x

    def _probe_engine(self, engine: str) -> tuple[bool, str | None]:
        """Return ``(usable, resolved_executable)`` for a named engine."""
        if engine == "tectonic":
            found = shutil.which("tectonic")
            if found:
                usable, supports_x = self._probe_tectonic(found)
                if usable:
                    self._tectonic_supports_x = supports_x
                    return True, found
            vendored = self.vendored_tectonic_path()
            if vendored is not None:
                usable, supports_x = self._probe_tectonic(str(vendored))
                if usable:
                    self._tectonic_supports_x = supports_x
                    return True, str(vendored)
            return False, None

        flag = "--version"
        found = shutil.which(engine)
        if not found:
            return False, None
        code, _ = _run_quiet([found, flag], timeout=_PROBE_TIMEOUT)
        if code != 0:
            code, _ = _run_quiet([found, "-v"], timeout=_PROBE_TIMEOUT)
        return (code == 0), (found if code == 0 else None)

    def detect(self) -> str | None:
        """Return the first usable engine name, or ``None``.  Result is cached.

        For tectonic the probe goes further than ``--version``: a binary that cannot
        write its bundle cache is reported as **not usable**, with the reason recorded
        in ``cache_block_reason()``.  This turns an otherwise inscrutable
        ``os error 5`` at compile time into an up-front, explainable capability gap.
        """
        if self._detect_done:
            return self._detected
        engine: str | None = None
        for candidate in ENGINE_ORDER:
            try:
                usable, found = self._probe_engine(candidate)
            except Exception:
                usable = False
                found = None
            if not usable:
                continue
            if candidate == "tectonic" and found:
                try:
                    ok, reason = self._tectonic_cache_preflight(str(found))
                except Exception as exc:  # pragma: no cover - 预检绝不抛
                    ok, reason = False, f"{type(exc).__name__}: {exc}"
                if not ok:
                    self._tectonic_cache_error = reason
                    self._log(
                        "compile_detect",
                        engine="tectonic-unusable",
                        reason=reason,
                        workdir=str(self.workdir),
                    )
                    continue
                self._tectonic_cache_error = ""
            engine = candidate
            break
        self._detected = engine
        self._detect_done = True
        self._log("compile_detect", engine=engine or "none", workdir=str(self.workdir))
        return engine

    def resolve_executable(self, engine: str) -> str | None:
        """Path (or name) to run for ``engine``; ``None`` if unusable."""
        _, found = self._probe_engine(engine)
        return found

    def bibtex_available(self) -> bool:
        try:
            return shutil.which("bibtex") is not None
        except Exception:  # pragma: no cover - defensive
            return False

    @staticmethod
    def extract_errors(log: str) -> list[str]:
        """Contract method: pull error lines from a TeX log.

        Thin wrapper over the module-level :func:`extract_errors` so the parser is
        usable both as ``compiler.extract_errors(log)`` and standalone.
        """
        return extract_errors(log)

    @staticmethod
    def extract_warnings(log: str) -> list[str]:
        """Contract-adjacent helper: pull warning lines from a TeX log."""
        return extract_warnings(log)

    # -- tectonic install -------------------------------------------------- #
    @staticmethod
    def _platform_suffix() -> str:
        system = platform.system().lower()
        machine = (platform.machine() or "").lower()
        if machine in ("arm64", "aarch64"):
            if system == "darwin":
                return "aarch64-apple-darwin"
            if system == "linux":
                return "aarch64-unknown-linux-musl"
        if system == "windows":
            return "x86_64-pc-windows-msvc"
        if system == "darwin":
            return "x86_64-apple-darwin"
        return "x86_64-unknown-linux-musl"

    @classmethod
    def _candidate_archives(cls, version: str) -> list[tuple[str, str]]:
        """``(filename, url)`` candidates, best first, with tar.gz fallback."""
        suffix = cls._platform_suffix()
        zip_name = f"tectonic-{version}-{suffix}.zip"
        tar_name = f"tectonic-{version}-{suffix}.tar.gz"
        names = [zip_name, tar_name] if os.name == "nt" else [tar_name, zip_name]
        base = f"{TECTONIC_RELEASE_BASE}/tectonic%40{version}"
        return [(name, f"{base}/{name}") for name in names]

    @staticmethod
    def _download(url: str, dest: Path) -> None:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT) as response:
            # urllib follows redirects by default (GitHub release assets redirect
            # to objects.githubusercontent.com); nothing extra is required here.
            with open(dest, "wb") as fh:
                shutil.copyfileobj(response, fh, length=1024 * 256)

    @staticmethod
    def _find_binary(root: Path) -> Path | None:
        names = {"tectonic.exe", "tectonic"} if os.name == "nt" else {"tectonic", "tectonic.exe"}
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name.lower() in names:
                return path
        return None

    @staticmethod
    def _make_scratch_dir(parent: Path) -> Path:
        """Create a unique writable scratch dir under ``parent``.

        Avoids ``tempfile`` (see the note in :meth:`install_tectonic`).
        """
        for _ in range(10):
            candidate = parent / f".tectonic_dl_{os.getpid()}_{uuid.uuid4().hex[:10]}"
            try:
                os.mkdir(candidate)
                return candidate
            except FileExistsError:
                continue
        raise LatexError(f"could not create a scratch dir under {parent}")

    @staticmethod
    def _drop_scratch_dir(path: Path) -> bool:
        """Best-effort recursive delete of a scratch dir.  Returns success."""
        try:
            shutil.rmtree(path, ignore_errors=True)
            if not path.exists():
                return True
            # A permission-restricted leftover cannot be removed here.  Drop a
            # sentinel inside it so it can never be mistaken for a fresh/
            # already-empty temp dir, and so a later cleanup pass skips it.
            try:
                (path / ".autoresearch_leftover").write_text(
                    "sandbox denied removal of this directory\n", encoding="utf-8"
                )
            except Exception:
                pass
            return False
        except Exception:
            return False

    def install_tectonic(self) -> Path | None:
        """Download the standalone tectonic binary.  Never raises."""
        try:
            existing = self.vendored_tectonic_path()
        except Exception:
            existing = None
        if existing is not None:
            usable, supports_x = self._probe_tectonic(str(existing))
            if usable:
                # 已下载的二进制也要过缓存预检——否则每次运行都会在 s7 重复失败，
                # 而失败原因（沙箱拒绝写 profile）与「没装」看起来完全不同。
                preflight_ok, preflight_reason = self._tectonic_cache_preflight(str(existing))
                if not preflight_ok:
                    self._tectonic_cache_error = preflight_reason
                    self._log(
                        "tectonic_install",
                        status="installed_but_unusable",
                        path=str(existing),
                        version=self._tectonic_version,
                        reason=preflight_reason,
                    )
                    return None
                self._tectonic_cache_error = ""
                self._tectonic_supports_x = supports_x
                self._detect_done = False  # vendored binary is now discoverable
                self._log(
                    "tectonic_install",
                    status="cached",
                    path=str(existing),
                    version=self._tectonic_version,
                )
                return existing

        version = self._tectonic_version
        tmp_root: Path | None = None
        archive: Path | None = None
        reason = "unknown error"
        try:
            vendor_dir = self.vendor_dir()
            vendor_dir.mkdir(parents=True, exist_ok=True)
            # NOTE: deliberately *not* `tempfile.mkdtemp`.  Under the DSH Windows
            # file sandbox, directories produced by the `tempfile` module are
            # created in a state where any further write inside them is denied
            # (WinError 5), which breaks extraction.  An explicit uuid-suffixed
            # mkdir inside our own vendor dir is both writable and safe.
            tmp_root = self._make_scratch_dir(vendor_dir)
            extract_dir = tmp_root / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)

            last_error = "no candidate archive available"
            downloaded = False
            for name, url in self._candidate_archives(version):
                archive = tmp_root / name
                try:
                    self._download(url, archive)
                    downloaded = True
                    break
                except urllib.error.HTTPError as exc:
                    last_error = f"HTTP {exc.code} for {url}"
                    continue
                except urllib.error.URLError as exc:
                    last_error = f"network error for {url}: {exc.reason}"
                    break  # no point trying the fallback name without network
                except Exception as exc:
                    last_error = f"{type(exc).__name__} downloading {url}: {exc}"
                    break

            if not downloaded or archive is None:
                reason = last_error
                raise LatexError(reason)

            if archive.name.endswith(".zip"):
                _safe_extract_zip(archive, extract_dir)
            else:
                _safe_extract_tar(archive, extract_dir)

            found = self._find_binary(extract_dir)
            if found is None:
                raise LatexError(f"no tectonic binary inside {archive.name}")

            target = vendor_dir / ("tectonic.exe" if found.suffix.lower() == ".exe" else "tectonic")
            if target.exists():
                try:
                    target.unlink()
                except OSError:
                    pass
            shutil.copy2(found, target)
            try:
                os.chmod(target, 0o755)
            except OSError:
                pass

            usable, supports_x = self._probe_tectonic(str(target))
            if not usable:
                try:
                    target.unlink()
                except OSError:
                    pass
                raise LatexError(
                    f"downloaded tectonic at {target} failed `--version` verification"
                )

            # ---- 可用性预检 -------------------------------------------------
            # tectonic 下载并解压成功，**不等于**它能编译：它第一次运行需要把
            # TeX bundle 落盘到缓存目录，并解包到自己的临时目录。在受限沙箱里那两次
            # 写入都可能被拒绝，报错是一句几乎无法归因的 `os error 5`。
            # 注意：本模块已经把缓存指到工作区内（见 _tectonic_cache_preflight），
            # 因此失败**不是**"缓存位置不对"，而是沙箱/ACL 拒绝在新建目录里写入。
            # 与其让用户在 s7 里看到一条晦涩的失败，不如在这里就把原因查清并记下来。
            preflight_ok, preflight_reason = self._tectonic_cache_preflight(str(target))
            if not preflight_ok:
                self._tectonic_cache_error = preflight_reason
                self._log(
                    "tectonic_install",
                    status="installed_but_unusable",
                    path=str(target),
                    version=version,
                    reason=preflight_reason,
                )
                return None
            self._tectonic_cache_error = ""

            self._tectonic_supports_x = supports_x
            self._detect_done = False
            self._log(
                "tectonic_install",
                status="ok",
                path=str(target),
                version=version,
                supports_x_compile=supports_x,
            )
            return target

        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}" if not isinstance(exc, LatexError) else str(exc)
            self._log("tectonic_install", status="failed", version=version, reason=reason)
            return None
        finally:
            # clean up the partial download + temp extraction dir
            if archive is not None:
                try:
                    if archive.exists():
                        archive.unlink()
                except OSError:
                    pass
            if tmp_root is not None:
                self._drop_scratch_dir(tmp_root)

    # -- compile ----------------------------------------------------------- #
    def _tectonic_cache_preflight(self, binary: str) -> tuple[bool, str]:
        """真实编译一个 3 行的最小文档，验证 tectonic 能否写出它的 bundle 缓存。

        返回 ``(可用, 原因)``。原因在失败时可直接展示给用户，包含**具体缺什么权限**。

        代价是首次约 5-20 秒（要下载/索引 bundle），因此结果会被缓存——
        这个检查比「让整条管线在 s7 编译一篇长论文时才发现同样的权限问题」便宜得多。
        """
        cache_dir = self.vendor_dir() / "tectonic_cache"
        scratch: Path | None = None
        try:
            scratch = self._make_scratch_dir(self.vendor_dir())
            tex = scratch / "preflight.tex"
            tex.write_text(
                "\\documentclass{article}\n"
                "\\begin{document}\n"
                "preflight\n"
                "\\end{document}\n",
                encoding="utf-8",
            )
            env = dict(os.environ)
            # 尽量把缓存留在工作区内；即使如此，某些沙箱仍会拒绝 tectonic 的
            # 内部临时文件写入，所以这里仍然要真实跑一遍而不是只看环境变量。
            env["TECTONIC_CACHE_DIR"] = str(cache_dir)
            proc = subprocess.run(
                [binary, "-X", "compile", str(tex), "--outdir", str(scratch)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                cwd=str(scratch),
                env=env,
            )
            pdf = scratch / "preflight.pdf"
            if proc.returncode == 0 and pdf.exists() and pdf.stat().st_size > 0:
                return True, ""
            blob = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            low = blob.lower()
            if "os error 5" in low or "access is denied" in low or "拒绝访问" in blob:
                # 诊断文案必须与实现一致。这里**不能**说"缓存默认在用户 profile 下"：
                # 本方法已经把 TECTONIC_CACHE_DIR 指到工作区内（见上方 env 赋值），
                # 而失败依旧发生——真实原因是沙箱/ACL 拒绝进程在**新建目录**里写入，
                # 与 profile 无关。写错归因会把人引向"去放开 profile 权限"这条死路。
                return False, (
                    "tectonic is installed and runs, but the OS denied a write it needs "
                    "(os error 5 / access denied). Its bundle cache was already redirected "
                    f"into the project ({cache_dir}) and the denial persisted, so this is not "
                    "a cache-location problem: the sandbox/ACL is blocking writes into newly "
                    "created directories (the same denial hits tectonic's own .tectonic_dl_* "
                    f"scratch dirs under {self.vendor_dir()}). "
                    "Fix options: run outside the sandbox; or use a system TeX distribution "
                    "(pdflatex/xelatex); or compile the generated paper/ on Overleaf."
                )
            return False, f"tectonic preflight compile failed: {blob[-600:] or 'no output'}"
        except subprocess.TimeoutExpired:
            return False, "tectonic preflight compile timed out after 180s"
        except Exception as exc:  # pragma: no cover - 预检本身绝不抛
            return False, f"tectonic preflight failed: {type(exc).__name__}: {exc}"
        finally:
            if scratch is not None:
                self._drop_scratch_dir(scratch)

    def cache_block_reason(self) -> str:
        """上一轮探测发现的「已安装但不可用」原因（可能为空串）。"""
        return str(getattr(self, "_tectonic_cache_error", "") or "")

    def _resolve_tex(self, tex_file) -> Path:
        p = Path(tex_file)
        if not p.is_absolute():
            p = self.workdir / p
        return p

    def _usable_tectonic(self) -> bool:
        """tectonic 是否**真的**可用（二进制存在 + 能完成一次真实编译）。

        ``--version`` 能跑不等于能编译：它第一次运行要把 TeX bundle 落盘并把内容
        解包到自己的临时目录，受限沙箱/ACL 会拒绝这类新建目录里的写入。只探测
        ``--version`` 会让编译阶段拿到一个「看起来可用」的引擎，然后在真正编译时
        以一句 ``os error 5`` 失败——这正是把不可诊断的失败提前成可诊断能力缺口的地方。
        """
        try:
            usable, found = self._probe_engine("tectonic")
        except Exception:
            return False
        if not usable or not found:
            return False
        try:
            ok, reason = self._tectonic_cache_preflight(str(found))
        except Exception as exc:  # pragma: no cover - 预检绝不抛
            ok, reason = False, f"{type(exc).__name__}: {exc}"
        self._tectonic_cache_error = "" if ok else reason
        return ok

    def _resolve_engine(self, explicit: str | None) -> str | None:
        if explicit:
            name = str(explicit).strip().lower()
            if name in ("none", ""):
                return None
            if name == "tectonic":
                return "tectonic" if self._usable_tectonic() else None
            try:
                usable, _ = self._probe_engine(name)
            except Exception:
                usable = False
            return name if usable else None

        cfg_engine = self.cfg_engine
        if cfg_engine in ("none",):
            # 用户显式禁用编译：这是明确的意图，不能用自动探测覆盖它。
            return None
        if cfg_engine:
            if cfg_engine == "tectonic":
                if self._usable_tectonic():
                    return "tectonic"
                if self._auto_install:
                    installed = self.install_tectonic()
                    if installed is not None and self._usable_tectonic():
                        return "tectonic"
            else:
                try:
                    usable, _ = self._probe_engine(cfg_engine)
                except Exception:
                    usable = False
                if usable:
                    return cfg_engine
            # 指定了引擎但它不可用 —— **不要**回退到自动探测。
            #
            # 早期实现在这里 `return self.detect()`，于是 `engine="xelatex"` 而机器上
            # 只有 tectonic 时会**静默改用 tectonic**：编译成功、产出 PDF、没有任何提示，
            # 但用户指定的引擎被完全忽略。对排版而言这不是等价替换（二者对字体、
            # 宏包与 Unicode 的处理不同），用户会拿到一份自己没要求的产物却看不出区别。
            # 现在如实返回 None，让 compile() 报出「指定的引擎不可用」及原因。
            self._tectonic_cache_error = self._tectonic_cache_error or (
                f"configured LaTeX engine '{cfg_engine}' is not usable on this machine; "
                "refusing to silently substitute a different engine "
                "(set engine='' to allow auto-detection, or 'none' to disable compiling)"
            )
            return None
        return self.detect()

    @staticmethod
    def _no_engine_result(reason: str) -> CompileResult:
        return CompileResult(
            ok=False,
            pdf=None,
            log=reason,
            errors=[reason],
            warnings=[],
            engine="none",
            duration=0.0,
        )

    def compile(
        self,
        tex_file: Path,
        runs: int = 2,
        timeout: int = 300,
        engine: str | None = None,
    ):
        """Compile a .tex file to PDF.  Missing .tex raises; everything else degrades."""
        resolved_tex = self._resolve_tex(tex_file)
        if not resolved_tex.exists() or not resolved_tex.is_file():
            raise LatexError(f"tex file not found: {resolved_tex}")

        if self.cfg_engine == "none" and not engine:
            self._log("compile_fail", engine="none", reason="no LaTeX engine available")
            return self._no_engine_result("no LaTeX engine available")

        chosen = self._resolve_engine(engine)
        if chosen is None:
            # 若是「已安装但不可用」，错误串必须带上具体原因——否则调用方只能看到
            # 一句 "no engine available"，而用户其实什么都不用装，只要放开一次写权限。
            block = self.cache_block_reason()
            reason = (
                f"no usable LaTeX engine: {block}"
                if block
                else "no LaTeX engine available"
            )
            self._log("compile_fail", engine="none", reason=reason[:500])
            return self._no_engine_result(reason)

        executable = self.resolve_executable(chosen)
        if executable is None:
            if chosen == "tectonic" and self._auto_install:
                installed = self.install_tectonic()
                if installed is not None:
                    executable = str(installed)
                    chosen = "tectonic"
            if executable is None:
                msg = f"no LaTeX engine available (engine {chosen} not usable)"
                self._log("compile_fail", engine=chosen, reason=msg)
                return self._no_engine_result(msg)

        stem = resolved_tex.stem
        workdir = resolved_tex.parent if str(resolved_tex.parent) else self.workdir
        log_chunks: list[str] = []
        duration = 0.0
        timed_out = False
        started = time.monotonic()

        try:
            if chosen == "tectonic":
                duration += self._compile_tectonic(
                    executable, resolved_tex, workdir, int(timeout), log_chunks
                )
            else:
                duration += self._compile_classic(
                    chosen, executable, resolved_tex, workdir,
                    int(runs), int(timeout), log_chunks,
                )
        except subprocess.TimeoutExpired:
            timed_out = True
            log_chunks.append(f"[timeout] {chosen} exceeded {timeout}s and was killed")
        except OSError as exc:
            log_chunks.append(f"[error] failed to run {chosen}: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            log_chunks.append(f"[error] {type(exc).__name__}: {exc}")

        duration = time.monotonic() - started
        engine_output = "\n".join(log_chunks)

        # read back the .log file the engine left behind
        log_path = workdir / f"{stem}.log"
        if log_path.is_file():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                log_chunks.append(f"----- {log_path.name} -----\n{text}")
            except OSError as exc:
                log_chunks.append(f"[warn] could not read {log_path}: {exc}")

        log_text = "\n".join(chunk for chunk in log_chunks if chunk)

        if timed_out:
            errors = [f"compile timed out after {int(timeout)}s"]
            warnings = extract_warnings(log_text)
            result = CompileResult(
                ok=False,
                pdf=None,
                log=log_text,
                errors=errors,
                warnings=warnings,
                engine=chosen,
                duration=duration,
            )
            self._log(
                "compile_fail",
                engine=chosen,
                reason=errors[0],
                duration=round(duration, 3),
            )
            return result

        pdf = self._locate_pdf(workdir, stem, started)
        ok = pdf is not None and pdf.stat().st_size > 0

        errors = extract_errors(log_text)
        warnings = extract_warnings(log_text)

        result = CompileResult(
            ok=ok,
            pdf=pdf if ok else None,
            log=log_text,
            errors=errors,
            warnings=warnings,
            engine=chosen,
            duration=duration,
        )

        if ok:
            self._log(
                "compile_run",
                ok=True,
                engine=chosen,
                duration=round(duration, 3),
                pdf=str(result.pdf),
                errors=errors[:5],
            )
        else:
            if not errors:
                tail_line = _last_output_line(engine_output)
                detail = f" (engine said: {tail_line})" if tail_line else ""
                errors.append(
                    f"{chosen} produced no PDF{detail}; see log for the engine output"
                    f" (tex={resolved_tex.name})"
                )
                result.errors = errors
            # a failed non-interactive run often prints its reason to stdout as a
            # line starting with "!" or with the error name; surface it as well
            if not any(e.startswith("!") or " error:" in e or e.startswith("error:") for e in errors):
                tail_line = _last_output_line(engine_output)
                if tail_line and tail_line not in " ".join(errors):
                    errors.append(tail_line)
                    result.errors = errors
            self._log(
                "compile_run",
                ok=False,
                engine=chosen,
                duration=round(duration, 3),
                pdf="",
                errors=errors[:5],
            )
            self._log(
                "compile_fail",
                engine=chosen,
                reason=errors[0] if errors else "no PDF produced",
                duration=round(duration, 3),
            )
        return result

    # -- engine drivers ---------------------------------------------------- #
    def _compile_tectonic(
        self,
        executable: str,
        tex: Path,
        workdir: Path,
        timeout: int,
        log_chunks: list[str],
    ) -> float:
        stem = tex.stem
        started = time.monotonic()

        if not self._detect_done and not self._tectonic_supports_x:
            _, supports_x = self._probe_tectonic(executable)
            self._tectonic_supports_x = supports_x

        common = ["--keep-logs", "--outdir", str(workdir)]
        attempts: list[list[str]] = []
        if self._tectonic_supports_x:
            # tectonic >= 0.15 uses the `-X compile` sub-command form
            attempts.append([executable, "-X", "compile", str(tex), *common])
            attempts.append([executable, "-X", "compile", str(tex), *common, "--print"])
        attempts.append([executable, *common, str(tex)])

        last_error = ""
        for index, cmd in enumerate(attempts):
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    cwd=str(workdir),
                    # 必须与预检用同一个缓存目录，否则预检通过、生产失败
                    env=_child_env(self.vendor_dir() / "tectonic_cache"),
                )
            except subprocess.TimeoutExpired:
                raise
            except OSError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            out = _decode(proc.stdout)
            err = _decode(proc.stderr)
            combined = out + (("\n" + err) if err else "")
            header = (
                f"$ {' '.join(cmd)}\n[note] tectonic caches packages; the first "
                f"compile downloads them and can take several minutes."
            )
            log_chunks.append(header + "\n" + combined)
            if proc.returncode == 0:
                return time.monotonic() - started

            last_line = (err.strip().splitlines() or [""])[-1] or (
                (out.strip().splitlines() or [""])[-1]
            )
            last_error = f"exit {proc.returncode}: {last_line}"

            # only retry with the next invocation form when this one looks like an
            # interface mismatch rather than a genuine TeX failure
            is_last = index == len(attempts) - 1
            if not _looks_like_interface_mismatch(combined) and not is_last:
                break

        if last_error:
            log_chunks.append(f"[tectonic] compile failed ({last_error})")
        return time.monotonic() - started

    def _compile_classic(
        self,
        engine: str,
        executable: str,
        tex: Path,
        workdir: Path,
        runs: int,
        timeout: int,
        log_chunks: list[str],
    ) -> float:
        stem = tex.stem
        started = time.monotonic()
        runs = max(1, int(runs))
        bibtex_ran = False

        for i in range(runs):
            cmd = [
                executable,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                tex.name,
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=str(workdir),
                env=_child_env(),
            )
            log_chunks.append(
                f"$ {' '.join(cmd)}  (run {i + 1}/{runs}, exit {proc.returncode})\n"
                + _decode(proc.stdout)
                + (("\n" + _decode(proc.stderr)) if proc.stderr else "")
            )

            if i == 0 and not bibtex_ran:
                aux = workdir / f"{stem}.aux"
                if self._aux_needs_bibtex(aux) and self.bibtex_available():
                    bib = subprocess.run(
                        ["bibtex", stem],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=timeout,
                        cwd=str(workdir),
                        env=_child_env(),
                    )
                    bibtex_ran = True
                    log_chunks.append(
                        f"$ bibtex {stem}  (exit {bib.returncode})\n"
                        + _decode(bib.stdout)
                        + (("\n" + _decode(bib.stderr)) if bib.stderr else "")
                    )

        return time.monotonic() - started

    @staticmethod
    def _aux_needs_bibtex(aux: Path) -> bool:
        if not aux.is_file():
            return False
        try:
            text = aux.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return ("\\citation{" in text) or ("\\bibdata{" in text)

    @staticmethod
    def _locate_pdf(workdir: Path, stem: str, since: float) -> Path | None:
        primary = workdir / f"{stem}.pdf"
        if primary.is_file() and primary.stat().st_size > 0:
            return primary
        # fallback: a PDF freshly written into workdir by this compile
        newest: Path | None = None
        newest_mtime = -1.0
        try:
            candidates = list(workdir.glob("*.pdf"))
        except OSError:
            return None
        for candidate in candidates:
            try:
                st = candidate.stat()
            except OSError:
                continue
            if st.st_size <= 0:
                continue
            # allow a small clock slack; the file must not predate this compile
            if st.st_mtime + 5.0 < since:
                continue
            if st.st_mtime > newest_mtime:
                newest_mtime = st.st_mtime
                newest = candidate
        if newest is not None:
            return newest
        if primary.is_file():
            return primary
        return None
