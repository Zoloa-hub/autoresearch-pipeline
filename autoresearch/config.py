"""配置层（CONTRACTS §1，已冻结）.

优先级：显式 overrides > 环境变量 AUTORESEARCH_* > .env 文件 > 默认值。

零强制第三方依赖：自带极简 .env 解析器，不依赖 python-dotenv。
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PROJECT_ROOT",
    "WORKSPACE_ROOT",
    "DEFAULT_RUNS_DIR",
    "LLMConfig",
    "SandboxConfig",
    "RetrieveConfig",
    "CompileConfig",
    "AutoResearchConfig",
    "load_dotenv",
    "load_config",
    "ensure_dirs",
    "coerce_value",
    "ENV_KEYS",
]

_log = logging.getLogger("autoresearch.config")

# --------------------------------------------------------------------------- #
# 路径常量
# --------------------------------------------------------------------------- #

#: autoresearch/ 包目录，绝对路径
PROJECT_ROOT: Path = Path(__file__).resolve().parent
#: 上一级工作区目录
WORKSPACE_ROOT: Path = PROJECT_ROOT.parent
#: 默认运行根目录
DEFAULT_RUNS_DIR: Path = WORKSPACE_ROOT / ".autoresearch" / "runs"

# --------------------------------------------------------------------------- #
# 数据类型
# --------------------------------------------------------------------------- #


@dataclass
class LLMConfig:
    provider: str = "openai"  # openai|deepseek|ollama|mock
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None  # 只从环境变量读，绝不落盘
    temperature: float = 0.3
    max_tokens: int = 8192
    timeout: float = 120.0
    max_retries: int = 3
    json_repair_attempts: int = 2


@dataclass
class SandboxConfig:
    backend: str = "subprocess"  # subprocess|docker
    timeout: int = 900
    memory_mb: int = 4096
    cpus: float = 2.0
    allow_network: bool = True
    docker_image: str = "python:3.11-slim"
    extra_deny: list[str] = field(default_factory=list)
    #: 单文件大小上限（MB）。真实训练会写 checkpoint，一次几十 MB 很正常；
    #: 但一个失控的循环能把磁盘写满，所以给一个宽松的上限而不是不设限。
    #: 0 表示不限制。仅 POSIX 生效（Windows 无对应 rlimit）。
    max_file_mb: int = 4096


@dataclass
class RetrieveConfig:
    sources: list[str] = field(
        default_factory=lambda: ["arxiv", "s2", "openalex", "crossref"]
    )
    max_results_per_query: int = 8
    cache_dir: Path | None = None
    offline: bool = False
    timeout: float = 20.0
    mailto: str = "autoresearch@example.org"


@dataclass
class CompileConfig:
    engine: str = "tectonic"  # tectonic|pdflatex|xelatex|none
    auto_install_tectonic: bool = True
    tectonic_version: str = "0.15.0"
    venv_bin_dir: Path | None = None


@dataclass
class AutoResearchConfig:
    run_id: str = ""
    runs_dir: Path = DEFAULT_RUNS_DIR
    direction: str = ""
    venue: str = "NeurIPS"
    language: str = "zh"  # zh|en，控制产出报告语言
    seed: int = 0
    max_review_rounds: int = 3
    max_debug_rounds: int = 4
    max_ideas: int = 6
    keep_top_ideas: int = 3
    resume: bool = False
    dry_run: bool = False
    #: 实验后端适配器。``None``/``default`` 用内置合成任务；
    #: 也可以是 ``path/to/adapter.py``、``pkg.module:Class``、entry point 名，
    #: 或内置名（``synthetic-toy`` / ``script-wrapper``）。
    experiment_adapter: str | None = None
    #: 传给适配器构造函数的参数（``--adapter-arg key=value``）。
    adapter_params: dict[str, Any] = field(default_factory=dict)
    #: 单次运行最多跑几个「臂」（主对照 + 消融格点）。实际值取
    #: ``min(这里的值, 适配器自己的 max_variants)``——代价高的后端可自行收紧，
    #: 但不可能把这个上限抬高到配置之上。
    max_variants: int = 6
    #: 单次参数扫描的运行数上限。**与 max_variants 分开**：
    #: 变体预算默认 2（baseline/method），而一次材料参数扫描
    #: 天然要 6-18 组；共用一个旋钮会让"想跑扫描"变成
    #: "必须先把变体上限调大"，语义混淆。
    max_sweep_runs: int = 24
    llm: LLMConfig = field(default_factory=LLMConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)

    # -- 序列化 ---------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """JSON 可序列化（dataclass -> dict，Path -> str）。"""
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "AutoResearchConfig":
        """重建（接受嵌套 dict；Path/str 均可）。"""
        if not isinstance(d, dict):
            raise TypeError(f"from_dict 需要 dict，收到 {type(d).__name__}")
        nested = {
            "llm": LLMConfig,
            "sandbox": SandboxConfig,
            "retrieve": RetrieveConfig,
            "compile": CompileConfig,
        }
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, val in d.items():
            if key not in known:
                _log.debug("from_dict: 忽略未知键 %r", key)
                continue
            if key in nested:
                sub_t = nested[key]
                if val is None:
                    kwargs[key] = sub_t()
                elif isinstance(val, sub_t):
                    kwargs[key] = val
                elif isinstance(val, dict):
                    kwargs[key] = _build_dataclass(sub_t, val)
                else:
                    raise TypeError(f"{key} 需要 dict 或 {sub_t.__name__}")
            elif key in _PATH_FIELDS[cls.__name__]:
                kwargs[key] = Path(val).expanduser() if val else val
            else:
                kwargs[key] = val
        return cls(**kwargs)

    def redacted_dict(self) -> dict[str, Any]:
        """给 CLI 打印用：api_key 非空时替换为 ***。"""
        d = self.to_dict()
        llm = d.get("llm")
        if isinstance(llm, dict) and llm.get("api_key"):
            llm["api_key"] = "***"
        return d

    # -- 便捷 ------------------------------------------------------------ #

    @property
    def run_dir(self) -> Path:
        """本次运行目录（run_id 为空时退化为 runs_dir）。"""
        return (Path(self.runs_dir) / self.run_id) if self.run_id else Path(self.runs_dir)


# --------------------------------------------------------------------------- #
# dataclass 辅助
# --------------------------------------------------------------------------- #

_PATH_FIELDS: dict[str, set[str]] = {
    "AutoResearchConfig": {"runs_dir"},
    "RetrieveConfig": {"cache_dir"},
    "CompileConfig": {"venv_bin_dir"},
}


def _plain(val: Any) -> Any:
    """递归转成 JSON 友好类型（dataclass -> dict，Path -> str）。"""
    if isinstance(val, Path):
        return str(val)
    if is_dataclass(val) and not isinstance(val, type):
        return {f.name: _plain(getattr(val, f.name)) for f in fields(val)}
    if isinstance(val, dict):
        return {str(k): _plain(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_plain(v) for v in val]
    return val


def _build_dataclass(cls_: type, data: dict) -> Any:
    path_fields = _PATH_FIELDS.get(cls_.__name__, set())
    known = {f.name for f in fields(cls_)}
    kwargs: dict[str, Any] = {}
    for key, val in data.items():
        if key not in known:
            _log.debug("%s: 忽略未知键 %r", cls_.__name__, key)
            continue
        if key in path_fields:
            kwargs[key] = Path(val).expanduser() if val else val
        else:
            kwargs[key] = val
    return cls_(**kwargs)


# --------------------------------------------------------------------------- #
# 健壮强制类型转换
# --------------------------------------------------------------------------- #

_TRUE = {"true", "1", "yes", "y", "on", "是", "真"}
_FALSE = {"false", "0", "no", "n", "off", "否", "假"}
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")


def _coerce_bool(raw: Any, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise ValueError(f"不是合法布尔值: {raw!r}")


def coerce_value(raw: Any, default: Any, key: str = "") -> Any:
    """把字符串环境变量转成与 default 同类型的值；失败则 warning + 默认值。

    永不抛异常 —— 坏值必须回退到默认值而不是让配置崩掉。
    """
    if raw is None:
        return default
    # 非字符串且已经是目标类型，直接用
    if isinstance(default, bool):
        try:
            return _coerce_bool(raw, bool(default))
        except Exception:
            _log.warning("配置项 %s=%r 不是合法布尔值，回退默认值 %r", key, raw, default)
            return default

    if isinstance(raw, str):
        s = raw.strip()
        if isinstance(default, int) and not isinstance(default, bool):
            if _INT_RE.match(s):
                return int(s)
            if _FLOAT_RE.match(s):
                return int(float(s))
            _log.warning("配置项 %s=%r 不是合法整数，回退默认值 %r", key, raw, default)
            return default
        if isinstance(default, float):
            if _FLOAT_RE.match(s):
                return float(s)
            _log.warning("配置项 %s=%r 不是合法浮点数，回退默认值 %r", key, raw, default)
            return default
        if default is None:
            return s
        if isinstance(default, (list, tuple, set, dict)):
            # 逗号/分号分隔
            parts = [p.strip() for p in re.split(r"[,;]", s) if p.strip()]
            if isinstance(default, dict):
                return default
            if isinstance(default, tuple):
                return tuple(parts)
            return parts
        if isinstance(default, Path):
            return Path(s).expanduser()
        if isinstance(default, str):
            return s

    # 未在字符串分支处理：类型匹配就直接返回
    if default is None:
        return raw
    expected = type(default)
    if isinstance(raw, expected):
        return raw
    try:
        if isinstance(default, (list, tuple)) and isinstance(raw, (list, tuple)):
            return expected(raw)
        return expected(raw)
    except Exception:
        _log.warning("配置项 %s=%r 无法转换为 %s，回退默认值 %r", key, raw, expected.__name__, default)
        return default


# --------------------------------------------------------------------------- #
# .env 解析
# --------------------------------------------------------------------------- #

_ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*)$")
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _strip_quotes(val: str) -> str:
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        return val[1:-1]
    return val


def _expand(value: str, known: dict[str, str]) -> str:
    def repl(m: "re.Match[str]") -> str:
        name = m.group(1)
        if name in known:
            return known[name]
        return os.environ.get(name, "")

    return _ENV_REF_RE.sub(repl, value)


def _parse_env_text(text: str, known: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE_RE.match(line)
        if not m:
            _log.debug(".env: 跳过无法解析的行: %r", line)
            continue
        key, value = m.group(1), m.group(2).strip()
        # 行尾注释（仅当不在引号内）
        if value[:1] not in ("'", '"'):
            hash_pos = value.find(" #")
            if hash_pos >= 0:
                value = value[:hash_pos].strip()
        value = _expand(_strip_quotes(value), {**known, **out})
        out[key] = value
    return out


#: 设为 1/true 时完全忽略 `.env` 文件，只认真实环境变量与显式入参。
#: 测试需要这个开关：开发机上的 `.env`（真实密钥、真实 provider）会泄漏进
#: 测试进程，让「provider 默认值」这类断言依赖开发者的本地配置而随机失败。
DOTENV_DISABLE_ENV = "AUTORESEARCH_NO_DOTENV"


def _dotenv_disabled() -> bool:
    return str(os.environ.get(DOTENV_DISABLE_ENV, "")).strip().lower() in ("1", "true", "yes", "on")


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """极简 .env 解析。

    搜索顺序：显式 path -> WORKSPACE_ROOT/.env -> PROJECT_ROOT/.env -> cwd/.env。
    解析 KEY=VALUE（允许 `export ` 前缀），跳过 # 注释与空行，剥离成对引号，
    展开 ${VAR}（来自已解析值 + os.environ）。
    返回解析结果 dict；同时 `os.environ.setdefault` 每个键
    （不覆盖真实存在的环境变量）。

    若 ``AUTORESEARCH_NO_DOTENV=1`` 且未显式传入 ``path``，直接返回空 dict
    （测试隔离用，见 :data:`DOTENV_DISABLE_ENV`）。
    """
    if path is None and _dotenv_disabled():
        return {}

    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        for base in (WORKSPACE_ROOT, PROJECT_ROOT, Path.cwd()):
            candidates.append(Path(base) / ".env")

    parsed: dict[str, str] = {}
    for cand in candidates:
        try:
            if not cand.is_file():
                continue
            text = cand.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - IO 异常
            _log.warning("读取 .env 失败 %s: %s", cand, exc)
            continue
        file_vals = _parse_env_text(text, parsed)
        parsed.update(file_vals)
        if path is not None:
            break

    for key, value in parsed.items():
        # 只补齐，不覆盖真实环境变量
        if key not in os.environ:
            os.environ[key] = value
    return parsed


# --------------------------------------------------------------------------- #
# 环境变量 -> 配置
# --------------------------------------------------------------------------- #

#: 单个 AUTORESEARCH_* 环境变量 -> 配置字段路径
ENV_KEYS: dict[str, tuple[str, ...]] = {
    "LLM_PROVIDER": ("llm", "provider"),
    "MODEL": ("llm", "model"),
    "BASE_URL": ("llm", "base_url"),
    "API_KEY": ("llm", "api_key"),
    "TEMPERATURE": ("llm", "temperature"),
    "MAX_TOKENS": ("llm", "max_tokens"),
    "RUNS_DIR": ("runs_dir",),
    "VENUE": ("venue",),
    "LANGUAGE": ("language",),
    "SANDBOX_BACKEND": ("sandbox", "backend"),
    "SANDBOX_TIMEOUT": ("sandbox", "timeout"),
    "OFFLINE": ("retrieve", "offline"),
    "MAX_REVIEW_ROUNDS": ("max_review_rounds",),
    "MAX_DEBUG_ROUNDS": ("max_debug_rounds",),
    "SEED": ("seed",),
    "TECTONIC_VERSION": ("compile", "tectonic_version"),
    "MAILTO": ("retrieve", "mailto"),
    "EXPERIMENT_ADAPTER": ("experiment_adapter",),
    "MAX_VARIANTS": ("max_variants",),
    #: 参数扫描的运行数预算。**与 MAX_VARIANTS 分开**：前者是"跑几个对照臂"，
    #: 后者是"扫多少个参数格点"，两者的合理量级不同（2-6 vs 6-60）。
    "MAX_SWEEP_RUNS": ("max_sweep_runs",),
}

_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        # deepseek-chat 支持 64k 输出；给足预算，避免长 JSON（综述/分析/章节）
        # 被 4096 上限截断——截断后的 JSON 一定解析失败，整段工作白做。
        "max_tokens": 32768,
    },
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "llama3.1"},
}

#: provider -> 读取 API key 的环境变量名
PROVIDER_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "ollama": "",  # 本地服务无需 key
    "mock": "",
}

#: provider -> 备用 base_url 环境变量名
PROVIDER_BASE_URL_ENV: dict[str, str] = {
    "openai": "OPENAI_BASE_URL",
}


def _env_name(suffix: str) -> str:
    return f"AUTORESEARCH_{suffix}"


def _set_by_path(d: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    if len(path) == 1:
        d[path[0]] = value
    else:
        sub = d.setdefault(path[0], {})
        if isinstance(sub, dict):
            sub[path[1]] = value


def _resolve_api_key(provider: str, explicit: str | None) -> str | None:
    """按 provider 映射 API key：AUTORESEARCH_API_KEY > provider 专属变量。"""
    env_api = os.environ.get(_env_name("API_KEY"))
    if env_api:
        return env_api
    if explicit:
        return explicit
    var = PROVIDER_API_KEY_ENV.get(provider, "")
    if var and os.environ.get(var):
        return os.environ[var]
    if provider == "ollama":
        return "ollama"
    return None


#: provider 推断顺序：谁有 Key 就用谁。
#: ``openai`` 放最后：``OPENAI_API_KEY`` 常常是别处遗留的通用变量，
#: 显式的 DEEPSEEK_API_KEY 应该优先。
_PROVIDER_INFERENCE_ORDER: tuple[str, ...] = ("deepseek", "anthropic", "openai")


def infer_provider(explicit: str | None = None) -> str:
    """在调用方没指定 provider 时，按「哪个 Key 存在」推断。

    为什么需要它：配置里只写 ``DEEPSEEK_API_KEY=sk-...`` 是最自然的用法，
    此时如果 provider 静默留在默认值 ``openai``，请求会带着 ``EMPTY`` key
    打到 OpenAI 并返回 401——报错信息指向 OpenAI，而用户明明配的是 DeepSeek，
    这个误导性极强（本项目实测踩过）。

    ``explicit`` 非空时原样返回（尊重调用方的选择，即使是 ``mock``）。
    """
    if explicit:
        return str(explicit).strip().lower()
    for provider in _PROVIDER_INFERENCE_ORDER:
        env_name = PROVIDER_API_KEY_ENV.get(provider, "")
        if env_name and str(os.environ.get(env_name) or "").strip():
            return provider
    return LLMConfig.provider


def _apply_provider_defaults(overrides: dict[str, Any]) -> None:
    """provider 相关的 base_url / model / api_key 默认值。"""
    llm = overrides.setdefault("llm", {})
    # 没指定 provider 时按存在的 Key 推断，而不是静默用默认的 openai。
    # 注意：此处读的是 os.environ，而 load_dotenv 已把 .env 的键 setdefault
    # 进 os.environ，因此「只在 .env 里写了 DEEPSEEK_API_KEY」也能被正确识别。
    llm["provider"] = infer_provider(llm.get("provider"))
    provider = llm["provider"]
    pdefaults = _PROVIDER_DEFAULTS.get(provider, {})

    base_url = llm.get("base_url")
    if base_url is None:
        env_var = PROVIDER_BASE_URL_ENV.get(provider, "")
        if env_var and os.environ.get(env_var):
            base_url = os.environ[env_var]
    if base_url is None and "base_url" in pdefaults:
        base_url = pdefaults["base_url"]
    llm["base_url"] = base_url

    model = llm.get("model")
    # 只有调用方/环境变量**没有给出** model 时，才用 provider 默认值。
    # 之前这里还额外把「model 恰好等于全局默认值」也当成未指定，导致
    # 一个 `.env` 里的 `AUTORESEARCH_MODEL=deepseek-chat` 会被复制到
    # ollama/mock 等所有 provider 上——实测把 ollama 的默认模型污染成了
    # deepseek-chat。provider 默认值不得跨 provider 泄漏。
    if not model and "model" in pdefaults:
        llm["model"] = pdefaults["model"]
    elif not model:
        llm["model"] = LLMConfig.model

    llm["api_key"] = _resolve_api_key(provider, llm.get("api_key"))

    # ``max_tokens`` 同理：provider 默认值只在调用方没给时生效。
    if not llm.get("max_tokens") and "max_tokens" in pdefaults:
        llm["max_tokens"] = pdefaults["max_tokens"]


# --------------------------------------------------------------------------- #
# load_config
# --------------------------------------------------------------------------- #

_NESTED_KEYS = ("llm", "sandbox", "retrieve", "compile")


def load_config(**overrides: Any) -> AutoResearchConfig:
    """构造 AutoResearchConfig。

    优先级：显式 overrides > AUTORESEARCH_* 环境变量 > .env > 默认值。
    支持 `load_config(llm={...}, sandbox={...}, retrieve={...}, compile={...})`
    以及任意扁平字段覆盖（dotted 风格由调用方自行拆成 dict 传入）。
    """
    # 1) .env（只在显式传入 dotenv_path 时限定路径）
    dotenv_path = overrides.pop("dotenv_path", None)
    env_file_vals = load_dotenv(Path(dotenv_path) if dotenv_path else None)

    explicit: dict[str, Any] = {
        k: (dict(v) if isinstance(v, dict) else v) for k, v in overrides.items()
    }
    # 记录调用方显式给出的键：环境变量不得覆盖它们
    provided: set[tuple[str, ...]] = set()
    for path in ENV_KEYS.values():
        if len(path) == 1:
            if path[0] in explicit:
                provided.add(path)
        else:
            sub = explicit.get(path[0])
            if isinstance(sub, dict) and path[1] in sub:
                provided.add(path)

    # 2) AUTORESEARCH_* 环境变量（真实环境变量优先于 .env 文件）
    overrides: dict[str, Any] = {}
    for suffix, path in ENV_KEYS.items():
        if path in provided:
            continue  # 显式入参 > 环境变量
        raw = os.environ.get(_env_name(suffix))
        if raw is None:
            raw = env_file_vals.get(_env_name(suffix))
        if raw is None or raw == "":
            continue
        _set_by_path(overrides, path, raw)

    # 3) 显式入参合并进来（最高优先级）
    for key, val in explicit.items():
        if key in _NESTED_KEYS and isinstance(val, dict):
            sub = overrides.setdefault(key, {})
            if isinstance(sub, dict):
                sub.update(val)
            else:  # pragma: no cover - 类型异常时以显式为准
                overrides[key] = dict(val)
        else:
            overrides[key] = val

    # 4) 按目标默认值强制类型（环境变量是字符串；显式入参类型正确则原样保留）
    defaults = AutoResearchConfig()
    for path in ENV_KEYS.values():
        if len(path) == 1:
            key = path[0]
            if key in overrides:
                default_val = getattr(defaults, key)
                overrides[key] = (
                    overrides[key]
                    if path in provided
                    else coerce_value(overrides[key], default_val, key)
                )
        else:
            parent, key = path
            sub = overrides.get(parent)
            if isinstance(sub, dict) and key in sub:
                base = getattr(getattr(defaults, parent), key)
                if path not in provided:
                    sub[key] = coerce_value(sub[key], base, f"{parent}.{key}")

    # 5) provider 专属默认值
    _apply_provider_defaults(overrides)

    # 6) 组装
    base = defaults.to_dict()
    _merge_known(base, overrides, defaults)
    cfg = AutoResearchConfig.from_dict(base)

    # 7) 统一后处理：runs_dir / cache_dir 绝对化
    try:
        cfg.runs_dir = Path(cfg.runs_dir).expanduser()
        if cfg.retrieve.cache_dir is not None:
            cfg.retrieve.cache_dir = Path(cfg.retrieve.cache_dir).expanduser()
        if cfg.compile.venv_bin_dir is not None:
            cfg.compile.venv_bin_dir = Path(cfg.compile.venv_bin_dir).expanduser()
    except Exception as exc:
        _log.warning("路径规范化失败，保留原值: %s", exc)

    return cfg


def _merge_known(base: dict[str, Any], incoming: dict[str, Any], defaults: Any) -> None:
    """把 incoming 合并进 base（只填已知键，dict 递归合并）。"""
    top_known = {f.name for f in fields(AutoResearchConfig)}
    for key, val in incoming.items():
        if key not in top_known:
            _log.debug("load_config: 忽略未知覆盖键 %r", key)
            continue
        if key in _NESTED_KEYS and isinstance(val, dict):
            sub_defaults = getattr(defaults, key)
            sub_known = {f.name for f in fields(sub_defaults)}
            sub_base = base[key]
            for sub_key, sub_val in val.items():
                if sub_key not in sub_known:
                    _log.debug("load_config: 忽略未知覆盖键 %s.%s", key, sub_key)
                    continue
                sub_base[sub_key] = _plain(sub_val)
        else:
            base[key] = _plain(val)


# --------------------------------------------------------------------------- #
# 目录
# --------------------------------------------------------------------------- #


def ensure_dirs(cfg: AutoResearchConfig) -> None:
    """创建 runs_dir 以及 runs_dir/<run_id>（若 run_id 已设置）。"""
    runs_dir = Path(cfg.runs_dir).expanduser()
    runs_dir.mkdir(parents=True, exist_ok=True)
    if cfg.run_id:
        (runs_dir / cfg.run_id).mkdir(parents=True, exist_ok=True)
