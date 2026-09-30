"""日志与事件（CONTRACTS §2，已冻结）.

- ``get_logger`` / ``configure_logging``：标准 logging，幂等，无第三方依赖。
- ``EventLogger``：run_dir/events.jsonl，每行一个 JSON 对象，线程安全。
- ``ProgressReporter``：CLI 进度打印，纯 ASCII/box-drawing，无 rich / 无 ANSI。
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "get_logger",
    "configure_logging",
    "event_logger",
    "EventLogger",
    "ProgressReporter",
    "utc_now_iso",
]

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configure_lock = threading.Lock()
_configured_signature: tuple | None = None

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def utc_now_iso() -> str:
    """ISO8601 UTC 时间戳（秒精度，带 Z）。"""
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def get_logger(name: str) -> logging.Logger:
    """返回命名 logger（不重复添加 handler）。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return _LEVELS.get(str(level).strip().upper(), logging.INFO)


#: 第三方库在 INFO 级别刷屏，会把管线自己的进展完全淹掉（figures 每次导出
#: 都会让 fontTools.subset 打几十行字形表）。这些库的 WARNING 才值得看。
_NOISY_LOGGERS = (
    "fontTools", "matplotlib", "matplotlib.font_manager", "PIL",
    "urllib3", "httpx", "httpcore", "openai",
)


def silence_noisy_loggers(level: int = 30) -> None:
    """把第三方库的日志压到 WARNING 以上。幂等，可在任意时刻调用。"""
    import logging as _logging

    for name in _NOISY_LOGGERS:
        try:
            _logging.getLogger(name).setLevel(level)
        except Exception:  # pragma: no cover - 防御
            pass


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    quiet: bool = False,
) -> None:
    """幂等配置根 logger：先清空已有 handler，再加 console（+ 可选 file）。

    ``quiet=True`` 时控制台只输出 WARNING 及以上。
    """
    global _configured_signature

    lvl = _resolve_level(level)
    signature = (lvl, str(log_file) if log_file else None, bool(quiet))
    with _configure_lock:
        if _configured_signature == signature:
            return
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

        console = logging.StreamHandler(stream=sys.stderr)
        console.setLevel(logging.WARNING if quiet else lvl)
        console.setFormatter(formatter)
        root.addHandler(console)

        if log_file is not None:
            try:
                path = Path(log_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(path, encoding="utf-8")
                file_handler.setLevel(lvl)
                file_handler.setFormatter(formatter)
                root.addHandler(file_handler)
            except Exception as exc:  # pragma: no cover - IO 异常
                root.warning("无法创建日志文件 %s: %s", log_file, exc)

        root.setLevel(lvl)
        # 顺手压掉第三方库的 INFO 刷屏（figures 每次导出都会让 fontTools 打几十行
        # 字形表，能把管线自己的阶段进展完全淹掉）。
        silence_noisy_loggers()
        _configured_signature = signature


# --------------------------------------------------------------------------- #
# EventLogger
# --------------------------------------------------------------------------- #


class EventLogger:
    """追加写 ``run_dir/events.jsonl``，每行一个 JSON 对象。线程安全。

    写入失败（磁盘满、权限等）只 warning，绝不向上抛。
    """

    FILENAME = "events.jsonl"

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / self.FILENAME
        self._lock = threading.Lock()
        self._logger = logging.getLogger("autoresearch.events")
        self._seq = 0

    # -- 写 -------------------------------------------------------------- #

    def log(self, event: str, **fields: Any) -> None:
        """追加一行：``{"ts": ..., "event": ..., ...fields}``。"""
        record: dict[str, Any] = {
            "ts": utc_now_iso(),
            "event": str(event),
        }
        for key, value in fields.items():
            record[key] = _jsonable(value)

        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
                self._seq += 1
            except Exception as exc:  # 绝不 raise
                try:
                    self._logger.warning("EventLogger 写入失败 (%s): %s", self.path, exc)
                except Exception:
                    pass

    # -- 读 -------------------------------------------------------------- #

    def tail(self, n: int = 20) -> list[dict]:
        """返回最后 n 条事件（按文件顺序）。"""
        n = max(0, int(n))
        if n == 0:
            return []
        try:
            if not self.path.is_file():
                return []
            with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except Exception as exc:
            self._logger.warning("EventLogger 读取失败 (%s): %s", self.path, exc)
            return []

        out: list[dict] = []
        for line in lines[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out

    def all(self) -> list[dict]:
        """读取全部事件（调试用）。"""
        return self.tail(10**9)

    def count(self) -> int:
        return len(self.all())

    def __repr__(self) -> str:  # pragma: no cover
        return f"EventLogger(path={str(self.path)!r})"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    return str(value)


def event_logger(run_dir: Path) -> EventLogger:
    """工厂函数（CONTRACTS §2）。"""
    return EventLogger(run_dir)


# --------------------------------------------------------------------------- #
# ProgressReporter
# --------------------------------------------------------------------------- #

_STATUS_SYMBOL = {
    "pending": ".",
    "running": ">",
    "done": "OK",
    "ok": "OK",
    "failed": "!!",
    "error": "!!",
    "skipped": "--",
    "skip": "--",
    "warn": "??",
    "warning": "??",
    "info": "  ",
}


class ProgressReporter:
    """CLI 友好进度打印。

    纯 ASCII / box-drawing，无 rich 依赖；``sys.stdout.isatty()`` 为 False
    时绝不输出 ANSI 转义序列（实际上任何情况下都不输出 ANSI）。
    """

    WIDTH_NAME = 20
    WIDTH_STATUS = 6

    def __init__(self, stream: Any = None, enabled: bool = True) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.enabled = bool(enabled)
        self._t0 = time.time()

    # -- 内部 ------------------------------------------------------------ #

    def _write(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            self.stream.write(text + "\n")
            flush = getattr(self.stream, "flush", None)
            if callable(flush):
                flush()
        except Exception:
            self.enabled = False

    @staticmethod
    def _fmt_elapsed(seconds: float | None) -> str:
        if seconds is None:
            return "-"
        try:
            seconds = float(seconds)
        except Exception:
            return "-"
        if seconds < 0:
            return "-"
        if seconds < 60:
            return f"{seconds:5.1f}s"
        minutes, secs = divmod(int(round(seconds)), 60)
        if minutes < 60:
            return f"{minutes:02d}:{secs:02d}"
        hours, minutes = divmod(minutes, 60)
        return f"{hours:d}h{minutes:02d}m"

    # -- 公开 API -------------------------------------------------------- #

    @staticmethod
    def _symbol(status: str) -> str:
        """状态 -> 2 字符标记；未知状态打 warning 但不抛。"""
        key = str(status or "").strip().lower()
        symbol = _STATUS_SYMBOL.get(key)
        if symbol is None:
            logging.getLogger("autoresearch.progress").warning(
                "ProgressReporter: 未知状态 %r，使用通用标记", status
            )
            symbol = "  "
        return symbol

    def step(
        self,
        name: str,
        status: str = "info",
        detail: str = "",
        elapsed: float | None = None,
        **kw: Any,
    ) -> None:
        """打印一行对齐的步骤状态。

        兼容: ``step(name, status, detail)`` / ``step(name, status, detail, elapsed=..)``
        / ``step(name, ok=True, detail=...)`` / ``step("...", "ok", "...", 1.23)``。
        """
        if "duration" in kw and elapsed is None:
            elapsed = kw.pop("duration")
        if "elapsed" in kw and elapsed is None:
            elapsed = kw.pop("elapsed")
        if "seconds" in kw and elapsed is None:
            elapsed = kw.pop("seconds")
        if "ok" in kw:
            ok = kw.pop("ok")
            if status in ("", "info", None):
                status = "done" if ok else "failed"
        if "error" in kw and not detail:
            detail = str(kw.pop("error"))
        if kw:
            logging.getLogger("autoresearch.progress").debug(
                "ProgressReporter.step: 忽略未知参数 %s", sorted(kw)
            )

        symbol = self._symbol(status)
        name_col = str(name)[: self.WIDTH_NAME].ljust(self.WIDTH_NAME)
        status_col = str(status)[: self.WIDTH_STATUS].ljust(self.WIDTH_STATUS)
        parts = ["[", symbol.ljust(2), "] ", name_col, " ", status_col]
        if elapsed is not None:
            parts.append(" " + self._fmt_elapsed(elapsed))
        if detail:
            parts.append("  " + str(detail))
        self._write("".join(parts))

    def info(self, detail: str) -> None:
        self._write(f"       {detail}")

    def rule(self, title: str = "") -> None:
        line = "+" + "-" * 68 + "+"
        self._write(line)
        if title:
            inner = str(title)[:64]
            self._write("| " + inner.ljust(66) + " |")
            self._write(line)

    def summary(self, trace: Iterable[Mapping[str, Any]]) -> None:
        """打印 stage/status/elapsed 表格。"""
        rows = list(trace or [])
        self.rule()
        self._write(
            "| "
            + "STAGE".ljust(20)
            + " "
            + "STATUS".ljust(9)
            + " "
            + "ELAPSED".rjust(8)
            + "  DETAIL"
        )
        self._write("+" + "-" * 20 + "+" + "-" * 10 + "+" + "-" * 9 + "+" + "-" * 30 + "+")
        total = 0.0
        n_failed = 0
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            stage = str(row.get("stage", row.get("name", "?")))[:19]
            ok = row.get("ok", None)
            status = row.get("status")
            if status is None:
                status = "done" if ok is True else ("failed" if ok is False else "?")
            status = str(status)[:8]
            if status.lower() in ("failed", "error"):
                n_failed += 1
            elapsed = row.get("elapsed", None)
            try:
                if elapsed is not None:
                    total += float(elapsed)
            except Exception:
                elapsed = None
            detail = str(row.get("detail", "") or "")[:40]
            self._write(
                "| "
                + stage.ljust(20)
                + " "
                + status.ljust(9)
                + " "
                + self._fmt_elapsed(elapsed).rjust(8)
                + "  "
                + detail
            )
        self._write("+" + "-" * 68 + "+")
        self._write(
            f"  stages={len(rows)}  failed={n_failed}  "
            f"total={self._fmt_elapsed(total).strip()}  "
            f"wall={self._fmt_elapsed(time.time() - self._t0).strip()}"
        )
