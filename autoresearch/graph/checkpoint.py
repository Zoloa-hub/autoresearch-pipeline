"""断点续跑：把状态、事件与产物按阶段快照到磁盘。

设计取舍
--------
* **快照次数优先于体积**：成本主要是 LLM 调用，几百 KB 的 JSON 可以忽略。
* **原子写**：``.tmp`` + ``os.replace``，断电/中断不会留下半截 JSON。
* **滚动保留**：只留最近 ``keep`` 份 ``state_<n>.json``，另有一份恒为最新的
  ``state.json``，供 ``cli resume`` 直接读取。
* **只前滚不回滚**：一个阶段若已 ``done``，恢复时不会因为旧快照而回到 ``pending``；
  合并规则是「状态里已有的 done/skipped 优先，其余字段取快照」。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .state import (
    STATUS_DONE,
    STATUS_SKIPPED,
    TERMINAL_STATUSES,
    first_incomplete_stage,
    to_jsonable,
)

STATE_FILENAME = "state.json"
SNAPSHOT_DIRNAME = "checkpoints"
INDEX_FILENAME = "index.json"


@dataclass
class CheckpointMeta:
    run_id: str
    step: int
    stage: str
    saved_at: float
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step": self.step,
            "stage": self.stage,
            "saved_at": self.saved_at,
            "path": self.path,
            "saved_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.saved_at)),
        }


class Checkpointer:
    """管理 ``<run_dir>/state.json`` 与 ``<run_dir>/checkpoints/``。"""

    def __init__(
        self,
        run_dir: Path,
        run_id: str = "",
        keep: int = 20,
        event_logger: Any = None,
        enabled: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id or self.run_dir.name
        self.keep = max(1, int(keep))
        self.events = event_logger
        self.enabled = enabled
        self.dir = self.run_dir / SNAPSHOT_DIRNAME
        self.step = 0
        self._last_stage = ""

    # ------------------------------------------------------------------ #
    # 基础
    # ------------------------------------------------------------------ #
    def _log(self, event: str, **fields: Any) -> None:
        fn = getattr(self.events, "log", None)
        if callable(fn):
            try:
                fn(event, **fields)
            except Exception:  # pragma: no cover - 日志失败绝不影响主流程
                pass

    def _atomic_write(self, path: Path, text: str) -> bool:
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            return True
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    # ------------------------------------------------------------------ #
    # 保存 / 读取
    # ------------------------------------------------------------------ #
    def save(self, state: dict[str, Any], stage: str = "") -> Path | None:
        """写一份最新状态 + 一份带序号的快照。返回快照路径（失败为 None）。"""
        if not self.enabled:
            return None
        self.step += 1
        self._last_stage = stage or state.get("current_stage", "") or self._last_stage
        state = dict(state)
        state["checkpoint_step"] = self.step
        state["checkpoint_stage"] = self._last_stage
        payload = json.dumps(to_jsonable(state), ensure_ascii=False, indent=1)

        latest = self.run_dir / STATE_FILENAME
        ok = self._atomic_write(latest, payload)

        snap = self.dir / f"state_{self.step:05d}.json"
        ok_snap = self._atomic_write(snap, payload)

        self._prune()
        self._write_index(snap if ok_snap else latest)

        self._log(
            "checkpoint_save",
            step=self.step,
            stage=self._last_stage,
            run_id=self.run_id,
            bytes=len(payload.encode("utf-8")),
            ok=bool(ok or ok_snap),
            path=(snap if ok_snap else latest).as_posix(),
        )
        return snap if ok_snap else (latest if ok else None)

    def _prune(self) -> None:
        try:
            snaps = sorted(self.dir.glob("state_*.json"))
        except OSError:
            return
        excess = len(snaps) - self.keep
        for old in snaps[: max(0, excess)]:
            try:
                old.unlink()
            except OSError:
                pass

    def _write_index(self, path: Path) -> None:
        meta = CheckpointMeta(
            run_id=self.run_id,
            step=self.step,
            stage=self._last_stage,
            saved_at=time.time(),
            path=path.as_posix(),
        )
        self._atomic_write(
            self.dir / INDEX_FILENAME,
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=1),
        )

    def latest_path(self) -> Path | None:
        latest = self.run_dir / STATE_FILENAME
        if latest.exists():
            return latest
        try:
            snaps = sorted(self.dir.glob("state_*.json"))
        except OSError:
            return None
        return snaps[-1] if snaps else None

    def list_snapshots(self) -> list[CheckpointMeta]:
        out: list[CheckpointMeta] = []
        try:
            candidates = sorted(self.dir.glob("state_*.json"))
        except OSError:
            return out
        for p in candidates:
            step = 0
            try:
                step = int(p.stem.split("_")[-1])
            except (ValueError, IndexError):
                pass
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = 0.0
            out.append(CheckpointMeta(self.run_id, step, "", mtime, p.as_posix()))
        return out

    def load(self) -> dict[str, Any] | None:
        path = self.latest_path()
        if path is None:
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------- #
# 恢复合并
# --------------------------------------------------------------------------- #


def merge_resume(
    fresh: dict[str, Any], saved: dict[str, Any], force_restart: bool = False
) -> dict[str, Any]:
    """把快照合并进全新状态。

    规则：

    * ``stage_status`` 中已 **done/skipped** 的阶段在新状态里保持完成态；
      ``failed``/``running``/``pending`` 一律重置为 ``pending``，除非
      ``force_restart``（则全部重置）。
    * 其余键：快照里存在且非空的值覆盖新状态默认值；空值不覆盖。
    * ``trace`` 拼接（保留历史，便于审计），``stage_errors`` 累加但去重。
    """
    merged = dict(fresh)

    saved_status = saved.get("stage_status") or {}
    new_status: dict[str, str] = {}
    for stage, default_status in (fresh.get("stage_status") or {}).items():
        if force_restart:
            new_status[stage] = default_status
            continue
        if saved_status.get(stage) in TERMINAL_STATUSES:
            new_status[stage] = saved_status[stage]
        else:
            new_status[stage] = default_status
    merged["stage_status"] = new_status

    for key, value in saved.items():
        if key in ("stage_status", "trace", "stage_errors", "stage_attempts", "checkpoint_step",
                   "checkpoint_stage"):
            continue
        if key not in merged:
            merged[key] = value
            continue
        if value in (None, "", [], {}, 0, 0.0, False):
            continue
        merged[key] = value

    # 尝试次数：保留历史计数，让重试预算在恢复后继续收窄
    saved_attempts = saved.get("stage_attempts") or {}
    merged["stage_attempts"] = {
        s: int(saved_attempts.get(s, 0)) for s in (fresh.get("stage_status") or {})
    }

    # 错误累加去重
    saved_errors = saved.get("stage_errors") or {}
    errors: dict[str, list[str]] = {}
    for stage, default_list in (fresh.get("stage_errors") or {}).items():
        seen: list[str] = []
        for item in list(default_list) + list(saved_errors.get(stage) or []):
            if item and item not in seen:
                seen.append(item)
        errors[stage] = seen
    merged["stage_errors"] = errors

    merged["trace"] = list(saved.get("trace") or [])

    # 磁盘上的产物比状态里的登记更可信：两边合并，按 path 去重（快照优先）
    saved_arts = [a for a in (saved.get("artifacts") or []) if isinstance(a, dict)]
    fresh_arts = [a for a in (fresh.get("artifacts") or []) if isinstance(a, dict)]
    merged_arts: dict[str, dict] = {}
    for art in fresh_arts + saved_arts:
        path = str(art.get("path") or "")
        if path:
            merged_arts[path] = art
    merged["artifacts"] = list(merged_arts.values())

    return merged


def resume_plan(state: dict[str, Any]) -> dict[str, Any]:
    """给出人类可读的恢复计划。"""
    nxt = first_incomplete_stage(state)
    status = state.get("stage_status", {})
    return {
        "run_id": state.get("run_id", ""),
        "completed": [s for s, st in status.items() if st in TERMINAL_STATUSES],
        "failed": [s for s, st in status.items() if st == "failed"],
        "next_stage": nxt,
        "complete": nxt is None,
        "stage_status": dict(status),
        "artifacts": len(state.get("artifacts") or []),
        "review_score": state.get("review_score", 0.0),
        "final_pdf": state.get("final_pdf", ""),
    }


def load_run_state(run_dir: Path) -> dict[str, Any] | None:
    """从任意运行目录读取最新状态（CLI ``status``/``resume`` 用）。"""
    return Checkpointer(run_dir).load()


def find_run_dir(runs_dir: Path, run_id: str) -> Path | None:
    """支持精确 ID 与唯一前缀匹配。"""
    runs_dir = Path(runs_dir)
    exact = runs_dir / run_id
    if (exact / STATE_FILENAME).exists():
        return exact
    if not run_id:
        return None
    try:
        candidates = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith(run_id)]
    except OSError:
        return None
    with_state = [p for p in candidates if (p / STATE_FILENAME).exists()]
    if len(with_state) == 1:
        return with_state[0]
    if len(with_state) > 1:
        with_state.sort(key=lambda p: p.name)
        return with_state[-1]
    return candidates[0] if len(candidates) == 1 else None


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """列出所有运行及摘要（CLI ``status`` 无参数时用）。"""
    runs_dir = Path(runs_dir)
    out: list[dict[str, Any]] = []
    try:
        dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()], reverse=True)
    except OSError:
        return out
    for d in dirs:
        st = load_run_state(d)
        if st is None:
            continue
        out.append(
            {
                "run_id": st.get("run_id", d.name),
                "dir": d.as_posix(),
                "direction": st.get("direction", ""),
                "completed": len(
                    [s for s, v in (st.get("stage_status") or {}).items()
                     if v in TERMINAL_STATUSES]
                ),
                "next_stage": first_incomplete_stage(st),
                "review_score": st.get("review_score", 0.0),
                "final_pdf": st.get("final_pdf", ""),
            }
        )
    return out
