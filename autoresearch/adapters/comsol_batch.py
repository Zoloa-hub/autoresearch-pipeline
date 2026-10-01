r"""COMSOL 批处理后端适配器 —— 材料/多物理场仿真的接入点。

## 这个适配器解决什么

COMSOL 的批处理（``comsol batch``）把结果写在模型文件里，只把**运行日志**打到
stdout/console。而那份日志有两个现实问题，实测确认：

1. **它是 UTF-16LE + BOM。** 用 ``read_text(encoding="utf-8")`` 读会得到乱码，
   而大多数工具默认就是 UTF-8。
2. **其中的中文错误信息已经被销毁。** 实测 21 个真实日志：
   非 ASCII 码点里 **168 个是 U+FFFD（REPLACEMENT CHARACTER）**——
   这意味着信息在 COMSOL **写出时**就丢了，**转码救不回来**。
   （我最初以为"是 GBK 编码、转码即可读"，实测证明那是错的：
   文件是 UTF-16，而且内容已是不可逆的替换字符。）

对第 2 条，本适配器的态度是**如实报告损坏程度**（``log_corruption_ratio``
这个指标），而不是把乱码当正常文本传下去、也不是假装能修好。
真正的解法是让 COMSOL 输出英文（``-locale en`` 或设置 locale），
适配器在 ``quality_note()`` 与报错里都会提示这一点。

## 两种工作模式

* **run**：调 ``comsol batch`` 跑模型（需要装 COMSOL）。命令与输出解析都已实现，
  但**在本机无法验证**——COMSOL 是商业软件，开发机与 CI 都没有。
* **ingest**：只分析**已有的** console 日志（不需要 COMSOL）。
  这条路径可测，而且实用：研究者手上常年积累着大量 batch 日志。

``validate_environment()`` 会如实说明当前哪种模式可用——
把"能力缺口"与"代码坏了"区分开，这是本项目的一贯做法。

## 指标（都能从日志里如实提取，不推测）

| 指标 | 方向 | 含义 |
|---|---|---|
| ``failed_steps`` | 越小越好 | 报 FAILED 的求解步骤数 |
| ``fl_exceptions`` | 越小越好 | FlException 出现次数 |
| ``resolved_features`` | **越大越好** | 成功解析的特征数（探测型脚本的产出） |
| ``log_corruption_ratio`` | 越小越好 | U+FFFD 占非 ASCII 字符的比例——错误文本的可读程度 |

最后一个是本适配器特有的：它把"日志能不能读"变成一个**可比较的数**，
这样"换个 locale 之后错误信息是否变可读"就有了判据。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from .base import BaseExperimentAdapter, MetricSeries, RunSpec, coerce_metric_series
except ImportError:  # pragma: no cover - 支持用 .py 路径独立加载
    from autoresearch.adapters.base import (  # type: ignore[no-redef]
        BaseExperimentAdapter,
        MetricSeries,
        RunSpec,
        coerce_metric_series,
    )

#: COMSOL console 日志的实测编码（21/21 个真实文件都是这个）
LOG_ENCODINGS: tuple[str, ...] = ("utf-16", "utf-16-le", "utf-8-sig", "gbk", "utf-8")

#: 不可恢复的替换字符
REPLACEMENT = "\ufffd"

#: 指标方向：全部由本适配器声明，不依赖 ML 中心的通用词表
DIRECTIONS: dict[str, bool] = {
    "failed_steps": False,          # 失败步骤 -> 越少越好
    "fl_exceptions": False,         # 异常 -> 越少越好
    "resolved_features": True,      # 解析成功的特征 -> 越多越好
    "log_corruption_ratio": False,  # 日志损坏比例 -> 越小越好
}

_STEP_FAILED = re.compile(r"^\s*([\w.]+)\s*:\s*FAILED\s*->")
_FEATURE_OK = re.compile(r"^\s*OK\s+(\S+)\s*->\s*type=")
_FEATURE_BAD = re.compile(r"^\s*FAIL\s+(\S+)")


def read_comsol_log(path: Path) -> tuple[str, str]:
    """读 COMSOL console 日志，返回 ``(文本, 实际编码)``。

    按 BOM 与候选编码依次试探。**不**用 ``errors="replace"`` 掩盖问题——
    调用方需要知道内容是否已被破坏（见 :func:`corruption_ratio`）。
    """
    raw = path.read_bytes()
    for enc in LOG_ENCODINGS:
        try:
            return raw.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", "replace"), "utf-8/replace"


def corruption_ratio(text: str) -> float:
    """非 ASCII 字符中 U+FFFD 的占比 —— "错误文本还能读吗"的量化。

    为什么值得单独做一个指标：COMSOL 的中文错误在写出时就已成 U+FFFD，
    **转码救不回来**。把损坏程度变成数字，才能判断"换 locale 之后是否变好"，
    也才能让读者一眼看出"这段日志不可信"而不是把乱码当原文引用。
    """
    non_ascii = [ch for ch in text if ord(ch) > 127]
    if not non_ascii:
        return 0.0
    bad = sum(1 for ch in non_ascii if ch == REPLACEMENT)
    return bad / len(non_ascii)


class ComsolBatchAdapter(BaseExperimentAdapter):
    """驱动 COMSOL 批处理，或分析已有的 batch console 日志。"""

    name = "comsol-batch"
    description = (
        "COMSOL 多物理场批处理：run 模式调 comsol batch 跑模型；"
        "ingest 模式分析已有 console 日志（含 UTF-16 解码与损坏度量化）。"
    )
    #: run 模式的命令行由适配器构造；ingest 模式不生成代码
    owns_code = True
    entrypoint = ""
    default_timeout = 3600
    max_variants = 2

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        #: run 模式：模型文件（.mph）
        self.model_file = str(self.params.get("model") or "")
        #: run 模式：要运行的 study 标签（如 std1）
        self.study = str(self.params.get("study") or "")
        #: ingest 模式：已有 console 日志所在目录
        self.log_dir = str(self.params.get("log_dir") or "")
        #: run 模式的 comsol 可执行文件名
        self.comsol = str(self.params.get("comsol") or "comsol")
        #: 强制英文 locale（推荐：避免中文错误被写成 U+FFFD）
        self.english_locale = bool(self.params.get("english_locale", True))

    # -- 环境 ----------------------------------------------------------- #
    def _comsol_available(self) -> tuple[bool, str]:
        import shutil as _shutil

        found = _shutil.which(self.comsol)
        if found:
            return True, found
        return False, f"`{self.comsol}` 不在 PATH 上"

    def validate_environment(self) -> tuple[bool, str]:
        """如实说明哪种模式可用 —— 而不是笼统地"失败"。"""
        has_logs = bool(self.log_dir) and Path(self.log_dir).is_dir()
        avail, why = self._comsol_available()
        if has_logs:
            n = len(list(Path(self.log_dir).glob("*.txt"))) + len(
                list(Path(self.log_dir).glob("*.log"))
            )
            return True, (
                f"ingest 模式可用：{self.log_dir} 下有 {n} 个日志"
                + ("；comsol 也可用" if avail else f"；run 模式不可用（{why}）")
            )
        if not self.model_file:
            return False, (
                "需要 adapter-arg 之一：log_dir=<已有 console 日志目录>（ingest 模式，"
                "不需要 COMSOL），或 model=<.mph 路径> + study=<study 标签>（run 模式）"
            )
        if not Path(self.model_file).is_file():
            return False, f"模型文件不存在: {self.model_file}"
        if not avail:
            return False, (
                f"run 模式需要 COMSOL：{why}。"
                "若只想分析已有日志，改用 adapter-arg log_dir=<目录>（不需要 COMSOL）"
            )
        return True, f"run 模式可用：{why}；模型 {Path(self.model_file).name}"

    # -- 执行 ----------------------------------------------------------- #
    def build_command(self, spec: RunSpec) -> list[str]:
        if not self.model_file:
            raise ValueError(
                "comsol-batch 的 run 模式需要 adapter-arg model=<.mph> 与 study=<标签>"
            )
        out = f"{spec.out_dir}/comsol_console.txt"
        argv = [
            self.comsol,
            "batch",
            "-inputfile",
            self.model_file,
            "-outputfile",
            f"{spec.out_dir}/out.mph",
            "-study",
            self.study or "std1",
            "-batchlog",
            out,
        ]
        if self.english_locale:
            # 关键：避免中文错误被写成不可恢复的 U+FFFD（实测 168/168 全毁）。
            # 让 COMSOL 用英文输出，是唯一能让错误信息可读的办法。
            argv += ["-locale", "en_US"]
        argv.extend(str(a) for a in spec.extra_args)
        return argv

    def code_files(self, workspace: Path) -> list[Path]:
        return []

    # -- 指标 ----------------------------------------------------------- #
    def parse_results(self, out_dir: Path) -> MetricSeries:
        """从 console 日志提取指标。

        ``out_dir`` 可以是 run 模式的输出目录，也可以是 ingest 模式下的日志目录。
        """
        logs = self._collect_logs(out_dir)
        if not logs:
            return {}
        series: dict[str, list[float]] = {}
        for log in logs:
            text, _enc = read_comsol_log(log)
            failed = sum(1 for ln in text.splitlines() if _STEP_FAILED.match(ln))
            exc = text.count("FlException")
            resolved = sum(1 for ln in text.splitlines() if _FEATURE_OK.match(ln))
            corrupt = corruption_ratio(text)
            series.setdefault("failed_steps", []).append(float(failed))
            series.setdefault("fl_exceptions", []).append(float(exc))
            series.setdefault("resolved_features", []).append(float(resolved))
            series.setdefault("log_corruption_ratio", []).append(round(corrupt, 4))
        return coerce_metric_series(series)

    def _collect_logs(self, out_dir: Path) -> list[Path]:
        if out_dir.is_dir():
            found = sorted(
                p for p in out_dir.iterdir()
                if p.is_file() and p.suffix.lower() in (".txt", ".log")
            )
            if found:
                return found
        if self.log_dir and Path(self.log_dir).is_dir():
            base = Path(self.log_dir)
            return sorted(
                p for p in base.rglob("*")
                if p.is_file() and p.suffix.lower() in (".txt", ".log")
            )
        return []

    # -- 领域知识 ------------------------------------------------------- #
    def metric_directions(self) -> dict[str, bool] | None:
        return dict(DIRECTIONS)

    def metric_axis(self) -> str | None:
        # 每个日志是一个独立观测（不同 study / 不同探测），没有物理顺序轴
        return "log"

    def quality_note(self) -> str:
        return (
            "结果来自 COMSOL 批处理 console 日志。**注意两类已知限制**："
            "（1）COMSOL 的中文错误信息在写出时就已成 U+FFFD，**转码无法恢复**——"
            "`log_corruption_ratio` 如实给出损坏比例，该值高时不要引用日志文本；"
            "建议加 `-locale en_US`（适配器默认已加）让错误变可读。"
            "（2）本适配器的 run 模式（真的调 `comsol batch`）**从未被执行过**——"
            "COMSOL 是商业软件，开发机与 CI 都没有。已实测的是 ingest 模式"
            "（解析真实日志）与命令构造。"
        )

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        avail, why = self._comsol_available()
        info.update(
            {
                "mode": "ingest" if self.log_dir else "run",
                "comsol_available": avail,
                "comsol_detail": why,
                "model": self.model_file or None,
                "study": self.study or None,
                "log_dir": self.log_dir or None,
                "english_locale": self.english_locale,
                "metric_directions": dict(DIRECTIONS),
                "metric_axis": self.metric_axis(),
                "log_encodings": list(LOG_ENCODINGS),
            }
        )
        return info


#: 支持用 `--experiment-adapter path/to/comsol_batch.py` 直接加载
ADAPTER = ComsolBatchAdapter
