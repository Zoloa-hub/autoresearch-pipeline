"""实验后端适配器层。

对外只需要两个名字：:class:`BaseExperimentAdapter`（自己写适配器时继承它）与
:func:`resolve_adapter`（把 CLI 传入的适配器描述解析成实例）。:class:`RunSpec`
描述一次运行的全部参数。

最小适配器示例（只实现两个必需方法）::

    from autoresearch.adapters import BaseExperimentAdapter, RunSpec

    class MyAdapter(BaseExperimentAdapter):
        name = "my-adapter"

        def build_command(self, spec: RunSpec):
            return ["python", "train.py", "--seed", str(spec.seed),
                    "--out", spec.out_dir]

        def parse_results(self, out_dir):
            return {"accuracy": [0.91]}      # 或读 out_dir 里的文件

然后 ``--experiment-adapter my_adapter.py``，或把它装成包并用 entry point 注册。
"""

from __future__ import annotations

from .base import (
    ENTRY_POINT_GROUP,
    LOWER_IS_BETTER_TOKENS,
    AdapterError,
    BaseExperimentAdapter,
    MetricSeries,
    RunSpec,
    builtin_adapters,
    coerce_metric_series,
    higher_is_better,
    read_standard_metrics,
    resolve_adapter,
)
from .lorenz_governance import LorenzGovernanceAdapter
from .script_wrapper import ScriptWrapperAdapter
from .synthetic_toy import SyntheticToyAdapter

__all__ = [
    "ENTRY_POINT_GROUP",
    "LOWER_IS_BETTER_TOKENS",
    "AdapterError",
    "BaseExperimentAdapter",
    "MetricSeries",
    "RunSpec",
    "LorenzGovernanceAdapter",
    "ScriptWrapperAdapter",
    "SyntheticToyAdapter",
    "builtin_adapters",
    "coerce_metric_series",
    "higher_is_better",
    "read_standard_metrics",
    "resolve_adapter",
]
