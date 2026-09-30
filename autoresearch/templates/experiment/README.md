# `templates/experiment/` — 参考实验实现（手动运行 / 测试夹具）

**这个目录不参与管线执行。** 它是给人看、给人手动跑的参考实现，也是测试夹具。

管线的**实时实验模板**在 `autoresearch/adapters/synthetic_toy.py` 的
`TRAIN_TEMPLATE`，由 `SyntheticToyAdapter.seed_code()` 返回、s4 负责落盘。
模板之所以归适配器所有：它必须与自己的 `build_command()` / `parse_results()`
配套，否则命令行参数与指标路径会对不上——这正是「模板放在阶段里」的旧设计
留下的问题。放在本目录的那份只服务两件事：

1. 让读者不跑管线也能看到「一个符合契约的实验脚本长什么样」；
2. 作为 `tests/test_prompts.py` / `tests/test_adapters.py` 的夹具，
   验证「确定性种子 + 指标文件契约」这些不变量。

## 内容

| 文件 | 作用 |
|---|---|
| `train.py` | 单文件、零第三方依赖（numpy 可选）、CPU 秒级、确定性种子的分类实验 |
| `run_baseline.py` / `run_method.py` | 手动跑两个变体的薄包装 |

## 手动运行

```powershell
python autoresearch/templates/experiment/train.py `
    --epochs 3 --seed 0 --variant baseline `
    --out-dir autoresearch/templates/experiment/runs/baseline
```

## 输出契约

`--out-dir` 下会写出：

- `metrics.csv`，表头固定 `epoch,loss,accuracy,f1,val_loss,val_accuracy`
- `metrics.jsonl`，每行一个 epoch 的 JSON 对象
- `run_config.json`，本次运行的参数

`autoresearch/tools/metrics.py` 与 `adapters/base.py::read_standard_metrics()`
都认这套约定。指标必须是**逐点序列**而不是最终标量——约八成的图、收敛判断
与逐 epoch 统计都依赖它。

## 想在管线里跑这个脚本

用脚本适配器指过来即可。它会跳过 LLM 代码生成（`owns_code = True`），
但**调试闭环与保真度检查仍然生效**：

```powershell
python -m autoresearch.cli run --direction "..." `
    --experiment-adapter script-wrapper `
    --adapter-arg script=autoresearch/templates/experiment/train.py `
    --adapter-arg epochs=3
```

## 相关文档

- 适配器协议与自定义适配器：[`docs/adapters.md`](../../../docs/adapters.md)
- 内置合成适配器：`autoresearch/adapters/synthetic_toy.py`
