"""收尾多参数扫描：修 helper、给 runner 加带验证的参数覆盖、跑真实 E2E。

## 覆盖机制（实测得出，不是猜的）

用户模块把 ``OSC_F/OSC_E0/OSC_G`` 绑成 ``eps_lorentz`` 的**默认参数**，
定义时就固定了；``setattr(mod, 'OSC_G', ...)`` **无效**（已实测确认）。
``N_VIS`` 在 ``nk_table`` 函数体内读取，``setattr`` 有效。

所以 runner 必须支持两种 target，并且**用 sig.bind_partial 覆盖**——
这样即使调用方按位置传了该参数，覆盖依然生效（否则又是一个静默失效点）。

## 生效验证（关键）

覆盖失败是**静默的**：所有格点得到同一组数值 → 效应全为 0 →
结论变成「这些参数都不重要」。所以 runner 在应用覆盖后会重算一次，
若结果与未覆盖时**完全相同**，就报错退出（``OVERRIDE_NO_EFFECT``），
而不是交出一份看起来正常的平坦曲线。
"""

from __future__ import annotations

import pathlib

MO = pathlib.Path(__file__).resolve().parents[1] / "autoresearch" / "adapters" / "materials_optics.py"
m = MO.read_text(encoding="utf-8")

# ------------------------------------------------ 1) 追加 _parse_sweep_spec --
if "def _parse_sweep_spec(" not in m:
    m = m.rstrip("\n") + '''


def _parse_sweep_spec(raw: str) -> tuple[SweepAxis, ...]:
    """解析 ``"osc_g:0.3|0.5|0.7;n_vis:2.1|2.3"`` 形式的扫描规格。

    已知轴名会继承其 ``unit`` 与 ``target``（覆盖机制），未知轴名则按
    ``target=同名属性`` 处理——后者对"参数是函数默认值"的模块会静默失效，
    所以 runner 侧还有一道生效验证兜底。
    """
    axes: list[SweepAxis] = []
    known = {a.name: a for a in DEFAULT_SWEEP_AXES}
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        name, values_raw = chunk.split(":", 1)
        name = name.strip()
        values: list[Any] = []
        for token in values_raw.split("|"):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(
                    float(token) if ("." in token or "e" in token.lower()) else int(token)
                )
            except ValueError:
                values.append(token)
        if len(values) < 2:
            continue
        src = known.get(name)
        axes.append(
            SweepAxis(
                name,
                tuple(values),
                src.unit if src else "",
                target=src.target if src else name,
            )
        )
    return tuple(axes)
'''
    print("  已追加 _parse_sweep_spec")

# ------------------------------------------------ 2) RUNNER_TEMPLATE 支持 --set --
OLD_MAIN_MARK = '''def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="materials optics metric runner")'''
NEW_MAIN_MARK = '''def apply_override(mod, target: str, value) -> str:
    """按 target 把参数写进被驱动模块，返回实际应用方式的说明。

    ``target`` 两种形态：

    * ``"N_VIS"``          —— 普通模块属性，直接 ``setattr``
    * ``"eps_lorentz:g"``  —— 该参数是函数 ``eps_lorentz`` 的**默认参数**。
      这种绑定在函数定义时就固定了，``setattr`` 模块常量**没有效果**
      （在真实模块上实测确认），必须重建函数再替换回模块。

    重建用 ``sig.bind_partial`` 而不是简单的 ``kwargs.setdefault``：
    后者在调用方**按位置**传了该参数时会被忽略，于是覆盖静默失效。
    """
    import functools
    import inspect

    if ":" not in target:
        if not hasattr(mod, target):
            raise SystemExit(f"OVERRIDE_TARGET_MISSING: 模块没有属性 {target!r}")
        setattr(mod, target, value)
        return f"setattr({target})"

    fname, pname = target.split(":", 1)
    fn = getattr(mod, fname, None)
    if fn is None or not callable(fn):
        raise SystemExit(f"OVERRIDE_TARGET_MISSING: 模块没有可调用对象 {fname!r}")
    sig = inspect.signature(fn)
    if pname not in sig.parameters:
        raise SystemExit(f"OVERRIDE_TARGET_MISSING: {fname}() 没有参数 {pname!r}")

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.arguments[pname] = value
        bound.apply_defaults()
        return fn(*bound.args, **bound.kwargs)

    setattr(mod, fname, wrapper)
    return f"rebind({fname}:{pname})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="materials optics metric runner")'''
m = m.replace(OLD_MAIN_MARK, NEW_MAIN_MARK, 1)

# 加 --set 参数
m = m.replace(
    '''    ap.add_argument("--out-dir", required=True)''',
    '''    ap.add_argument("--set", dest="sets", action="append", default=[],
                    metavar="TARGET=VALUE",
                    help="覆盖被驱动模块的参数，如 N_VIS=2.3 或 eps_lorentz:g=0.35")
    ap.add_argument("--out-dir", required=True)''',
    1,
)

# 应用覆盖 + 生效验证
OLD_LOAD = '''    mod = load_module(Path(args.module))

    rows = []'''
NEW_LOAD = '''    mod = load_module(Path(args.module))

    # --- 覆盖参数，并**验证真的生效** -------------------------------- #
    # 覆盖失效是静默的：所有格点算出同一组数值 -> 效应全为 0 ->
    # 结论变成「这些参数都不重要」。这是一份看起来完全正常的错误结论，
    # 所以这里主动检测并报错，而不是交出去。
    baseline_probe = None
    applied: list[str] = []
    if args.sets:
        # 覆盖前的参考值（用同一套判据，避免模块自带随机性造成假阳性）
        base_mod_probe = compute_metrics(mod, 0.8, args.jitter, args.seed * 1000)
        for item in args.sets:
            if "=" not in item:
                raise SystemExit(f"OVERRIDE_BAD_SYNTAX: 需要 TARGET=VALUE，得到 {item!r}")
            target, raw_value = item.split("=", 1)
            try:
                value: object = float(raw_value)
            except ValueError:
                value = raw_value
            how = apply_override(mod, target.strip(), value)
            applied.append(f"{target.strip()}={raw_value}({how})")
        after = compute_metrics(mod, 0.8, args.jitter, args.seed * 1000)
        same = all(
            abs(after[k] - base_mod_probe[k]) <= 1e-12 * max(1.0, abs(base_mod_probe[k]))
            for k in base_mod_probe
        )
        if same:
            sys.stderr.write(
                "OVERRIDE_NO_EFFECT: 参数覆盖没有改变任何指标 —— 说明覆盖方式与被驱动"
                "模块的绑定方式不匹配。\\n"
                f"  已尝试: {applied}\\n"
                "  常见原因: 该参数是函数默认值（定义时已绑定），必须写成 "
                "TARGET='函数名:参数名'。\\n"
                "  继续跑下去会得到一条平坦的效应曲线，并被误读成"
                "「这些参数都不重要」，因此这里直接失败。\\n"
            )
            return 3
        baseline_probe = base_mod_probe
        print(f"overrides applied: {'; '.join(applied)}", flush=True)

    rows = []'''
m = m.replace(OLD_LOAD, NEW_LOAD, 1)

MO.write_text(m, encoding="utf-8")

import py_compile  # noqa: E402

py_compile.compile(str(MO), doraise=True)
print("  runner 已支持 --set 与生效验证；编译通过")
