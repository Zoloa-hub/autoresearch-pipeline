r"""修 s7 的修复闭环判据：按**抽出的错误**触发，而不是只看 ok。

## 问题

    if result is not None and result.ok:
        break

`compile()` 的 `ok` 含义是「**产出了 PDF**」，不是「没有 LaTeX 错误」。
本次实测确认：注入一个未闭合的数学模式（`$x = 1 上更优`）后，
tectonic 的退出码仍是 0、PDF 仍被写出，同时抽出了
`! Missing $ inserted.` —— 即 `ok=True` 且 `errors` 非空。

于是 s7 会**立刻 break**，那个错误既不会被修复，也不会出现在任何警告里。
**一篇带 LaTeX 错误的论文被静默放行**，而这是整条管线的最终产物。

## 修法

1. 只有在「产出 PDF **且** 没有抽出的错误」时才 break。
2. 有错误时照常走修复流程（原本那段代码是够用的，只是够不到）。
3. 循环结束若仍有错误残留，**如实记进 warnings** —— 降级交付而不是假装成功，
   也不要因为"毕竟有 PDF"就把它咽下去。
"""

from __future__ import annotations

import pathlib

p = pathlib.Path(__file__).resolve().parents[1] / "autoresearch" / "stages" / "s7_compile.py"
t = p.read_text(encoding="utf-8")
applied = 0

OLD = '''            if result is not None and result.ok:
                break

            errors = list(getattr(result, "errors", []) or []) if result is not None else ["compiler raised"]'''
NEW = '''            errors = list(getattr(result, "errors", []) or []) if result is not None else ["compiler raised"]

            # 判据是「**产出 PDF 且没有抽出的错误**」，而不是只看 ok。
            #
            # `ok` 的含义是"产出了 PDF"：实测过，注入一个未闭合的数学模式后，
            # tectonic 退出码仍为 0、PDF 仍被写出，同时抽出了
            # `! Missing $ inserted.`。若只按 ok 判断，这里会立刻 break，
            # 那个错误既不会被修，也不会进任何警告——
            # **一篇带 LaTeX 错误的论文被静默放行**，而它是整条管线的最终产物。
            if result is not None and result.ok and not errors:
                break'''
if OLD in t:
    t = t.replace(OLD, NEW, 1)
    applied += 1
    print("  s7: 判据已改为 ok 且无错误")
else:
    print("  MISS: s7 的 break 段")

# 循环结束后如实上报残留错误
OLD_TAIL = '''        pdf: Path | None = getattr(result, "pdf", None) if result is not None else None
        ok = bool(result is not None and result.ok and pdf and Path(pdf).exists())'''
NEW_TAIL = '''        pdf: Path | None = getattr(result, "pdf", None) if result is not None else None
        ok = bool(result is not None and result.ok and pdf and Path(pdf).exists())

        # 有 PDF 但仍有 LaTeX 错误时要**说出来**：降级交付，而不是假装干净。
        # 这类错误会让正文出现乱码或缺失内容，而 PDF 本身完全正常打开。
        residual = list(getattr(result, "errors", []) or []) if result is not None else []
        if ok and residual:
            warnings.append(
                f"编译产出了 PDF，但仍有 {len(residual)} 条 LaTeX 错误未被修复"
                f"（正文可能有渲染问题）：{residual[:3]}"
            )'''
if OLD_TAIL in t:
    t = t.replace(OLD_TAIL, NEW_TAIL, 1)
    applied += 1
    print("  s7: 已加残留错误上报")
else:
    print("  MISS: s7 的收尾段")

p.write_text(t, encoding="utf-8")

import py_compile  # noqa: E402

py_compile.compile(str(p), doraise=True)
print(f"  已改 {applied} 处；编译通过")
