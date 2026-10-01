"""用正确编码解析 COMSOL console 日志，并统计行类型。

纠正一个我先前说错的事实：这些日志不是 GBK，而是 **UTF-16LE + BOM**（FF FE）。
之前的乱码是因为按 UTF-8 解码 UTF-16 文件。
"""

from __future__ import annotations

import collections
import pathlib
import re

D = pathlib.Path(r"D:\user\Documents\deepseekv4flash harness\comsol_la2ti2o7")


def decode_log(path: pathlib.Path) -> tuple[str, str]:
    """按 BOM 与试探解码，返回 (文本, 使用的编码)。"""
    raw = path.read_bytes()
    for enc in ("utf-16", "utf-16-le", "utf-8-sig", "gbk", "utf-8"):
        try:
            return raw.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", "replace"), "utf-8/replace"


def main() -> int:
    files = sorted(D.glob("*.console.txt"))
    print(f"共 {len(files)} 个 console 日志\n")

    encs: collections.Counter[str] = collections.Counter()
    kinds: collections.Counter[str] = collections.Counter()
    sample_messages: list[str] = []

    for f in files:
        text, enc = decode_log(f)
        encs[enc] += 1
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if ": FAILED ->" in s:
                kinds["步骤 FAILED"] += 1
            elif s.startswith("set ") and "->" in s:
                kinds["属性 set"] += 1
            elif s.startswith("OK ") and "-> type=" in s:
                kinds["特征解析 OK"] += 1
            elif s.startswith("FAIL ") and ("未知特征" in s or "Unknown" in s):
                kinds["未知特征 ID"] += 1
            elif "FlException" in s:
                kinds["FlException"] += 1
            elif "std.run" in s:
                kinds["std.run 调用"] += 1
            elif "Messages:" in s:
                kinds["Messages 段"] += 1
            if ("未知特征" in s or "未知" in s) and len(sample_messages) < 5:
                sample_messages.append(f"[{f.name}] {s[:110]}")

    print("编码分布:")
    for e, n in encs.most_common():
        print(f"  {e:<14} {n}")
    print("\n行类型统计:")
    for k, v in kinds.most_common():
        print(f"  {k:<16} {v}")
    print("\n中文报错样例（正确解码后）:")
    for m in sample_messages:
        print(f"  {m}")

    # 一个完整日志的开头，看清结构
    print("\n--- eps.console.txt 的正确解码内容（前 18 行）---")
    text, enc = decode_log(D / "eps.console.txt")
    print(f"   (编码: {enc})")
    for line in text.splitlines()[:18]:
        print(f"   {line[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
