"""验证 COMSOL 中文报错的真实编码链。

假设：COMSOL 输出 GBK 字节 -> 被某层按 latin-1/cp1252 解码成字符串
      -> 再以 UTF-16 写入日志文件。

验证方法：UTF-16 解码得到中间字符串 -> 按 latin-1 编回字节 -> 按 GBK 解码。
若得到可读中文，假设成立。
"""

from __future__ import annotations

import pathlib

D = pathlib.Path(r"D:\user\Documents\deepseekv4flash harness\comsol_la2ti2o7")


def recover(path: pathlib.Path) -> str:
    """按"UTF-16 -> latin-1 -> GBK"链条还原文本。"""
    mid = path.read_bytes().decode("utf-16")
    try:
        raw = mid.encode("latin-1")
    except UnicodeEncodeError:
        raw = mid.encode("latin-1", "replace")
    for enc in ("gbk", "gb18030", "big5"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def main() -> int:
    f = D / "eps.console.txt"
    print("=" * 72)
    print("原始（UTF-16 解码后，仍是乱码）")
    print("=" * 72)
    mid = f.read_bytes().decode("utf-16")
    for line in mid.splitlines():
        if "FlException" in line or "Messages" in line or "实际" in line:
            print(f"  {line[:110]}")

    print()
    print("=" * 72)
    print("按 UTF-16 -> latin-1 -> GBK 还原后")
    print("=" * 72)
    fixed = recover(f)
    for line in fixed.splitlines():
        s = line.strip()
        if s and ("失败" in s or "未知" in s or "尚未" in s or "错误" in s
                  or "Messages" in s or "FAILED" in s or s.startswith("-")):
            print(f"  {s[:120]}")

    print()
    print("=" * 72)
    print("所有日志里能还原出的中文错误（去重）")
    print("=" * 72)
    seen: set[str] = set()
    for p in sorted(D.glob("*.console.txt")):
        try:
            text = recover(p)
        except Exception:
            continue
        for line in text.splitlines():
            s = line.strip().lstrip("-").strip()
            if not s or not any("\u4e00" <= ch <= "\u9fff" for ch in s):
                continue
            if len(s) < 6 or len(s) > 160:
                continue
            if s in seen:
                continue
            seen.add(s)
            if len(seen) <= 18:
                print(f"  {s}")
    print(f"\n  共还原出 {len(seen)} 条不同的中文消息")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
