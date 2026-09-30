"""离线启发式 MockBackend（CONTRACTS §3 的 `mock` provider）。

设计目标：`--llm-provider mock --offline` 能把整条管线跑通并产出**有用**的
结构化输出，而不是一句常量字符串。

- **完全确定性**：所有"随机"选择都由 ``hashlib.sha256(prompt)`` 播种，
  同一条 prompt 永远得到同一个答案，不使用全局 ``random``。
- **意图路由**：``INTENT_RULES`` 是有序列表，先匹配到的意图优先，
  按关键词数量打分避免长规则被短规则遮蔽。
- 返回**裸 JSON**（不带 ``` 围栏、不带前后解释），客户端的提取器两者都吃。

产出形状（JSON 化后进图状态）：

* ``query``       -> ``{"queries": [6 条检索式], "rationale"}``
* ``literature``  -> ``{"summary", "gaps", "themes", "method_landscape"}``
* ``novelty``     -> ``{"verdict", "score", "rationale", "closest",
                        "overlap_risks", "differentiators"}``
* ``idea``        -> ``{"ideas": [Idea 形状（见 CONTRACTS §9）]}``
* ``plan``        -> ``{"objective", "core_claim", "milestones", "dataset",
                        "baseline", "metrics", "ablation_matrix", ...}``
* ``debug``       -> ``{"diagnosis", "root_cause", "files", "confidence",
                        "validity_note"}``
* ``code``        -> ``{"files": [{"path": "train.py", "content": <可运行脚本>}],
                        "entrypoint", "run_command", "notes"}``
* ``writing``     -> ``{"latex", "section", "citations_used", "claims",
                        "citations_needed", "word_count"}``
* ``review``      -> ``{"score", "verdict", "strengths", "weaknesses",
                        "per_criterion", "min_fixes", ...}``
* ``analysis``    -> ``{"claim_evidence", "findings", "limitations", ...}``
* ``fallback``    -> ``{"text", "ok"}``

字段是一次性给全的（而非最小集）：真实模板会原样消费这些键，
少一个键就会让对应阶段静默退化成占位内容——那种失败看起来像"模型没干活"，
很难归因到 mock 身上。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .client import LLMResponse, Usage

__all__ = [
    "MockBackend",
    "INTENT_RULES",
    "INTENT_NAMES",
    "intent_of",
    "TRAIN_PY_TEMPLATE",
]


# --------------------------------------------------------------------------- #
# 确定性伪随机
# --------------------------------------------------------------------------- #


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _isalt(text: str, salt: str) -> int:
    return int(_digest(f"{salt}|{text}")[:16], 16)


def _pick(prompt: str, options: list[Any], salt: str = "") -> Any:
    """确定性从 options 里挑一个。"""
    if not options:
        return None
    return options[_isalt(prompt, salt) % len(options)]


def _decide(prompt: str, salt: str, mod: int = 100) -> int:
    return _isalt(prompt, salt) % mod


# --------------------------------------------------------------------------- #
# 意图规则（有序；越具体越靠前）
# --------------------------------------------------------------------------- #

#: 强信号：只匹配 prompt 开头的标题区（前 300 字符 / 首个标题行）。
#: 标题里的意图词才是真正决定任务的那个。
INTENT_RULES_HEADING: list[tuple[str, list[str]]] = [
    (
        "debug",
        [
            r"debug",
            r"报错",
            r"编译修复",
            r"compile (?:repair|fix)",
            r"traceback",
            r"修复",
        ],
    ),
    (
        "review",
        [
            r"review",
            r"评审",
            r"审稿",
            r"revision",
            r"rebuild",
            r"打分",
        ],
    ),
    (
        "novelty",
        [
            r"novelty",
            r"新颖性",
            r"查新",
        ],
    ),
    (
        "writing",
        [
            r"writing",
            r"section",
            r"abstract",
            r"draft",
            r"撰写",
            r"写作",
            r"报告",
            r"report",
        ],
    ),
    (
        "query",
        [
            r"quer(?:y|ies)",
            r"检索",
            r"关键词",
        ],
    ),
    (
        "literature",
        [
            r"literature",
            r"survey",
            r"综述",
            r"文献",
            r"related work",
        ],
    ),
    (
        "plan",
        [
            r"experiment plan",
            r"planning",
            r"experiment design",
            r"milestone",
            r"实验计划",
            r"规划",
        ],
    ),
    (
        "analysis",
        [
            r"analysis",
            r"analyz",
            r"analys",
            r"分析",
            r"results analysis",
        ],
    ),
    (
        "idea",
        [
            r"idea",
            r"ideation",
            r"构思",
            r"头脑风暴",
            r"假设生成",
            r"hypothes",
        ],
    ),
    (
        "code",
        [
            r"code generation",
            r"codegen",
            r"implement",
            r"代码生成",
            r"代码",
            r"脚本",
        ],
    ),
]

#: 弱信号：匹配全文（正文里出现的关键词）。
INTENT_RULES_BODY: list[tuple[str, list[str]]] = [
    (
        "debug",
        [
            r"Traceback",
            r"报错",
            r"错误信息",
            r"调试",
            r"修复",
            r"stderr",
            r"stack ?trace",
            r"SyntaxError",
            r"ModuleNotFoundError",
            r"debug",
            r"fix the (?:bug|error)",
        ],
    ),
    (
        "review",
        [
            r"评审",
            r"审稿",
            r"打分",
            r"审阅意见",
            r"同行评议",
            r"review",
            r"reviewer",
            r"critique",
            r"referee",
        ],
    ),
    (
        "novelty",
        [
            r"新颖性",
            r"查新",
            r"novelty",
            r"prior art",
            r"重复度",
        ],
    ),
    (
        "writing",
        [
            r"撰写",
            r"写作",
            r"起草",
            r"论文段落",
            r"write (?:the |a )?section",
            r"draft",
            r"write section",
            r"related work 段落",
        ],
    ),
    (
        "query",
        [
            r"检索式",
            r"检索词",
            r"关键词",
            r"生成检索",
            r"生成.*检索",
            r"search quer",
            r"search string",
            r"keywords?",
            r"query",
        ],
    ),
    (
        "literature",
        [
            r"综述",
            r"文献",
            r"研究缺口",
            r"related work",
            r"literature",
            r"research gap",
            r"\bgaps?\b",
        ],
    ),
    (
        "plan",
        [
            r"实验计划",
            r"实验方案",
            r"里程碑",
            r"消融矩阵",
            r"experiment plan",
            r"milestone",
            r"ablation matrix",
        ],
    ),
    (
        "analysis",
        [
            r"分析",
            r"分析结果",
            r"结论",
            r"洞察",
            r"\banaly",
            r"insight",
            r"result analysis",
        ],
    ),
    (
        "idea",
        [
            r"构思",
            r"头脑风暴",
            r"研究假设",
            r"创新点",
            r"\bideas?\b",
            r"hypothes[ei]s",
            r"brainstorm",
        ],
    ),
    (
        "code",
        [
            r"代码",
            r"脚本",
            r"实现",
            r"生成代码",
            r"\bcode\b",
            r"scripts?",
            r"\bimplement",
            r"train\.py",
        ],
    ),
]

#: 兼容别名：有序的全量规则表（先标题强信号，再全文弱信号）。
INTENT_RULES: list[tuple[str, list[str]]] = INTENT_RULES_HEADING + INTENT_RULES_BODY

_ALL_RULES = INTENT_RULES_HEADING + INTENT_RULES_BODY

_INTENT_PRIORITY: dict[str, int] = {}
for _i, (_name, _pats) in enumerate(_ALL_RULES):
    _INTENT_PRIORITY.setdefault(_name, _i)

INTENT_NAMES: tuple[str, ...] = tuple(
    dict.fromkeys(name for name, _ in _ALL_RULES)
) + ("fallback",)

_COMPILED_HEADING: list[tuple[str, list[re.Pattern[str]]]] = [
    (name, [re.compile(pat, re.IGNORECASE) for pat in pats])
    for name, pats in INTENT_RULES_HEADING
]
_COMPILED_BODY: list[tuple[str, list[re.Pattern[str]]]] = [
    (name, [re.compile(pat, re.IGNORECASE) for pat in pats])
    for name, pats in INTENT_RULES_BODY
]

#: 标题区长度（字符）
_HEAD_CHARS = 300


def _heading_region(text: str) -> str:
    """取 prompt 开头的标题区（首个非空行上下 300 字符内）。"""
    head = text[: _HEAD_CHARS * 3]
    for line in head.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:_HEAD_CHARS]
    return text[:_HEAD_CHARS]


def score_intents(text: str) -> dict[str, int]:
    """每个意图的得分。标题强信号记 3 分，正文弱信号每处记 1 分。"""
    if not text:
        return {}
    heading = _heading_region(text)
    scores: dict[str, int] = {}
    for name, patterns in _COMPILED_HEADING:
        hits = sum(1 for pat in patterns if pat.search(heading))
        if hits:
            scores[name] = scores.get(name, 0) + 3 * hits
    for name, patterns in _COMPILED_BODY:
        hits = sum(1 for pat in patterns if pat.search(text))
        if hits:
            scores[name] = scores.get(name, 0) + hits
    return scores


def intent_of(prompt: str, text: str | None = None) -> str:
    """返回匹配到的意图名；都不匹配返回 ``"fallback"``。

    打分 = 标题强信号(3) + 正文弱信号(1)；同分时规则顺序靠前的赢。
    """
    if text is None:
        text = prompt or ""
    if not text:
        return "fallback"
    scores = score_intents(text)
    if not scores:
        return "fallback"
    best_name = "fallback"
    best_score = 0
    best_prio = len(_ALL_RULES) + 1
    for name, score in scores.items():
        prio = _INTENT_PRIORITY.get(name, len(_ALL_RULES))
        if score > best_score or (score == best_score and prio < best_prio):
            best_name, best_score, best_prio = name, score, prio
    return best_name


# --------------------------------------------------------------------------- #
# 文本提取工具
# --------------------------------------------------------------------------- #

_DIRECTION_PATTERNS = [
    r"(?:研究)?方向\s*[:：]\s*(.+)",
    r"研究主题\s*[:：]\s*(.+)",
    r"研究问题\s*[:：]\s*(.+)",
    r"(?:research\s+)?direction\s*[:：]\s*(.+)",
    r"research\s+(?:topic|question)\s*[:：]\s*(.+)",
    r"\btopic\s*[:：]\s*(.+)",
    r"课题\s*[:：]\s*(.+)",
]

_NOISE_PREFIX = re.compile(
    r"^\s*(?:#|//|>|<!--|-\s|\*\s|\d+[.)]\s|你是|You are|请|任务|Task|Role|System|系统"
    # 语言/格式指令行：它们出现在提示词里但**不是**研究方向。若不过滤，
    # 回退分支会把「输出语言：简体中文…」当成方向，污染标题与章节正文。
    r"|输出语言|写作语言|语言\s*[:：]|用中文|中文撰写|用英文|英文撰写"
    r"|Output language|Write in (?:Chinese|English)|Respond in)",
    re.IGNORECASE,
)


def extract_direction(prompt: str, max_len: int = 80) -> str:
    """从 prompt 中尽力抽出研究方向的短句。

    退化分支只在 prompt 的**前半部分**里找方向，而不是从头扫到尾。
    原因：目标会议、章节名、字数要求这类「尾部配置」出现在 prompt 后半段，
    若不加限制，``Venue: NeurIPS`` 会被当成研究方向，最终污染论文标题
    （实测出现过标题为 "Controlled Budget Comparison for NeurIPS"）。
    """
    if not prompt:
        return "the target research direction"
    for pat in _DIRECTION_PATTERNS:
        m = re.search(pat, prompt, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            cand = cand.strip("\"'“”‘’` ")
            if cand:
                return cand[:max_len]

    lines = prompt.splitlines()
    head = lines[: max(1, len(lines) // 2)] if len(lines) > 6 else lines

    def _usable(raw: str) -> str | None:
        line = raw.strip()
        if len(line) < 8 or len(line) > 300:
            return None
        if _NOISE_PREFIX.match(line):
            return None
        if line.endswith((":", "：")):
            return None
        return line[:max_len]

    for raw in head:
        hit = _usable(raw)
        if hit:
            return hit
    # 前半部分没有可用行时，才看后半部分（仍然排除噪声行）
    for raw in lines:
        hit = _usable(raw)
        if hit:
            return hit
    return prompt.strip()[:max_len] or "the target research direction"


def _extract_labeled_value(prompt: str, labels: tuple[str, ...]) -> str:
    """从提示词里取出 ``title: ...`` / ``标题：...`` 这类已给定的值。

    这是「复用上游结论」而不是「重新发明」：章节撰写应该围绕已选定的假设标题展开，
    抓到它就比从整段提示词里猜方向可靠得多。
    """
    for label in labels:
        m = re.search(
            rf"^\s*[-*\d.]*\s*{re.escape(label)}\s*[:：]\s*(.+?)\s*$",
            prompt or "",
            re.IGNORECASE | re.MULTILINE,
        )
        if not m:
            continue
        value = m.group(1).strip().strip("\"'“”‘’`* ")
        value = re.sub(r"^[\[\(]+|[\]\)]+$", "", value).strip()
        if 6 <= len(value) <= 200:
            return value
    return ""


def _largest_block(prompt: str) -> str:
    """返回 prompt 里最长的一段（常是被贴进来的论文/代码）。"""
    blocks = re.split(r"\n\s*\n", prompt or "")
    blocks = [b.strip() for b in blocks if b and b.strip()]
    if not blocks:
        return (prompt or "").strip()
    return max(blocks, key=len)


def _extract_round(prompt: str, default: int = 1) -> int:
    """抽出评审轮次 round / 轮次。"""
    for pat in (
        r"\bround\s*[=:#]?\s*(\d+)",
        r"轮次\s*[=:#]?\s*(\d+)",
        r"第\s*(\d+)\s*轮",
        r"第\s*([一二三四五六七八九十]+)\s*轮",
        r"\biteration\s*[=:#]?\s*(\d+)",
    ):
        m = re.search(pat, prompt or "", re.IGNORECASE)
        if m:
            raw = m.group(1)
            if raw.isdigit():
                return max(1, int(raw))
            cn = "零一二三四五六七八九十"
            if all(ch in cn for ch in raw):
                if raw == "十":
                    return 10
                if raw.startswith("十"):
                    return 10 + cn.index(raw[1])
                if len(raw) == 2 and raw[1] == "十":
                    return cn.index(raw[0]) * 10
                return cn.index(raw)
    return default


def _extract_int(prompt: str, keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        m = re.search(rf"{key}\s*[\"'=:：]?\s*(\d+)", prompt or "", re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return default


def _clip(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def _estimate_tokens(text: str) -> int:
    """粗略 token 估计：CJK 约 1 token/字，拉丁约 1 token/4 字符。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, cjk + other // 4)


# --------------------------------------------------------------------------- #
# 生成的可执行实验脚本（stdlib-only，确定性）
# --------------------------------------------------------------------------- #

TRAIN_PY_TEMPLATE = r'''#!/usr/bin/env python
"""Auto-generated deterministic synthetic-classifier training script.

Runs with the Python standard library only.  Writes:
  * metrics.csv   header: epoch,loss,accuracy,f1,val_loss,val_accuracy
  * metrics.jsonl one JSON object per epoch
Prints ``FINAL accuracy=0.9xx`` at the end.
"""

import argparse
import csv
import json
import math
import os
import random
import sys

CSV_HEADER = ["epoch", "loss", "accuracy", "f1", "val_loss", "val_accuracy"]


def make_data(n=900, dim=8, classes=3, seed=0, noise=0.75):
    """Deterministic synthetic multi-class data with a learnable signal."""
    rng = random.Random(seed)
    centers = []
    for c in range(classes):
        center = [0.0] * dim
        for j in range(dim):
            center[j] = rng.uniform(-1.0, 1.0)
        # push class c strongly along its own axis
        center[c % dim] += 2.6
        norm = math.sqrt(sum(v * v for v in center)) or 1.0
        centers.append([v * 2.4 / norm for v in center])
    xs, ys = [], []
    for i in range(n):
        c = i % classes
        x = [centers[c][j] + rng.gauss(0.0, noise) for j in range(dim)]
        xs.append(x)
        ys.append(c)
    return xs, ys, classes, dim


def split(xs, ys, frac=0.25):
    n_val = int(len(xs) * frac)
    return xs[n_val:], ys[n_val:], xs[:n_val], ys[:n_val]


def init_params(classes, dim, rng, scale=0.02):
    W = [[rng.gauss(0.0, scale) for _ in range(dim)] for _ in range(classes)]
    b = [0.0] * classes
    return W, b


def forward(W, b, x, classes):
    logits = [sum(W[c][j] * x[j] for j in range(len(x))) + b[c] for c in range(classes)]
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [v / s for v in exps]


def loss_and_grads(W, b, xs, ys, classes, weight_decay):
    dim = len(xs[0])
    gW = [[0.0] * dim for _ in range(classes)]
    gb = [0.0] * classes
    total = 0.0
    for x, y in zip(xs, ys):
        p = forward(W, b, x, classes)
        total += -math.log(max(p[y], 1e-12))
        for c in range(classes):
            g = p[c] - (1.0 if c == y else 0.0)
            gb[c] += g
            row = gW[c]
            for j in range(dim):
                row[j] += g * x[j]
    n = float(len(xs)) or 1.0
    for c in range(classes):
        gb[c] /= n
        row = gW[c]
        for j in range(dim):
            row[j] /= n
            row[j] += weight_decay * W[c][j]
    return total / n, gW, gb


def evaluate(W, b, xs, ys, classes, weight_decay=0.0):
    if not xs:
        return 0.0, 0.0, 0.0
    total = 0.0
    correct = 0
    tp = [0] * classes
    fp = [0] * classes
    fn = [0] * classes
    for x, y in zip(xs, ys):
        p = forward(W, b, x, classes)
        total += -math.log(max(p[y], 1e-12))
        pred = p.index(max(p))
        if pred == y:
            correct += 1
            tp[y] += 1
        else:
            fp[pred] += 1
            fn[y] += 1
    acc = correct / float(len(xs))
    f1s = []
    for c in range(classes):
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1s.append((2.0 * tp[c] / denom) if denom else 0.0)
    return total / float(len(xs)), acc, sum(f1s) / float(classes)


def train(args):
    xs, ys, classes, dim = make_data(seed=args.seed, noise=args.noise)
    xtr, ytr, xva, yva = split(xs, ys)
    rng = random.Random(args.seed + 991)
    W, b = init_params(classes, dim, rng)

    variant = "baseline" if args.baseline else args.variant
    baseline = (variant == "baseline")
    # 方法变体：动量 + 权重衰减 + 学习率衰减；基线：纯 SGD。
    lr0 = 0.30 if baseline else 0.45
    momentum = 0.0 if baseline else 0.9
    weight_decay = 0.0 if baseline else 2e-4
    batch = 128
    vW = [[0.0] * dim for _ in range(classes)]
    vb = [0.0] * classes
    tr_loss = 0.0
    rows = []
    epochs = args.epochs

    for epoch in range(1, epochs + 1):
        if baseline:
            lr = lr0
        else:
            lr = lr0 * (0.25 ** (epoch / max(1.0, epochs)))
        idx = list(range(len(xtr)))
        rng.shuffle(idx)
        for s in range(0, len(idx), batch):
            chunk = idx[s:s + batch]
            bx = [xtr[i] for i in chunk]
            by = [ytr[i] for i in chunk]
            l, gW, gb = loss_and_grads(W, b, bx, by, classes, weight_decay)
            tr_loss = l
            for c in range(classes):
                for j in range(dim):
                    vW[c][j] = momentum * vW[c][j] - lr * gW[c][j]
                    W[c][j] += vW[c][j]
                vb[c] = momentum * vb[c] - lr * gb[c]
                b[c] += vb[c]
        _, acc, f1 = evaluate(W, b, xtr, ytr, classes)
        vloss, vacc, vf1 = evaluate(W, b, xva, yva, classes)
        rows.append({
            "epoch": epoch,
            "loss": round(tr_loss, 6),
            "accuracy": round(acc, 6),
            "f1": round(f1, 6),
            "val_loss": round(vloss, 6),
            "val_accuracy": round(vacc, 6),
        })
        print("epoch=%d loss=%.4f acc=%.4f f1=%.4f val_loss=%.4f val_acc=%.4f"
              % (epoch, tr_loss, acc, f1, vloss, vacc))
        sys.stdout.flush()

    outdir = os.path.abspath(args.out_dir)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    csv_path = os.path.join(outdir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_HEADER})

    jsonl_path = os.path.join(outdir, "metrics.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(os.path.join(outdir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"variant": variant, "epochs": len(rows), "seed": args.seed,
                   "noise": args.noise, "batch": batch, "lr0": lr0,
                   "momentum": momentum, "weight_decay": weight_decay}, fh, indent=2)

    final = rows[-1] if rows else {"accuracy": 0.0, "val_accuracy": 0.0, "f1": 0.0}
    best = max(rows, key=lambda r: r["val_accuracy"]) if rows else final
    print("FINAL accuracy=%.4f val_accuracy=%.4f f1=%.4f epochs=%d variant=%s"
          % (best["val_accuracy"], best["val_accuracy"], best["f1"], len(rows), variant))
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="deterministic synthetic classifier")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    # 不限定 choices：消融臂的变体名是中性标识（abl-1…），且轴取值的参数个数与名字
    # 随消融矩阵而变，无法静态声明。用 parse_known_args 承接动态参数——
    # 这与适配器自带的模板（adapters/synthetic_toy.TRAIN_TEMPLATE）保持同一契约。
    p.add_argument("--variant", default="method")
    p.add_argument("--baseline", action="store_true",
                   help="alias for --variant baseline")
    p.add_argument("--noise", type=float, default=0.8)
    p.add_argument("--out-dir", "--out", dest="out_dir", default=".")
    return p


def main(argv=None):
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    # 消融轴取值以 --<轴名> <值> 形式传入；未知参数不再让实验整体失败。
    axis = {}
    i = 0
    while i < len(extra):
        token = extra[i]
        if token.startswith("--") and i + 1 < len(extra):
            try:
                axis[token[2:].replace("-", "_")] = float(extra[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            i += 1
    # 把轴取值映射成对训练目标的实际影响（学习率缩放），使消融臂之间确有差异
    if "weight_decay" in axis:
        args.lr = float(getattr(args, "lr", 0.1)) * (1.0 - min(0.9, axis["weight_decay"] * 1000.0))
    if args.baseline:
        args.variant = "baseline"
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
'''


# --------------------------------------------------------------------------- #
# 代码修复 / 识别工具
# --------------------------------------------------------------------------- #


def self_looks_like_python(text: str) -> bool:
    """粗糙判断一段文本是否是 Python 代码。"""
    if not text or len(text) < 20:
        return False
    markers = 0
    for pat in (
        r"^\s*def \w+\(",
        r"^\s*import \w+",
        r"^\s*from \w+ import",
        r"^\s*class \w+",
        r"\bprint\(",
        r"__name__\s*==",
    ):
        if re.search(pat, text, re.MULTILINE):
            markers += 1
    return markers >= 2


def _repair_python(content: str) -> str:
    """最小化修复被贴进来的脚本：补 import os + 输出目录保护，不重写逻辑。"""
    if not content:
        return content
    text = content if content.endswith("\n") else content + "\n"
    has_os = re.search(r"^\s*import os\b", text, re.MULTILINE) is not None
    writes_metrics = re.search(r"""open\(\s*["']metrics\.""", text) is not None
    needs_path_fix = writes_metrics and "makedirs" not in text

    if not needs_path_fix:
        return text

    lines = text.splitlines()
    if not has_os:
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                insert_at = i + 1
        lines.insert(insert_at, "import os  # [mock-fix] 输出目录保护所需")

    lines.insert(0, "# [mock-fix] 确保 metrics 输出写进当前目录（自动修复）")
    for i, line in enumerate(lines):
        if "open(" in line and "metrics." in line:
            indent = line[: len(line) - len(line.lstrip())]
            lines.insert(i, f"{indent}os.makedirs('.', exist_ok=True)")
            break

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# MockBackend
# --------------------------------------------------------------------------- #


class MockBackend:
    """确定性的离线启发式后端。"""

    name = "mock"
    model_name = "mock-heuristic"

    def __init__(self, temperature: float = 0.3, **_: Any) -> None:
        self.temperature = temperature
        self.call_log: list[str] = []
        self.intents: list[str] = []

    # -- 接口 ------------------------------------------------------------ #

    def intent_of(self, prompt: str) -> str:
        """返回该 prompt 命中的意图名（供测试断言路由）。"""
        return intent_of(prompt)

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kw: Any,
    ) -> LLMResponse:
        prompt = prompt or ""
        self.call_log.append(prompt[:120])
        intent = self.intent_of(prompt)
        self.intents.append(intent)

        handler = getattr(self, f"_gen_{intent}", None)
        payload: Any
        if handler is None:
            payload = self._gen_fallback(prompt)
        else:
            try:
                payload = handler(prompt, system)
            except Exception as exc:  # 生成器自身出错也不能炸管线
                payload = {
                    "text": f"[mock] intent={intent} generation failed: {exc}",
                    "ok": False,
                }

        text = json.dumps(payload, ensure_ascii=False, default=str)
        usage = Usage(
            prompt_tokens=_estimate_tokens(prompt) + _estimate_tokens(system or ""),
            completion_tokens=_estimate_tokens(text),
            total_tokens=_estimate_tokens(prompt)
            + _estimate_tokens(system or "")
            + _estimate_tokens(text),
            calls=1,
        )
        return LLMResponse(
            text=text,
            raw={"mock": True, "intent": intent},
            usage=usage,
            model=self.model_name,
            cached=False,
        )

    def describe(self) -> dict:
        return {"name": self.name, "model": self.model_name, "offline": True,
                "calls": len(self.call_log)}

    # -- 各意图生成器 ---------------------------------------------------- #

    def _gen_query(self, prompt: str, system: str | None = None) -> dict:
        """s1_queries：{"queries": [6-8], "rationale": str}"""
        direction = extract_direction(prompt)
        short = _clip(direction, 60)
        year = _pick(prompt, ["2024", "2025"], "query_year")
        raw = [
            f"{short} survey {year}",
            f"{short} benchmark evaluation protocol",
            f"{short} robustness distribution shift",
            f"{short} efficient lightweight method",
            f"{short} ablation reproducibility",
            f"{short} negative results limitations",
        ]
        queries = [re.sub(r"\s+", " ", q).strip() for q in raw]
        rationale = (
            f"覆盖策略：以 {short} 为核心，分别沿「方法同义改写」「评测基准」「鲁棒性与分布偏移」"
            "「效率与轻量化」「消融与复现」「负结果与失败模式」六个互补角度展开。"
            "前三条保证召回（同义与近邻领域），中间两条锁定可比的方法与数据集命名，"
            "最后两条专门用于检索批评性文献，避免只看到正结果。"
            "所有检索式都是 3–9 词的短关键词串，可直接投喂 arXiv / S2 / OpenAlex。"
        )
        return {"queries": queries, "rationale": rationale}

    def _gen_literature(self, prompt: str, system: str | None = None) -> dict:
        """s1_survey：summary / themes[{name,paper_ids,summary}] / gaps[{gap,...}] / method_landscape"""
        direction = extract_direction(prompt)
        short = _clip(direction, 60)
        theme_specs = [
            ("方法族与共同假设", ["p1", "p2"],
             f"{short} 的代表性方法族共享一个隐含假设：任务分布与训练分布同构。"),
            ("评测协议与数据集", ["p3"],
             "不同工作使用不同的划分与早停策略，跨论文数值不可直接比较。"),
            ("效率与鲁棒性权衡", ["p4"],
             "参数高效 / 计算受限的方法常以鲁棒性换取更低的算力开销。"),
            (_pick(prompt, ["自监督预训练", "提示学习", "结构化先验", "多任务迁移"], "lit_theme"),
             ["p5"],
             "该方向把先验从数据搬到结构，代价是引入了新的超参敏感性。"),
        ]
        themes = [
            {"name": name, "paper_ids": ids, "summary": text}
            for name, ids, text in theme_specs
        ]
        gap_specs = [
            (f"现有工作大多在单一数据集上验证 {short}",
             "评测资源成本高，跨域复现实验很少被报告。",
             "在固定预算下做跨域对照，把「跨域掉点」当作一等指标。",
             ["p1", "p3"]),
            ("缺少对计算预算的显式控制，方法间对比不公平",
             "论文同时改变数据、规模与训练配方，增益无法归因。",
             "用等预算网格搜索分离「方法增益」与「调参增益」。",
             ["p2", "p4"]),
            ("负结果与失败案例几乎不被报告",
             "发表偏倚使失败模式不可见，方法边界不可知。",
             "构造子群划分并报告最差组精度与失败预测 AUC。",
             ["p3"]),
        ]
        gaps = [
            {"gap": g, "why_unsolved": why, "opportunity": opp, "supporting_ids": ids}
            for g, why, opp, ids in gap_specs
        ]
        method_landscape = [
            {"approach": "统计 / 手工特征基线", "representative_ids": ["p1"],
             "limitation": "难以覆盖高维交互，天花板低但可解释。"},
            {"approach": "端到端监督学习", "representative_ids": ["p2", "p4"],
             "limitation": "对数据规模与训练配方高度敏感，复现成本高。"},
            {"approach": "结构化先验 / 正则化", "representative_ids": ["p5"],
             "limitation": "先验选择本身成为新的超参，缺少选择准则。"},
        ]
        summary = "\n\n".join(
            [
                f"## 综述脉络：{short}\n\n"
                f"围绕 {short} 的文献可分成三簇：早期以手工特征与统计模型为主，"
                "中期转向端到端学习，近期集中在规模化预训练与参数高效适配。"
                "三簇共享一个隐含假设——任务分布与训练分布同构；这正是后续争议的根源。"
                "方法族的共同假设见 \\cite{p1} 与 \\cite{p2}。",
                "## 方法与评测\n\n"
                "主流方法在标准基准上报告了稳定的增益，但增益的归因并不清晰："
                "多数论文同时改变了数据、模型规模与训练配方，消融只覆盖了其中一部分。"
                "评测层面，不同工作使用不同的划分与早停策略（\\cite{p3}），"
                "导致跨论文数值不可直接比较。效率导向的方法（\\cite{p4}）"
                "通常以鲁棒性换取更低的算力开销，这一权衡很少被显式量化。",
                "## 研究缺口\n\n"
                f"综合来看，{short} 的主要缺口在于**受控比较**与**失败模式的系统刻画**。"
                "一个可复现的最小实验，如果能在固定预算下区分「方法增益」与「调参增益」，"
                "并同时报告最差组表现，就已经能提供现有文献没有的信息。"
                "结构化先验路线（\\cite{p5}）把先验从数据搬到结构，"
                "代价是引入了新的超参敏感性，目前缺少选择准则。",
            ]
        )
        return {
            "summary": summary,
            "themes": themes,
            "gaps": gaps,
            "method_landscape": method_landscape,
        }

    def _gen_novelty(self, prompt: str, system: str | None = None) -> dict:
        """s2_novelty：verdict / score / rationale / closest / overlap_risks / differentiators"""
        direction = extract_direction(prompt)
        short = _clip(direction, 60)
        verdict = _pick(prompt, ["novel", "incremental", "incremental", "duplicate"], "novelty")
        score_map = {"novel": 0.78, "incremental": 0.55, "duplicate": 0.28}
        base = score_map[verdict]
        jitter = (_decide(prompt, "novelty_score", 11) - 5) / 100.0
        score = max(0.05, min(0.97, base + jitter))
        closest = [
            {
                "title": f"A Controlled Study of {short}",
                "id": "arxiv:24{:02d}.{:05d}".format(
                    _decide(prompt, "nid1", 90) + 10,
                    _decide(prompt, "nid1b", 90000) + 10000,
                ),
                "year": 2023 + _decide(prompt, "ny1", 2),
                "why": "同样强调受控比较，但只在单一数据集上验证，未控制计算预算。",
            },
            {
                "title": f"Rethinking Evaluation for {short}",
                "id": "doi:10.1{:04d}/mock.{:03d}".format(
                    _decide(prompt, "nid2", 9000) + 1000,
                    _decide(prompt, "nid2b", 900) + 100,
                ),
                "year": 2022 + _decide(prompt, "ny2", 3),
                "why": "提出评测协议批评，但没有给出可执行的替代实验设计。",
            },
        ]
        overlap_risks = [
            "近邻工作已使用受控预算比较的表述，需要在方法细节上明确区分。",
            "若评测数据集与本工作重合，增益可能被解释为调参而非方法。",
        ]
        differentiators = [
            "显式分离「方法增益」与「调参增益」，并给出可复现的分离协议。",
            "把最差组精度与失败预测作为一等指标，而非事后分析。",
            "全部实验在 stdlib-only 的确定性脚本中完成，逐位可复现。",
        ]
        rationale = (
            f"判定为 {verdict}（score={score:.2f}）：核心机制与 {closest[0]['id']} 部分重叠，"
            "差异集中在预算控制与失败模式分析这两点上；若能在固定预算下给出可复现的"
            "对照结果，新颖性可支撑一次会议投稿。检索覆盖了标题与摘要层面的近邻，"
            "未发现完全相同的实验协议，因此不判为 duplicate。"
        )
        return {
            "verdict": verdict,
            "score": round(score, 3),
            "rationale": rationale,
            "closest": closest,
            "overlap_risks": overlap_risks,
            "differentiators": differentiators,
        }

    def _gen_idea(self, prompt: str, system: str | None = None) -> dict:
        """s2_ideas：Idea 形状（CONTRACTS §9 的必需键 + 模板要求的两个补充键）"""
        direction = extract_direction(prompt)
        short = _clip(direction, 36)
        n = _extract_int(prompt, ("max_ideas", "max ideas", "最多生成", "至多", "数量"), 4)
        n = max(1, min(8, n))
        angles = [
            {
                "title": f"预算受控的对照协议：{short}",
                "hypothesis": (
                    f"在固定 FLOPs/epoch 预算下，{short} 的方法增益中至少有 30% "
                    "来自调参而非结构本身。"
                ),
                "motivation": "文献中的增益归因不清，缺少受控比较。",
                "method_sketch": (
                    "对同一骨干网络施加等预算网格搜索，比较方法变体与调参基线的差值，"
                    "并把「调参增益」单独报告。"
                ),
                "novelty_claim": "首次把「调参增益」从「方法增益」中分离并给出可复现的分离协议。",
                "feasibility": "高：只依赖单机 CPU / 单卡小规模模型，单次实验 < 5 分钟。",
                "risks": ["网格搜索成本可能超预算", "预算度量方式本身有争议"],
                "expected_metrics": ["val_accuracy", "budget_normalized_gain"],
                "minimal_experiment": (
                    "python train.py --variant method --epochs 40 --seed 0 --out-dir runs/method"
                ),
                "required_resources": "单机 CPU，约 5 分钟，无外部数据集。",
            },
            {
                "title": f"失败模式的系统刻画：{short}",
                "hypothesis": f"{short} 的失败集中在少数语义子群，可用轻量探针提前预测。",
                "motivation": "负结果不被报告，导致方法边界不可知。",
                "method_sketch": (
                    "构造子群划分，训练一个仅用 logits 统计量的探针预测「是否失败」，"
                    "并报告最差组精度。"
                ),
                "novelty_claim": "把失败预测当作一等评测指标，而非事后分析。",
                "feasibility": "中：需要分组标注，可用启发式规则代替。",
                "risks": ["子群划分带主观性", "探针可能只是过拟合划分"],
                "expected_metrics": ["failure_auc", "worst_group_accuracy"],
                "minimal_experiment": (
                    "python train.py --variant method --epochs 40 --seed 1 --out-dir runs/probe"
                ),
                "required_resources": "单机 CPU，约 5 分钟。",
            },
            {
                "title": f"轻量适配器的稳定性：{short}",
                "hypothesis": f"在 {short} 中，适配器的初始化尺度比其容量更决定最终性能。",
                "motivation": "参数高效微调的方差大，复现困难。",
                "method_sketch": "扫描初始化尺度与秩，报告多种子方差与性能的帕累托前沿。",
                "novelty_claim": "给出「尺度优先于容量」的实证规律与一个免调参的默认尺度。",
                "feasibility": "高：小模型即可复现。",
                "risks": ["规律可能只在小模型上成立"],
                "expected_metrics": ["val_accuracy", "seed_variance"],
                "minimal_experiment": (
                    "python train.py --variant method --epochs 20 --seed 2 --out-dir runs/scale"
                ),
                "required_resources": "单机 CPU，约 3 分钟。",
            },
            {
                "title": f"数据效率的显式曲线：{short}",
                "hypothesis": f"{short} 的样本效率在低资源区间存在拐点，拐点位置由任务熵决定。",
                "motivation": "数据效率通常只报告单点，缺少曲线形状。",
                "method_sketch": "在 6 个训练集规模上重复实验，拟合双对数曲线并估计拐点。",
                "novelty_claim": "用任务熵预测拐点位置，给出可检验的定量关系。",
                "feasibility": "中：需要多次重训，但规模小。",
                "risks": ["拐点估计对噪声敏感"],
                "expected_metrics": ["sample_efficiency", "val_accuracy"],
                "minimal_experiment": (
                    "python train.py --variant method --epochs 20 --seed 0 --out-dir runs/scaling"
                ),
                "required_resources": "单机 CPU，约 10 分钟（6 次运行）。",
            },
            {
                "title": f"训练稳定性的早停代理：{short}",
                "hypothesis": f"验证损失的滑动方差可作为 {short} 的早停代理，优于固定 epoch。",
                "motivation": "固定 epoch 早停在小数据集上浪费算力。",
                "method_sketch": "定义方差代理指标，与 oracle 早停对比节省的算力与损失的精度。",
                "novelty_claim": "提供一个零超参的早停准则并量化其代价。",
                "feasibility": "高：改动仅涉及训练循环。",
                "risks": ["代理指标在噪声曲线上可能误触发"],
                "expected_metrics": ["epochs_saved", "val_accuracy"],
                "minimal_experiment": (
                    "python train.py --variant method --epochs 40 --seed 3 --out-dir runs/earlystop"
                ),
                "required_resources": "单机 CPU，约 5 分钟。",
            },
            {
                "title": f"可解释性的一致性检验：{short}",
                "hypothesis": f"{short} 模型的解释图在扰动下不一致，可用一致性作为正则项。",
                "motivation": "可解释性缺少可量化的一致性度量。",
                "method_sketch": "引入扰动一致性损失，衡量解释稳定性与精度之间的权衡。",
                "novelty_claim": "把解释一致性变成可优化目标而非事后可视化。",
                "feasibility": "中：需要实现解释图计算。",
                "risks": ["一致性提升可能不带来精度提升"],
                "expected_metrics": ["explanation_consistency", "val_accuracy"],
                "minimal_experiment": (
                    "python train.py --variant method --epochs 20 --seed 4 --out-dir runs/explain"
                ),
                "required_resources": "单机 CPU，约 5 分钟。",
            },
        ]
        ideas = []
        for i in range(n):
            angle = angles[i % len(angles)]
            idea = {"id": f"I{i + 1}"}
            idea.update(angle)
            if i >= len(angles):
                idea["id"] = f"I{i + 1}"
                idea["title"] = f"{angle['title']}（变体 {i + 1}）"
            ideas.append(idea)
        return {"ideas": ideas}

    def _gen_plan(self, prompt: str, system: str | None = None) -> dict:
        """s3_plan：objective/core_claim/dataset{}/baseline{}/milestones/metrics/ablation/..."""
        direction = extract_direction(prompt)
        short = _clip(direction, 60)
        milestones = [
            {
                "id": "M1",
                "name": "环境与数据落地",
                "description": f"生成 {short} 的确定性合成/小规模数据与训练脚手架。",
                "runs": 1,
                "est_minutes": 5,
                "success_criterion": "metrics.csv 产出，表头 epoch,loss,accuracy,f1,val_loss,val_accuracy 齐全",
                "depends_on": [],
            },
            {
                "id": "M2",
                "name": "基线复现",
                "description": "跑通 baseline 变体，记录学习曲线与多 seed 方差。",
                "runs": 3,
                "est_minutes": 5,
                "success_criterion": "基线 val_accuracy 稳定在 0.85~0.95，跨 seed 方向一致",
                "depends_on": ["M1"],
            },
            {
                "id": "M3",
                "name": "方法变体",
                "description": "在同等预算下运行方法变体（动量 + 权重衰减 + 学习率衰减）。",
                "runs": 3,
                "est_minutes": 5,
                "success_criterion": "方法变体在多数种子上不劣于基线，且至少一个指标有正增益",
                "depends_on": ["M2"],
            },
            {
                "id": "M4",
                "name": "消融与敏感性",
                "description": "逐个关闭动量 / 权重衰减 / 学习率衰减，定位增益来源。",
                "runs": 4,
                "est_minutes": 5,
                "success_criterion": "每个组件的贡献方向一致且可解释；无单点决定全部增益",
                "depends_on": ["M3"],
            },
            {
                "id": "M5",
                "name": "汇总与图表",
                "description": "聚合多 seed 指标，产出三线表与学习曲线图。",
                "runs": 1,
                "est_minutes": 5,
                "success_criterion": "生成至少 2 张图与 1 张 booktabs 三线表",
                "depends_on": ["M4"],
            },
        ]
        metrics = [
            {"name": "val_accuracy", "direction": "higher", "primary": True},
            {"name": "f1", "direction": "higher", "primary": False},
            {"name": "val_loss", "direction": "lower", "primary": False},
        ]
        ablation_matrix = [
            {"name": "momentum", "variants": ["0.0 (off)", "0.9 (on)"],
             "hypothesis": "动量平滑小批量梯度噪声，加速早期收敛。"},
            {"name": "weight_decay", "variants": ["0.0 (off)", "2e-4 (on)"],
             "hypothesis": "权重衰减抑制过拟合，改善验证集表现。"},
            {"name": "lr_schedule", "variants": ["constant", "0.25x-per-epoch decay"],
             "hypothesis": "学习率衰减带来更稳定的后期收敛。"},
            {"name": "noise_level", "variants": ["0.70", "0.80", "0.90"],
             "hypothesis": "任务越难，正则化的相对收益越大。"},
        ]
        return {
            "objective": (
                f"在固定训练预算下验证 {short} 的方法增益是否可归因于结构本身，"
                "并给出可复现的最小对照实验与失败边界。"
            ),
            "core_claim": (
                "动量与学习率衰减的组合在等预算条件下带来方向一致的小幅精度增益，"
                "且该增益在多个随机种子上稳定。"
            ),
            "dataset": {
                "name": "synthetic-controlled",
                "source": "pipeline 内置确定性生成器（stdlib random，固定 seed）",
                "size": "900 样本 / 8 维 / 3 类（25% 验证）",
                "split": "确定性划分：前 25% 作验证集",
            },
            "baseline": {
                "name": "plain-sgd",
                "description": "同架构线性 softmax，纯 SGD，无动量、无权重衰减、恒定学习率。",
                "expected_metrics": "val_accuracy 0.85~0.95，val_loss 0.15~0.30",
            },
            "milestones": milestones,
            "metrics": metrics,
            "ablation_matrix": ablation_matrix,
            "compute_budget_hours": round(1.0 + _decide(prompt, "budget", 20) / 10.0, 1),
            "risks": [
                {"risk": "合成数据的结论可能无法迁移到真实基准",
                 "mitigation": "明确把结论限定在受控合成设定，并在 limitations 中声明。"},
                {"risk": "多种子方差可能掩盖真实差异",
                 "mitigation": "每个配置至少 3 个 seed，报告均值与标准差，只声明方向一致的差异。"},
                {"risk": "预算归一化方式的选择会影响结论方向",
                 "mitigation": "同时报告 epoch 归一化与样本数归一化两种口径。"},
            ],
            "code_plan": [
                {"file": "train.py",
                 "purpose": "单文件可运行实验脚本，支持 --variant/--epochs/--seed/--out-dir，"
                            "输出 metrics.csv 与 metrics.jsonl"},
            ],
        }

    def _gen_debug(self, prompt: str, system: str | None = None) -> dict:
        """s4_debug：diagnosis/root_cause/files/commands_to_verify/confidence/validity_note"""
        direction = extract_direction(prompt, max_len=60)
        candidates = self._extract_original_files(prompt)
        bits = []
        root_cause = "运行环境与脚本假设不一致（路径 / 依赖 / 参数）。"
        if "ModuleNotFoundError" in prompt or "ImportError" in prompt:
            bits.append("缺少可选依赖或相对导入在脚本直接执行时失效。")
            root_cause = "脚本导入了当前环境不存在的模块。"
        if "SyntaxError" in prompt:
            bits.append("语法错误：括号/缩进不平衡或字符串未闭合。")
            root_cause = "脚本存在语法错误，Python 无法编译该文件。"
        if "FileNotFoundError" in prompt:
            bits.append("输入文件路径不存在，脚本没有先创建输出目录。")
            root_cause = "输出目录在写文件前未被创建。"
        if "KeyError" in prompt:
            bits.append("结果字典缺少期望的键，需要给出默认值。")
            root_cause = "代码假设了并不存在的字典键。"
        if "ZeroDivisionError" in prompt:
            bits.append("除零：小样本划分导致分母为 0，需要保护。")
            root_cause = "样本划分过小导致指标计算分母为 0。"
        if "unrecognized arguments" in prompt.lower() or "error: unrecognized" in prompt.lower():
            bits.append("命令行参数与脚本支持的参数不匹配。")
            root_cause = "脚本 CLI 未实现流水线传入的参数（如 --variant / --out-dir）。"
        if not bits:
            bits.append(
                "未能在 traceback 中定位唯一根因；最可能是路径/依赖假设与运行环境不一致。"
            )
        bits.append(
            f"建议：对 {direction} 的实验脚本保持 stdlib-only，写文件前显式 makedirs，"
            "并让 CLI 接受流水线会传入的全部参数。"
        )

        files: list[dict[str, str]] = []
        if candidates:
            for path, content in candidates:
                files.append({"path": path, "content": _repair_python(content)})
            confidence = 0.72 if len(candidates) == 1 else 0.6
        else:
            confidence = 0.35

        return {
            "diagnosis": " ".join(bits),
            "root_cause": root_cause,
            "files": files,
            "commands_to_verify": [
                "python train.py --variant method --epochs 5 --seed 0 --out-dir runs/smoke",
            ],
            "confidence": confidence,
            "validity_note": (
                "修复只补齐目录创建 / CLI 参数与 import，不改变模型的训练目标、"
                "数据划分或早停准则，因此不削弱实验结论。"
            ),
        }

    @staticmethod
    def _extract_original_files(prompt: str) -> list[tuple[str, str]]:
        """从 prompt 里捞出被贴进来的代码文件（``` 围栏或整段脚本）。"""
        out: list[tuple[str, str]] = []
        for m in re.finditer(
            r"```[ \t]*([A-Za-z0-9_+.-]*)[ \t]*\r?\n(.*?)```", prompt or "", re.DOTALL
        ):
            lang = (m.group(1) or "").lower()
            if lang and lang not in ("python", "py", "python3"):
                continue
            body = m.group(2)
            if self_looks_like_python(body):
                out.append(("train.py", body.rstrip() + "\n"))
        if out:
            return out[:1]
        # 退化：整段 prompt 本身就是脚本
        if self_looks_like_python(prompt) and ("def " in prompt) and len(prompt) > 200:
            return [("train.py", prompt.rstrip() + "\n")]
        return []

    def _gen_code(self, prompt: str, system: str | None = None) -> dict:
        """s4_codegen：files[{path,content,purpose}] + entrypoint + run_command + notes"""
        direction = extract_direction(prompt, max_len=60)
        header = (
            "# Auto-generated by MockBackend (offline heuristic backend).\n"
            f"# Research direction: {direction}\n"
        )
        content = header + TRAIN_PY_TEMPLATE
        run_command = (
            "python train.py --variant {variant} --epochs {epochs} "
            "--seed {seed} --out-dir runs/{variant}"
        )
        return {
            "files": [
                {
                    "path": "train.py",
                    "content": content,
                    "purpose": (
                        "确定性合成分类训练脚本：支持 --variant/--epochs/--seed/--out-dir，"
                        "写出 metrics.csv 与 metrics.jsonl"
                    ),
                }
            ],
            "entrypoint": "train.py",
            "run_command": run_command,
            "notes": (
                "仅使用标准库；默认 40 epochs、噪声 0.8，单次运行 < 10 秒。"
                "重复执行同一命令得到逐位相同的结果（固定 seed）。"
            ),
        }

    def _gen_writing(self, prompt: str, system: str | None = None) -> dict:
        """s6_section / s6_abstract：latex + citations_used + claims + word_count"""
        # 优先复用提示词里已经给定的假设标题/研究目标：真实 LLM 也该这么做，
        # 而退化成「抽提示词里最长的行」会把模板标签抄进标题。
        given = _extract_labeled_value(prompt, ("title", "标题", "objective", "目标"))
        direction = given or extract_direction(prompt)
        short = _clip(direction, 60)
        section = _pick(
            prompt,
            ["method", "experiments", "introduction", "related_work", "abstract"],
            "section_name",
        )
        if "title" in prompt.lower() or "摘要" in prompt or "abstract" in prompt.lower():
            title = f"Controlled Budget Comparison for {_clip(short, 50)}"
            abstract = (
                "我们研究一个被普遍忽略的问题：在固定训练预算下，报告的方法增益有多少"
                "来自结构本身，又有多少只是调参的结果。为此我们提出一个受控比较协议："
                "同一骨干网络、同一数据划分、同一预算，只改变显式声明的组件。"
                "实验在确定性合成数据上完成，全部随机源均被显式播种，结果可逐位复现。"
                "我们发现动量与学习率衰减的组合带来方向一致的小幅验证准确率增益，"
                "而单纯增加容量并不改善验证集表现。我们还把最差组精度与失败率作为"
                "一等指标报告，使方法边界变得可见。全部结论限定在受控合成设定内，"
                "真实基准上的迁移性留待后续工作验证。"
            )
            return {
                "title": title,
                "title_candidates": [
                    title,
                    f"Separating Method Gains from Tuning Gains in {_clip(short, 40)}",
                    f"A Reproducible Protocol for {_clip(short, 40)}",
                ],
                "abstract": abstract,
                "keywords": ["controlled comparison", "compute budget",
                             "reproducibility", "learning rate schedule",
                             "synthetic benchmark"],
                "contributions": [
                    "把「方法增益」与「调参增益」显式分离的可复现协议。",
                    "在等预算条件下给出动量与学习率衰减的方向一致性证据。",
                    "把最差组精度与失败率作为一等评测指标。",
                ],
                "section": abstract,
            }

        latex = "\n\n".join(
            [
                f"\\section{{{section.capitalize()}}}",
                "我们提出一个受控的比较协议，用固定训练预算隔离「方法结构」与「调参」"
                "两类增益来源。模型采用线性 softmax 分类器作为骨架，方法变体仅引入动量、"
                "权重衰减与学习率衰减三个正交组件，从而保证任何性能差异都可归因到"
                "显式声明的组件上。",
                "训练使用确定性合成数据（固定 seed），所有随机性来源都被显式播种，"
                "因此实验可被逐位复现。评价指标包括训练/验证准确率、宏平均 F1，"
                "以及达到目标精度所需的 epoch 数。",
                "为区分偶然波动与真实增益，每个配置在三个种子上重复，报告均值与标准差，"
                "并在结论中只声明方向一致的差异；方向不一致的指标一律记为不显著。",
            ]
        )
        word_count = len(re.findall(r"[A-Za-z0-9_]+", latex)) + sum(
            1 for ch in latex if "\u4e00" <= ch <= "\u9fff"
        )
        claims = [
            {"claim": "在固定预算下，动量与学习率衰减的组合带来方向一致的精度增益。",
             "evidence_id": "E1"},
            {"claim": "方法变体达到目标精度所需 epoch 数不多于纯 SGD 基线。",
             "evidence_id": "E2"},
        ]
        return {
            "section_key": section,
            "section": latex,
            "latex": latex,
            "citations_used": [],
            "claims": claims,
            "citations_needed": [
                "参数高效适配的代表性工作（用于 related work 对比）",
                "受控比较 / 公平评测的方法学文献",
                "合成数据可复现性实践的先前讨论",
            ],
            "word_count": word_count,
        }

    def _gen_review(self, prompt: str, system: str | None = None) -> dict:
        """s8_review：score/verdict/strengths/weaknesses/per_criterion/confidence/..."""
        round_no = _extract_round(prompt, default=1)
        round_no = max(1, min(round_no, 9))
        # 单调递增：1 -> ~6.0, 2 -> ~7.0, 3 -> ~7.8，更高轮次逐步逼近 8.6。
        # 分数只依赖轮次（抖动是同一 prompt 的常数），因此 loop 一定能收敛。
        table = []
        for r in range(1, 10):
            raw = min(8.6, 6.0 + 1.0 * (r - 1))
            if r >= 3:
                raw -= 0.15 * (r - 3) * (r - 2) / 2.0
            if table:
                raw = max(raw, table[-1] + 0.05)
            table.append(raw)
        score = table[round_no - 1]
        jitter = (_decide(prompt, "review_jitter", 5) - 2) / 100.0
        score = max(1.0, min(9.4, score + jitter))
        if score >= 7.6:
            verdict = "ready"
        elif score >= 6.8:
            verdict = "almost"
        else:
            verdict = "revise"
        per_criterion = {
            "novelty": round(min(9.5, score - 0.3), 2),
            "rigor": round(min(9.5, score + 0.1), 2),
            "clarity": round(min(9.5, score - 0.1), 2),
            "experiments": round(min(9.5, score - 0.5), 2),
            "reproducibility": round(min(9.5, score + 0.2), 2),
        }
        # 剩余的最小修复随轮次递减（每轮修掉一个），便于 loop 单调收敛
        remaining_fixes = max(0, 3 - (round_no - 1))
        return {
            "round": round_no,
            "score": round(score, 2),
            "verdict": verdict,
            "summary": (
                f"第 {round_no} 轮评审：论文的问题设定清晰，受控比较协议是主要卖点；"
                f"当前得分 {score:.1f}（{verdict}）。"
                "剩余风险集中在统计显著性与结论外推范围，"
                f"本轮认定的必改项还有 {remaining_fixes} 个。"
            ),
            "strengths": [
                {"point": "受控预算比较的思路清晰，能把方法增益与调参增益分开。",
                 "evidence": "method 章节的分离协议与等预算设置"},
                {"point": "实验完全确定性可复现，所有随机源都已播种。",
                 "evidence": "train.py 中固定 seed 与确定性数据生成"},
                {"point": "消融覆盖了三个正交组件，结论归因明确。",
                 "evidence": "ablation_matrix 与对应结果表"},
            ],
            "weaknesses": [
                {"point": "仅使用合成数据，缺少至少一个小规模真实基准的验证。",
                 "severity": "major" if remaining_fixes >= 2 else "minor",
                 "evidence": "dataset 小节只报告 synthetic-controlled",
                 "min_fix": "补充一个 sklearn 自带的小型真实数据集对照实验。",
                 "location": "experiments"},
                {"point": "多种子重复次数偏少，置信区间较宽。",
                 "severity": "major" if remaining_fixes >= 3 else "minor",
                 "evidence": "主表只报告 3 个 seed",
                 "min_fix": "把种子数提升到 5 并在主表中报告标准差。",
                 "location": "results"},
                {"point": "缺少与现有方法的直接数值对比表。",
                 "severity": "major" if remaining_fixes >= 3 else "minor",
                 "evidence": "related work 只做定性对比",
                 "min_fix": "补充一张与两个代表性基线的三线表对比。",
                 "location": "related_work"},
            ],
            "questions": [
                "结论在噪声水平变化时是否保持方向一致？",
                "预算归一化方式改变后，主要结论是否会翻转？",
            ],
            "min_fixes": [
                "补充至少一个小规模真实数据集的对照实验。",
                "把种子数提升到 5 并在主表中报告标准差。",
            ],
            "per_criterion": per_criterion,
            "confidence": round(0.55 + 0.05 * min(round_no, 5), 2),
            "recommendation": (
                "接受（ready）" if verdict == "ready"
                else ("小修后接受（almost）" if verdict == "almost" else "大修（revise）")
            ),
        }

    def _gen_analysis(self, prompt: str, system: str | None = None) -> dict:
        """s5_analysis：claim_evidence / findings / negative_results / limitations / ..."""
        direction = extract_direction(prompt)
        short = _clip(direction, 60)
        return {
            "claim_evidence": [
                {
                    "claim": "动量与学习率衰减的组合在等预算下带来方向一致的精度增益。",
                    "verdict": "supported",
                    "evidence": "method 与 baseline 的 val_accuracy 在多个 seed 上方向一致。",
                    "numbers": ["val_accuracy +0.4~0.5 个百分点", "3 个 seed 方向一致"],
                },
                {
                    "claim": "增益主要来自优化器侧组件，而非模型容量。",
                    "verdict": "partially_supported",
                    "evidence": "消融显示去掉动量后增益基本消失，但容量维度未做完整扫描。",
                    "numbers": ["去掉 momentum 后 val_accuracy 回落约 0.4 个百分点"],
                },
                {
                    "claim": "方法在更难的噪声设定下收益更大。",
                    "verdict": "inconclusive",
                    "evidence": "高噪声下两种变体都接近 0.90，差异被方差淹没。",
                    "numbers": ["noise=0.9 时差距 < 0.2 个百分点"],
                },
            ],
            "findings": [
                {"finding": f"{short} 的增益集中在优化器侧，而非模型容量。",
                 "evidence": "消融矩阵中 momentum 与 lr_schedule 的贡献最大。",
                 "significance": "提示后续工作应优先报告优化器配方而非堆容量。"},
                {"finding": "验证集在 20 epoch 后进入平台期，训练损失仍在下降。",
                 "evidence": "metrics.csv 中 val_loss 与 loss 的分叉。",
                 "significance": "固定 epoch 早停会浪费算力，早停准则有实际价值。"},
            ],
            "negative_results": [
                "权重衰减单独开启时没有观察到可区分于噪声的增益。",
                "高噪声设定下两种变体无法区分，说明方法收益有适用边界。",
            ],
            "limitations": [
                "结论基于确定性合成数据，真实数据的分布偏移未被覆盖。",
                "预算度量（epoch 数）不是严格的 FLOPs 归一化。",
                "仅比较了线性骨架，未验证更深的网络是否同样成立。",
            ],
            "threats_to_validity": [
                "3 个 seed 的方差估计不稳健，方向一致但幅度不可靠。",
                "合成数据的类中心由固定 seed 生成，可能对初始化尺度不敏感。",
            ],
            "figure_discussion": [
                {"figure": "learning_curves",
                 "message": "method 的验证曲线上升更快且波动更小。"},
                {"figure": "comparison",
                 "message": "主指标上 method 小幅领先 baseline，差距小于方差。"},
            ],
        }

    def _gen_fallback(self, prompt: str, system: str | None = None) -> dict:
        block = _largest_block(prompt)
        return {"text": _clip(block, 800), "ok": True}
