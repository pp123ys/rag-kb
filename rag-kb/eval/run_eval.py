"""离线检索评测入口（任务 17：评测集 + 指标，纯标准库）。

用法:
    py -3.12 eval/run_eval.py [eval_set.jsonl]

当前为骨架：加载评测集并复用 ragkb.eval.metrics 中的 Recall@K / MRR；
接入真实 retriever（对每条 query 取 top-K chunk ids 后与 gold 对比）
是后续任务，metrics 与评测集格式为本任务的交付物。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragkb.eval.metrics import mrr, recall_at_k  # noqa: E402


def load_set(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    items = load_set(sys.argv[1] if len(sys.argv) > 1 else "eval/eval_set.jsonl")
    # 实际评测：对每条 query 调 retriever，得到 ranked chunk ids，与 gold 对比。
    # 骨架（接入 retriever 后填充）：
    print(f"评测集共 {len(items)} 条；接入 retriever 后计算 Recall@K 与 MRR。")


if __name__ == "__main__":
    main()
