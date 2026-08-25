"""离线检索评测入口：评测集 + 指标（Recall@K / MRR）。

用法:
    py -3.12 eval/run_eval.py [eval_set.jsonl]

加载评测集，对每条 query 用真实检索链路（Qdrant 双路召回 + RRF，配置走
get_settings）取 top-K chunk ids，与 gold（chunk id）对比计算 Recall@K 与 MRR。
embedder 懒加载，首次运行会下载 BGE-M3 模型。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragkb.config import get_settings
from ragkb.embedder import Embedder
from ragkb.eval.metrics import mrr, recall_at_k
from ragkb.indexers import QdrantIndexer
from ragkb.retriever import Retriever


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
    settings = get_settings()
    indexer = QdrantIndexer(settings)
    embedder = Embedder(settings.embed_model)
    retriever = Retriever(indexer=indexer)
    ranked_lists, golds = [], []
    for item in items:
        query = item["query"]
        gold = item["gold"][0]  # 单 gold 假设
        query_vec = embedder.embed([query])[0].tolist()
        hits = retriever.retrieve(query, query_vec, top_m=20)
        ranked = [c.chunk_id for c in hits]
        ranked_lists.append(ranked)
        golds.append(gold)
    print(f"评测集 {len(items)} 条")
    for k in (5, 10):
        print(f"Recall@{k}: {recall_at_k(ranked_lists, golds, k):.3f}")
    print(f"MRR: {mrr(ranked_lists, golds):.3f}")


if __name__ == "__main__":
    main()
