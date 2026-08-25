# tests/test_reranker.py
import pytest

from ragkb.models import Chunk
from ragkb.reranker.reranker import Reranker


class _FakeReranker:
    """假打分引擎：文本越长分越高（与 CrossEncoder.predict 同形）。"""
    def predict(self, pairs):
        return [len(text) for _, text in pairs]


def _chunk(cid, text):
    return Chunk(chunk_id=cid, doc_id="d1", doc_type="pdf", source="a.pdf",
                 text=text, version="v1.0", effective_date="2026-01-01")


@pytest.mark.model
def test_real_reranker_reorders(settings):
    r = Reranker(model_name=settings.rerank_model)
    chunks = [_chunk("a", "短"), _chunk("b", "这是一段很长的相关内容")]
    out = r.rerank("查询", chunks)
    assert out[0].chunk_id == "b"


def test_reranker_interface_with_fake():
    r = Reranker(model_name="fake", reranker=_FakeReranker())
    chunks = [_chunk("a", "短"), _chunk("b", "长内容长内容长内容")]
    out = r.rerank("查询", chunks)
    assert out[0].chunk_id == "b"
    assert {c.chunk_id for c in out} == {"a", "b"}


def test_reranker_empty_chunks_returns_empty():
    r = Reranker(model_name="fake", reranker=_FakeReranker())
    assert r.rerank("查询", []) == []
