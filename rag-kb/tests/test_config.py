import os
from ragkb.config import Settings


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("RAGKB_QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("RAGKB_EMBED_MODEL", "BAAI/bge-m3")
    s = Settings()
    assert s.qdrant_url == "http://localhost:6333"
    assert s.embed_model == "BAAI/bge-m3"
    assert s.collection_name == "chunks"


def test_settings_defaults():
    s = Settings()
    assert s.rrf_k == 60
    assert s.top_n_rerank == 20
    assert s.top_m_context == 5
