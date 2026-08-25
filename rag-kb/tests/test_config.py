import os
from ragkb.config import Settings


def test_settings_from_env(monkeypatch):
    # 与默认值不同的覆盖，确保 env 覆盖真实生效
    monkeypatch.setenv("RAGKB_QDRANT_URL", "http://env-host:9999")
    monkeypatch.setenv("RAGKB_EMBED_MODEL", "fake/model")
    # 无 RAGKB_ 前缀的环境变量应被忽略
    monkeypatch.setenv("QDRANT_URL", "http://no-prefix:1111")
    s = Settings()
    assert s.qdrant_url == "http://env-host:9999"
    assert s.embed_model == "fake/model"
    # 未覆盖的默认值仍然生效
    assert s.collection_name == "chunks"


def test_settings_defaults():
    s = Settings()
    assert s.rrf_k == 60
    assert s.top_n_rerank == 20
    assert s.top_m_context == 5
