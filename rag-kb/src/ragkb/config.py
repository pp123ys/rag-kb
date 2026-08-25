from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根（src/ragkb/config.py → rag-kb/）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """全局配置，可用环境变量覆盖（前缀 RAGKB_）。"""

    model_config = SettingsConfigDict(env_prefix="RAGKB_", env_file=".env")

    # 检索存储
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "chunks"
    vector_size: int = 1024  # BGE-M3 dense 维度

    # PostgreSQL（表格索引）
    pg_dsn: str = "postgresql://ragkb:ragkb@localhost:5432/ragkb"

    # MinIO（原图存储）
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "ragkb"
    minio_secret_key: str = "ragkb-secret"
    minio_bucket: str = "ragkb-images"

    # 模型（下载与加载均走项目内目录，不占 C 盘用户缓存）
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    ocr_lang: str = "ch"
    model_cache_dir: str = str(_PROJECT_ROOT / "models")

    # 切块
    chunk_target_chars: int = 400      # 中文按字符近似 token 预算
    chunk_overlap_chars: int = 60      # overlap 约 10–15%
    chunk_max_chars: int = 800

    # 检索
    recall_k: int = 50                 # 每路召回数
    rrf_k: int = 60                    # RRF 常数
    top_n_rerank: int = 20             # 融合后送重排的条数
    top_m_context: int = 5             # 重排后进上下文的条数


@lru_cache
def get_settings() -> Settings:
    return Settings()
