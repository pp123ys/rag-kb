from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根（src/ragkb/config.py → rag-kb/）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """全局配置，可用环境变量覆盖（前缀 RAGKB_）。"""

    model_config = SettingsConfigDict(env_prefix="RAGKB_", env_file=".env")

    # 检索存储（默认嵌入式，免 Docker；设 qdrant_url 并清空 qdrant_path 可切远程）
    qdrant_url: str = "http://localhost:6333"
    qdrant_path: str = "data/qdrant"  # 非空=嵌入式本地模式（默认）；空=连 qdrant_url
    collection_name: str = "chunks"
    vector_size: int = 1024  # BGE-M3 dense 维度

    # 表格/版本存储（默认嵌入式 SQLite；设 pg_dsn 可切 PostgreSQL）
    pg_dsn: str = ""  # 空=SQLite（默认）；非空=PostgreSQL 连接串
    sqlite_path: str = "data/ragkb.db"

    # 图片存储（默认本地目录；设 minio_endpoint 可切 MinIO）
    minio_endpoint: str = ""  # 空=本地目录（默认）；非空=MinIO 服务地址
    minio_access_key: str = "ragkb"
    minio_secret_key: str = "ragkb-secret"
    minio_bucket: str = "ragkb-images"
    images_dir: str = "data/images"

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
    min_relevance_score: float = 0.1   # 重排分数阈值：低于则判定「没有找到」（防幻觉）


@lru_cache
def get_settings() -> Settings:
    return Settings()
