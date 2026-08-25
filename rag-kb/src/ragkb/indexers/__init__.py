from ragkb.indexers.local_image_store import LocalImageStore
from ragkb.indexers.minio_store import MinioImageStore
from ragkb.indexers.pg_table_indexer import PgTableIndexer
from ragkb.indexers.qdrant_indexer import QdrantIndexer
from ragkb.indexers.sqlite_table_indexer import SqliteTableIndexer

__all__ = ["QdrantIndexer", "PgTableIndexer", "SqliteTableIndexer",
           "MinioImageStore", "LocalImageStore",
           "get_table_indexer", "get_image_store"]


def get_table_indexer(settings):
    """表格/版本存储工厂：pg_dsn 非空 → PostgreSQL，否则 → SQLite（嵌入式）。

    返回前自动建表（init_schema 幂等），生产路径无需手动初始化。
    """
    if settings.pg_dsn:
        idx = PgTableIndexer(settings.pg_dsn)
    else:
        idx = SqliteTableIndexer(settings.sqlite_path)
    idx.init_schema()
    return idx


def get_image_store(settings):
    """图片存储工厂：minio_endpoint 非空 → MinIO，否则 → 本地目录（嵌入式）。"""
    if settings.minio_endpoint:
        return MinioImageStore(settings)
    return LocalImageStore(settings.images_dir)
