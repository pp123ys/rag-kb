import psycopg
from psycopg.types.json import Jsonb

from ragkb.models import TableData


class PgTableIndexer:
    """表格 B 路 + 版本登记：行列数据入 PostgreSQL，支持按表头/table_id 查询。"""

    def __init__(self, dsn: str):
        self._dsn = dsn

    def init_schema(self):
        with psycopg.connect(self._dsn) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tables_index (
                    table_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    headers JSONB NOT NULL,
                    rows JSONB NOT NULL,
                    source TEXT NOT NULL,
                    headers_text TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_versions (
                    doc_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    effective_date TEXT NOT NULL,
                    source TEXT NOT NULL,
                    PRIMARY KEY (doc_id, version)
                )
            """)
            conn.commit()

    def upsert(self, table: TableData):
        headers_text = " ".join(table.headers)
        with psycopg.connect(self._dsn) as conn:
            conn.execute(
                """
                INSERT INTO tables_index (table_id, name, headers, rows, source, headers_text)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (table_id) DO UPDATE SET
                    name = EXCLUDED.name, headers = EXCLUDED.headers,
                    rows = EXCLUDED.rows, source = EXCLUDED.source,
                    headers_text = EXCLUDED.headers_text
                """,
                (table.table_id, table.name, Jsonb(table.headers),
                 Jsonb(table.rows), table.source, headers_text),
            )
            conn.commit()

    def register_version(self, doc_id: str, version: str,
                         effective_date: str, source: str):
        with psycopg.connect(self._dsn) as conn:
            conn.execute(
                """
                INSERT INTO document_versions (doc_id, version, effective_date, source)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (doc_id, version) DO UPDATE SET
                    effective_date = EXCLUDED.effective_date,
                    source = EXCLUDED.source
                """,
                (doc_id, version, effective_date, source),
            )
            conn.commit()

    def versions(self, doc_id: str) -> list[dict]:
        with psycopg.connect(self._dsn) as conn:
            rows = conn.execute(
                "SELECT version, effective_date FROM document_versions "
                "WHERE doc_id = %s ORDER BY effective_date DESC",
                (doc_id,)).fetchall()
        return [{"version": r[0], "effective_date": r[1]} for r in rows]

    def query(self, table_id: str) -> list[dict] | None:
        with psycopg.connect(self._dsn) as conn:
            row = conn.execute(
                "SELECT headers, rows FROM tables_index WHERE table_id = %s",
                (table_id,)).fetchone()
        if not row:
            return None
        # psycopg3 读取 jsonb 时已自动解析为 Python 对象，无需再 json.loads
        headers = row[0]
        return [dict(zip(headers, r)) for r in row[1]]

    def search_headers(self, keyword: str) -> list[dict]:
        with psycopg.connect(self._dsn) as conn:
            rows = conn.execute(
                "SELECT table_id, name, source FROM tables_index "
                "WHERE headers_text ILIKE %s", (f"%{keyword}%",)).fetchall()
        return [{"table_id": r[0], "name": r[1], "source": r[2]} for r in rows]
