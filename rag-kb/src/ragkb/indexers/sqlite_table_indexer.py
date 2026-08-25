"""SQLite 表格索引与版本登记（嵌入式，免 Docker）。

接口与 PgTableIndexer 完全一致（upsert/register_version/versions/query/
search_headers/init_schema），由 get_table_indexer 工厂按配置选择：
- settings.pg_dsn 非空 → PgTableIndexer（远程 PostgreSQL）
- 否则 → SqliteTableIndexer（嵌入式，数据落盘 settings.sqlite_path）
"""
import json
import sqlite3
from pathlib import Path

from ragkb.models import TableData


class SqliteTableIndexer:
    """表格 B 路 + 版本登记：行列数据入 SQLite，支持按表头/table_id 查询。"""

    def __init__(self, path: str):
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tables_index (
                    table_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    headers TEXT NOT NULL,      -- JSON 数组
                    rows TEXT NOT NULL,         -- JSON 二维数组
                    source TEXT NOT NULL,
                    headers_text TEXT NOT NULL,
                    doc_id TEXT NOT NULL DEFAULT ''
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
            # 旧表迁移：doc_id 列是删除/归属用，早期 schema 没有
            cols = {r["name"] for r in conn.execute(
                "PRAGMA table_info(tables_index)")}
            if "doc_id" not in cols:
                conn.execute(
                    "ALTER TABLE tables_index "
                    "ADD COLUMN doc_id TEXT NOT NULL DEFAULT ''")
            conn.commit()

    def upsert(self, table: TableData):
        # 先校验行列数一致，避免 query 阶段 dict(zip(headers, row)) 静默丢列
        for i, row in enumerate(table.rows, start=1):
            if len(row) != len(table.headers):
                raise ValueError(
                    f"表格 {table.table_id} 第 {i} 行 "
                    f"列数({len(row)}) 与表头列数({len(table.headers)})不一致")
        headers_text = " ".join(table.headers)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO tables_index (table_id, name, headers, rows, source, headers_text, doc_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (table_id) DO UPDATE SET
                    name = excluded.name, headers = excluded.headers,
                    rows = excluded.rows, source = excluded.source,
                    headers_text = excluded.headers_text,
                    doc_id = excluded.doc_id
                """,
                (table.table_id, table.name,
                 json.dumps(table.headers, ensure_ascii=False),
                 json.dumps(table.rows, ensure_ascii=False),
                 table.source, headers_text, table.doc_id),
            )
            conn.commit()

    def register_version(self, doc_id: str, version: str,
                         effective_date: str, source: str):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO document_versions (doc_id, version, effective_date, source)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (doc_id, version) DO UPDATE SET
                    effective_date = excluded.effective_date,
                    source = excluded.source
                """,
                (doc_id, version, effective_date, source),
            )
            conn.commit()

    def versions(self, doc_id: str) -> list[dict]:
        # 排序依赖 ISO YYYY-MM-DD 字符串字典序即时间序的约定
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT version, effective_date FROM document_versions "
                "WHERE doc_id = ? ORDER BY effective_date DESC",
                (doc_id,)).fetchall()
        return [{"version": r["version"], "effective_date": r["effective_date"]}
                for r in rows]

    def query(self, table_id: str) -> list[dict] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT headers, rows FROM tables_index WHERE table_id = ?",
                (table_id,)).fetchone()
        if not row:
            return None
        headers = json.loads(row["headers"])
        return [dict(zip(headers, r)) for r in json.loads(row["rows"])]

    def search_headers(self, keyword: str) -> list[dict]:
        # 转义 LIKE 通配符（% 和 _）与转义符本身，避免用户关键字被当作通配符
        escaped = (keyword.replace("\\", "\\\\")
                   .replace("%", "\\%").replace("_", "\\_"))
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT table_id, name, source FROM tables_index "
                "WHERE headers_text LIKE ? ESCAPE '\\'",
                (f"%{escaped}%",)).fetchall()
        return [{"table_id": r["table_id"], "name": r["name"],
                 "source": r["source"]} for r in rows]

    def delete_by_doc_id(self, doc_id: str) -> dict:
        """删除某文档的全部表格与版本登记，返回删除条数。"""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM tables_index WHERE doc_id = ?", (doc_id,))
            tables = cur.rowcount
            cur = conn.execute(
                "DELETE FROM document_versions WHERE doc_id = ?", (doc_id,))
            versions = cur.rowcount
            conn.commit()
        return {"tables": tables, "versions": versions}
