import pytest

from ragkb.indexers.pg_table_indexer import PgTableIndexer
from ragkb.models import TableData


@pytest.fixture
def pg(settings):
    idx = PgTableIndexer(settings.pg_dsn)
    idx.init_schema()
    return idx


@pytest.mark.integration
def test_upsert_and_query_by_column(pg):
    pg.upsert(TableData(table_id="t1", name="报价表", headers=["型号", "单价"],
                        rows=[["A-100", "99"]], source="a.xlsx:报价"))
    rows = pg.query(table_id="t1")
    assert rows[0]["型号"] == "A-100"
    assert rows[0]["单价"] == "99"


@pytest.mark.integration
def test_search_by_header(pg):
    pg.upsert(TableData(table_id="t2", name="库存表", headers=["型号", "数量"],
                        rows=[["B-200", "50"]], source="b.xlsx:库存"))
    found = pg.search_headers("数量")
    assert any(r["table_id"] == "t2" for r in found)
    # t1（报价表，表头 型号/单价）无 "数量" 表头，不应被命中
    assert not any(r["table_id"] == "t1" for r in found)


@pytest.mark.integration
def test_search_headers_escapes_wildcards(pg):
    pg.upsert(TableData(table_id="t3", name="表", headers=["型号_1", "单价"],
                        rows=[["A-1", "9"]], source="c.xlsx:表"))
    # 下划线不应作通配符：搜 "型号_" 只匹配字面 "型号_"，t3 命中
    found = pg.search_headers("型号_")
    assert any(r["table_id"] == "t3" for r in found)
    # 搜 "型号X" 无字面匹配，不应误命中 "型号_1"（X 无匹配）
    none_found = pg.search_headers("型号X")
    assert not any(r["table_id"] == "t3" for r in none_found)


def test_upsert_rejects_misaligned_row():
    # 行列数不一致应在建立 DB 连接之前就抛错，无需外部服务
    idx = PgTableIndexer("postgresql://fake:fake@127.0.0.1:1/fake")
    table = TableData(table_id="t9", name="错表", headers=["型号", "单价"],
                      rows=[["A-1", "9", "多余列"]], source="x.xlsx:错")
    with pytest.raises(ValueError, match="列数"):
        idx.upsert(table)


@pytest.mark.integration
def test_register_and_query_versions(pg):
    pg.register_version("d1", "v1.0", "2025-01-01", "a.pdf")
    pg.register_version("d1", "v2.0", "2026-06-01", "a.pdf")
    versions = pg.versions("d1")
    assert versions[0]["version"] == "v2.0"  # 生效日期降序
    assert len(versions) == 2
