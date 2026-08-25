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
    found = pg.search_headers("型号")
    assert any(r["table_id"] == "t2" for r in found)


@pytest.mark.integration
def test_register_and_query_versions(pg):
    pg.register_version("d1", "v1.0", "2025-01-01", "a.pdf")
    pg.register_version("d1", "v2.0", "2026-06-01", "a.pdf")
    versions = pg.versions("d1")
    assert versions[0]["version"] == "v2.0"  # 生效日期降序
    assert len(versions) == 2
