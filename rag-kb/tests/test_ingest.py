# 入库编排流水线：全部注入 fake，只验证编排逻辑（解析→OCR→切块→嵌入→入库→版本登记）。
import os
import subprocess
import sys
from pathlib import Path

from ragkb.models import ImageData, ParsedDocument, TableData
from ragkb.ocr import OCRUnavailableError
from ragkb.pipeline.ingest import IngestPipeline


class _FakeParser:
    def parse(self, path, doc_id, source, **meta):
        return ParsedDocument(doc_id=doc_id, doc_type="pdf", source=source,
                              text="产品型号 A-100 单价 99 元。",
                              tables=[TableData(table_id="t1", name="报价",
                                                headers=["型号", "单价"],
                                                rows=[["A-100", "99"]],
                                                source="a.pdf:1")],
                              images=[])


class _FakeImageParser(_FakeParser):
    def parse(self, path, doc_id, source, **meta):
        doc = super().parse(path, doc_id, source, **meta)
        doc.images = [ImageData(image_id="img1", data=b"png-bytes",
                                source="a.pdf:1")]
        return doc


class _FakeEmptyTextParser(_FakeParser):
    def parse(self, path, doc_id, source, **meta):
        doc = super().parse(path, doc_id, source, **meta)
        doc.text = ""
        return doc


class _FakeNoContentParser(_FakeParser):
    """既无正文也无表格：全链路应无任何 chunk 可嵌入。"""

    def parse(self, path, doc_id, source, **meta):
        doc = super().parse(path, doc_id, source, **meta)
        doc.text = ""
        doc.tables = []
        return doc


class _FakeChunker:
    def chunk(self, doc):
        from ragkb.models import Chunk
        return [Chunk(chunk_id="c1", doc_id=doc.doc_id, doc_type=doc.doc_type,
                      source=doc.source, text=doc.text, version="v1.0",
                      effective_date="2026-01-15")]


class _FakeEmptyChunker:
    def chunk(self, doc):
        if not doc.text.strip():
            return []
        return _FakeChunker().chunk(doc)


class _FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[0.1, 0.2] for _ in texts]


class _FakeIndexer:
    """带 upsert_table 的索引器：表格走索引器（生产 QdrantIndexer 无此方法）。"""

    def __init__(self):
        self.chunks = []
        self.tables = []

    def upsert(self, chunks, embeddings):
        self.chunks.extend(chunks)

    def upsert_table(self, table):
        self.tables.append(table)


class _FakeIndexerNoTables:
    """无 upsert_table：表格应落到 pg.upsert。"""

    def __init__(self):
        self.chunks = []

    def upsert(self, chunks, embeddings):
        self.chunks.extend(chunks)


class _FakePg:
    def __init__(self):
        self.tables = []
        self.versions = []

    def upsert(self, table):
        self.tables.append(table)

    def register_version(self, doc_id, version, effective_date, source):
        self.versions.append((doc_id, version, effective_date, source))


class _FakeMinio:
    def __init__(self):
        self.images = []

    def put(self, image_id, data):
        self.images.append((image_id, data))


class _FakeOcr:
    def __init__(self, text="识别出的文字"):
        self.text = text

    def extract_text(self, data):
        return self.text


class _RaisingOcr:
    def extract_text(self, data):
        raise OCRUnavailableError("OCR 引擎不可用")


class _ValueErrorOcr:
    """OCR 抛非 OCRUnavailableError 的异常（如 PIL 解码坏图字节）。"""

    def extract_text(self, data):
        raise ValueError("corrupt image bytes")


def _pipeline(**overrides):
    kwargs = dict(parser=_FakeParser(), chunker=_FakeChunker(),
                  embedder=_FakeEmbedder(), indexer=_FakeIndexer(),
                  ocr=_FakeOcr(), pg=_FakePg(), minio=_FakeMinio())
    kwargs.update(overrides)
    return IngestPipeline(**kwargs)


def test_ingest_runs_full_pipeline():
    pg = _FakePg()
    pipe = _pipeline(pg=pg)
    idx = pipe._indexer
    result = pipe.ingest("/tmp/fake.pdf", doc_id="d1", source="fake.pdf",
                         version="v1.0", effective_date="2026-01-15")
    assert len(idx.chunks) == 2  # 1 正文块 + 1 表格 A 路块
    table_chunks = [c for c in idx.chunks if c.table_id == "t1"]
    assert len(table_chunks) == 1
    assert "| 型号 | 单价 |" in table_chunks[0].text
    assert len(idx.tables) == 1
    assert pg.versions == [("d1", "v1.0", "2026-01-15", "fake.pdf")]
    assert result == {"doc_id": "d1", "chunks": 2, "tables": 1}


def test_ingest_tables_go_to_pg_when_indexer_lacks_upsert_table():
    pg = _FakePg()
    pipe = _pipeline(indexer=_FakeIndexerNoTables(), pg=pg)
    pipe.ingest("/tmp/fake.pdf", doc_id="d1", source="fake.pdf")
    assert len(pipe._indexer.chunks) == 2  # 1 正文 + 1 表格块（A 路照常向量入库）
    assert len(pg.tables) == 1
    assert pg.tables[0].table_id == "t1"


def test_ingest_ocr_appends_text_and_stores_originals():
    minio = _FakeMinio()
    pipe = _pipeline(parser=_FakeImageParser(),
                     ocr=_FakeOcr(text="合同号 HT-2026"), minio=minio)
    pipe.ingest("/tmp/fake.pdf", doc_id="d1", source="fake.pdf")
    assert pipe._indexer.chunks[0].text.endswith("合同号 HT-2026")
    assert minio.images == [("img1", b"png-bytes")]


def test_ingest_ocr_unavailable_keeps_text_but_stores_image():
    minio = _FakeMinio()
    pipe = _pipeline(parser=_FakeImageParser(), ocr=_RaisingOcr(), minio=minio)
    pipe.ingest("/tmp/fake.pdf", doc_id="d1", source="fake.pdf")
    assert pipe._indexer.chunks[0].text == "产品型号 A-100 单价 99 元。"
    assert minio.images == [("img1", b"png-bytes")]


def test_ingest_any_ocr_exception_keeps_pipeline_and_stores_image():
    """OCR 抛任意异常（如 ValueError）不阻断流水线，原图仍入库（spec §8）。"""
    minio = _FakeMinio()
    pipe = _pipeline(parser=_FakeImageParser(), ocr=_ValueErrorOcr(), minio=minio)
    pipe.ingest("/tmp/fake.pdf", doc_id="d1", source="fake.pdf")
    assert pipe._indexer.chunks[0].text == "产品型号 A-100 单价 99 元。"
    assert minio.images == [("img1", b"png-bytes")]


def test_ingest_applies_cleaning_before_chunking():
    class _DirtyParser(_FakeParser):
        def parse(self, path, doc_id, source, **meta):
            doc = super().parse(path, doc_id, source, **meta)
            doc.text = "产品型号 A-100 单价 99 元。\n第 1 页 / 共 3 页"
            return doc

    pipe = _pipeline(parser=_DirtyParser(), chunker=_FakeChunker(),
                     embedder=_FakeEmbedder(), indexer=_FakeIndexer(),
                     ocr=_FakeOcr(), pg=_FakePg(), minio=_FakeMinio())
    pipe.ingest("/tmp/fake.pdf", doc_id="d1", source="fake.pdf",
                version="v1.0", effective_date="2026-01-15")
    idx = pipe._indexer
    assert len(idx.chunks) == 2  # 1 正文 + 1 表格块
    assert "第 1 页" not in idx.chunks[0].text


def test_ingest_skips_version_registration_when_no_version():
    pg = _FakePg()
    pipe = _pipeline(pg=pg)
    pipe.ingest("/tmp/fake.pdf", doc_id="d1", source="fake.pdf")
    assert pg.versions == []


def test_ingest_skips_embedding_when_no_chunks():
    embedder = _FakeEmbedder()
    pipe = _pipeline(parser=_FakeNoContentParser(), chunker=_FakeEmptyChunker(),
                     embedder=embedder)
    idx = pipe._indexer
    pipe.ingest("/tmp/fake.pdf", doc_id="d1", source="fake.pdf",
                version="v1.0", effective_date="2026-01-15")
    assert idx.chunks == []
    assert embedder.calls == 0


def test_ingest_table_achunks_embedded_even_when_body_empty():
    """表格 A 路：正文为空（如 Excel 无文本）时，表格 Markdown 块仍嵌入入库。"""
    embedder = _FakeEmbedder()
    pipe = _pipeline(parser=_FakeEmptyTextParser(), chunker=_FakeEmptyChunker(),
                     embedder=embedder)
    idx = pipe._indexer
    pipe.ingest("/tmp/fake.pdf", doc_id="d1", source="fake.pdf",
                version="v1.0", effective_date="2026-01-15")
    assert len(idx.chunks) == 1
    assert idx.chunks[0].table_id == "t1"
    assert "| 型号 | 单价 |" in idx.chunks[0].text
    assert len(idx.tables) == 1  # 表格 B 路独立于正文继续入库
    assert embedder.calls == 1


def test_ingest_skip_embed_skips_vector_upsert_but_keeps_rest():
    """--skip-embed：不嵌入、不向量入库，表格/版本等其余链路照常（冒烟用）。"""
    embedder = _FakeEmbedder()
    pg = _FakePg()
    pipe = _pipeline(embedder=embedder, pg=pg)
    idx = pipe._indexer
    pipe.ingest("/tmp/fake.pdf", doc_id="d1", source="fake.pdf",
                version="v1.0", effective_date="2026-01-15",
                skip_embed=True)
    assert idx.chunks == []          # 向量入库被跳过
    assert embedder.calls == 0       # 嵌入器未被调用
    assert len(idx.tables) == 1      # 表格仍入库
    assert pg.versions == [("d1", "v1.0", "2026-01-15", "fake.pdf")]


def test_ingest_markdown_file_via_real_parser(tmp_path):
    """真实 Markdown 解析器进完整入库链路：正文块 + 表格 A/B 路。"""
    from ragkb.parsers.md_parser import MarkdownParser

    path = tmp_path / "doc.md"
    path.write_text("# 标题\n\n产品说明。\n\n| 型号 | 单价 |\n| --- | --- |\n| A-100 | 99 |\n",
                    encoding="utf-8")
    pg = _FakePg()
    pipe = _pipeline(parser=MarkdownParser(), pg=pg)
    idx = pipe._indexer
    result = pipe.ingest(str(path), doc_id="d1", source="doc.md", version="v1.0")
    assert len(idx.chunks) == 2  # 正文块 + 表格 A 路块
    assert len(idx.tables) == 1  # 表格 B 路
    assert result == {"doc_id": "d1", "chunks": 2, "tables": 1}


def test_cli_module_executes_via_python_dash_m():
    """Task 18 回归：`python -m ragkb.pipeline.ingest` 必须真正执行 main()
    （曾因缺少 __main__ 守卫而静默空转，冒烟测试发现）。"""
    env = dict(os.environ,
               PYTHONPATH=str(Path(__file__).parent.parent / "src"))
    proc = subprocess.run(
        [sys.executable, "-m", "ragkb.pipeline.ingest", "--help"],
        capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 0
    assert "--skip-embed" in proc.stdout  # main() 的 argparse 已生效
