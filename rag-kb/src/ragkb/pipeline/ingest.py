import argparse
import logging
import os
import uuid

from ragkb.chunker import Chunker
from ragkb.config import get_settings
from ragkb.embedder import Embedder
from ragkb.indexers import QdrantIndexer
from ragkb.indexers.minio_store import MinioImageStore
from ragkb.indexers.pg_table_indexer import PgTableIndexer
from ragkb.ocr import OCRClient
from ragkb.parsers import parse_document

logger = logging.getLogger(__name__)


class IngestPipeline:
    """入库编排：解析 → 清洗 → 切块 → 嵌入 → Qdrant/表格/图片入库。"""

    def __init__(self, parser=None, chunker=None, embedder=None,
                 indexer=None, ocr=None, pg=None, minio=None, settings=None):
        settings = settings or get_settings()
        self._settings = settings
        self._parser = parser or parse_document
        self._chunker = chunker or Chunker(
            settings.chunk_target_chars, settings.chunk_overlap_chars,
            settings.chunk_max_chars)
        self._embedder = embedder or Embedder(settings.embed_model)
        self._indexer = indexer or QdrantIndexer(settings)
        self._ocr = ocr if ocr is not None else OCRClient(settings.ocr_lang)
        self._pg = pg or PgTableIndexer(settings.pg_dsn)
        self._minio = minio or MinioImageStore(settings)

    def ingest(self, path: str, doc_id: str | None = None, source: str | None = None,
               department: str = "", version: str = "", effective_date: str = "",
               parse_images: bool = True):
        doc_id = doc_id or str(uuid.uuid4())
        source = source or os.path.basename(path)
        logger.info("ingest %s (%s)", source, doc_id)

        # 兼容两种解析协议：模块级 parse_document 函数，或解析器对象 .parse()
        parse = getattr(self._parser, "parse", None) or self._parser
        parsed = parse(path, doc_id=doc_id, source=source,
                       department=department, version=version,
                       effective_date=effective_date)

        # 图片 OCR → 追加为正文（文字进检索，原图进 MinIO）
        if parse_images and parsed.images and self._ocr is not None:
            ocr_texts = []
            for img in parsed.images:
                try:
                    text = self._ocr.extract_text(img.data)
                except Exception:  # 任何 OCR 失败都不阻断流水线（spec §8）
                    logger.warning("OCR 失败，跳过图片 %s", img.image_id)
                    text = ""
                if text:
                    ocr_texts.append(text)
                self._minio.put(img.image_id, img.data)
            if ocr_texts:
                parsed.text = (parsed.text + "\n\n" + "\n".join(ocr_texts)).strip()

        # 清洗：去页眉页脚/乱码/归一空白（spec §4.2）
        from ragkb.cleaners import clean_text
        parsed.text = clean_text(parsed.text)

        chunks = self._chunker.chunk(parsed)
        if chunks:
            vectors = self._embedder.embed([c.text for c in chunks])
            self._indexer.upsert(chunks, embeddings=vectors)

        for table in parsed.tables:
            self._indexer.upsert_table(table) if hasattr(self._indexer, "upsert_table") \
                else self._pg.upsert(table)

        # 版本登记（供 list_versions / 版本过滤）
        if version:
            self._pg.register_version(doc_id, version, effective_date, source)

        logger.info("done: %d chunks, %d tables", len(chunks), len(parsed.tables))
        return {"doc_id": doc_id, "chunks": len(chunks),
                "tables": len(parsed.tables)}


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="RAG 知识库入库")
    ap.add_argument("path")
    ap.add_argument("--doc-id")
    ap.add_argument("--source")
    ap.add_argument("--department", default="")
    ap.add_argument("--version", default="")
    ap.add_argument("--effective-date", default="")
    args = ap.parse_args()
    pipe = IngestPipeline(ocr=OCRClient())
    result = pipe.ingest(args.path, doc_id=args.doc_id, source=args.source,
                         department=args.department, version=args.version,
                         effective_date=args.effective_date)
    print(result)
