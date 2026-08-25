import argparse
import logging
import os
import uuid

from ragkb.chunker import Chunker
from ragkb.config import get_settings
from ragkb.embedder import Embedder
from ragkb.indexers import (QdrantIndexer, get_image_store,
                            get_table_indexer)
from ragkb.models import Chunk
from ragkb.ocr import OCRClient
from ragkb.parsers import parse_document
from ragkb.parsers.base import table_to_markdown

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
        self._embedder = embedder or Embedder(
            settings.embed_model, cache_dir=settings.model_cache_dir)
        self._indexer = indexer or QdrantIndexer(settings)
        self._ocr = ocr if ocr is not None else OCRClient(settings.ocr_lang)
        # 存储后端工厂：pg_dsn 非空用 PG，否则 SQLite；minio_endpoint 非空用 MinIO，否则本地目录
        self._pg = pg if pg is not None else get_table_indexer(settings)
        self._minio = minio if minio is not None else get_image_store(settings)

    def ingest(self, path: str, doc_id: str | None = None, source: str | None = None,
               department: str = "", version: str = "", effective_date: str = "",
               parse_images: bool = True, skip_embed: bool = False):
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
        # 表格 A 路（spec §4.4）：每个表格独立成块，text 为 Markdown 序列化，
        # 携带 table_id 打通 A↔B 联动（chunk.table_id → retrieve_table），
        # 与正文块一起嵌入入库；正文为空（如 Excel）时表格块仍可语义检索。
        # chunk_id 确定性（doc-version-t索引）保证幂等重入，不产生重复块。
        for i, table in enumerate(parsed.tables):
            chunks.append(Chunk(
                chunk_id=f"{parsed.doc_id}-{parsed.version}-t{i}",
                doc_id=parsed.doc_id, doc_type=parsed.doc_type,
                source=table.source, text=table_to_markdown(table),
                department=parsed.department, version=parsed.version,
                effective_date=parsed.effective_date,
                table_id=table.table_id,
            ))
        if chunks:
            if skip_embed:
                # 无模型缓存的机器（冒烟/CI）跳过嵌入与向量入库，其余链路照常
                logger.warning(
                    "skip_embed：跳过嵌入与向量入库（%d 个 chunk 未写入 Qdrant）",
                    len(chunks))
            else:
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
    ap.add_argument("--skip-embed", action="store_true",
                    help="跳过嵌入与向量入库（无模型下载环境的冒烟验证）")
    args = ap.parse_args()
    pipe = IngestPipeline(ocr=OCRClient())
    result = pipe.ingest(args.path, doc_id=args.doc_id, source=args.source,
                         department=args.department, version=args.version,
                         effective_date=args.effective_date,
                         skip_embed=args.skip_embed)
    print(result)


if __name__ == "__main__":
    # 缺此守卫时 `python -m ragkb.pipeline.ingest` 会静默空转（Task 18 冒烟发现）
    main()
