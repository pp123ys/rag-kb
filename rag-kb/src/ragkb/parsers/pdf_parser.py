# 用 pymupdf 官方入口名而非 fitz：PyMuPDF>=1.24 的 fitz 是别名模块，
# import fitz 会 print 弃用警告到 stdout —— 在 MCP stdio 传输下污染 JSON-RPC 流
import pymupdf as fitz  # noqa: N813  与 fitz API 完全一致
import pdfplumber

from ragkb.models import ImageData, ParsedDocument, TableData
from ragkb.parsers.base import DocumentParser, cell_to_str


class PdfParser(DocumentParser):
    """PDF：正文 / 表格 / 图片三路分离。"""

    def parse(self, path, doc_id, source, department="", version="",
              effective_date=""):
        text_parts, tables, images = [], [], []
        with pdfplumber.open(path) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text:
                    text_parts.append(page_text)
                for tbl_idx, tbl in enumerate(page.extract_tables() or [], start=1):
                    if not tbl:
                        continue
                    headers = [cell_to_str(c) for c in tbl[0]]
                    rows = [[cell_to_str(c) for c in row] for row in tbl[1:]]
                    tables.append(self._make_table(headers, rows, source, page_no, tbl_idx))

        with fitz.open(path) as doc:
            for page_no, page in enumerate(doc, start=1):
                for img in page.get_images(full=True):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha > 3:  # CMYK 转 RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    images.append(ImageData(
                        image_id=f"{doc_id}-img-{page_no}-{xref}",
                        data=pix.tobytes("png"),
                        source=f"{source}:{page_no}",
                    ))

        return ParsedDocument(
            doc_id=doc_id, doc_type="pdf", source=source,
            text="\n\n".join(text_parts), tables=tables, images=images,
            department=department, version=version, effective_date=effective_date,
        )

    @staticmethod
    def _make_table(headers, rows, source, page_no, tbl_idx=1):
        table_id = f"{source}-{page_no}-{tbl_idx}-{len(headers)}x{len(rows)}"
        return TableData(table_id=table_id, name=f"第{page_no}页第{tbl_idx}张表格",
                         headers=headers, rows=rows, source=f"{source}:{page_no}")
