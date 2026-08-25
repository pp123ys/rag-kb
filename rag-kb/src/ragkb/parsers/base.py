from ragkb.models import ParsedDocument


class DocumentParser:
    """解析器协议：入参为文件路径与文档元数据，输出 ParsedDocument。"""

    def parse(self, path: str, doc_id: str, source: str,
              department: str = "", version: str = "",
              effective_date: str = "") -> ParsedDocument:
        raise NotImplementedError


def parse_document(path: str, doc_id: str, source: str, **meta) -> ParsedDocument:
    """按扩展名分发到具体解析器。"""
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        from ragkb.parsers.pdf_parser import PdfParser
        return PdfParser().parse(path, doc_id=doc_id, source=source, **meta)
    if ext in ("xlsx", "xls"):
        from ragkb.parsers.excel_parser import ExcelParser
        return ExcelParser().parse(path, doc_id=doc_id, source=source, **meta)
    if ext in ("eml", "msg"):
        from ragkb.parsers.email_parser import EmailParser
        return EmailParser().parse(path, doc_id=doc_id, source=source, **meta)
    raise ValueError(f"不支持的文档类型: {ext}")
