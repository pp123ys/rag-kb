from ragkb.models import ParsedDocument, TableData


def cell_to_str(value) -> str:
    """单元格值归一：None/空→N/A，其余 strip 后转 str。"""
    if value is None:
        return "N/A"
    s = str(value).strip()
    return s if s else "N/A"


def table_to_markdown(table: TableData) -> str:
    """表格 → Markdown 文本（A 路：参与语义检索的表格块正文）。"""
    lines = [f"## 表格：{table.name}", f"来源：{table.source}"]
    header = "| " + " | ".join(table.headers) + " |"
    sep = "| " + " | ".join(["---"] * len(table.headers)) + " |"
    lines.append(header)
    lines.append(sep)
    for row in table.rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


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
    if ext == "xlsx":
        # 仅 xlsx（OpenXML）；旧版 .xls 由 openpyxl 无法解析，
        # 不列入分发，落入下方不支持的扩展名分支抛 ValueError。
        from ragkb.parsers.excel_parser import ExcelParser
        return ExcelParser().parse(path, doc_id=doc_id, source=source, **meta)
    if ext in ("eml", "msg"):
        from ragkb.parsers.email_parser import EmailParser
        return EmailParser().parse(path, doc_id=doc_id, source=source, **meta)
    if ext == "md":
        from ragkb.parsers.md_parser import MarkdownParser
        return MarkdownParser().parse(path, doc_id=doc_id, source=source, **meta)
    if ext == "csv":
        from ragkb.parsers.csv_parser import CsvParser
        return CsvParser().parse(path, doc_id=doc_id, source=source, **meta)
    raise ValueError(f"不支持的文档类型: {ext}")
