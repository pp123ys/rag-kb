"""Markdown 解析器：正文 + GFM 表格（| 管道表格）双通道。

- 正文：剔除表格行后的原文（保留 Markdown 标记，语义检索不受影响）；
- 表格：GFM 管道表格（表头行 + |---| 分隔行 + 数据行）提取为 TableData，
  走表格 A 路（语义检索块）与 B 路（retrieve_table 精确取数）。
"""
from ragkb.models import ParsedDocument, TableData
from ragkb.parsers.base import DocumentParser, cell_to_str


def _read_text(path: str) -> str:
    """UTF-8 优先读取，容错 BOM 与 GBK（Windows 常见编码）。"""
    data = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _is_separator_row(line: str) -> bool:
    """GFM 表格分隔行：仅由 | - : 空格 组成且至少含一个 -。"""
    s = line.strip().strip("|")
    return bool(s) and set(s) <= set("|:- ") and "-" in s


def _split_cells(line: str) -> list[str]:
    """拆管道表格行：去掉首尾 |，按 | 拆分并 trim。"""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _extract_tables(lines: list[str], source: str,
                    doc_id: str = "") -> tuple[list[TableData], list[str]]:
    """扫描 GFM 表格块；返回 (表格列表, 剔除表格行后的正文行)。"""
    tables: list[TableData] = []
    body: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 表头行 + 紧跟分隔行 → 表格块起点
        if "|" in line and i + 1 < len(lines) and _is_separator_row(lines[i + 1]):
            headers = _split_cells(line)
            i += 2  # 跳过表头与分隔行
            rows: list[list[str]] = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(_split_cells(lines[i]))
                i += 1
            # 行宽与表头对齐（与 Excel/CSV 一致，保证 B 路入库行列数一致）
            n = len(headers)
            rows = [r[:n] + ["N/A"] * max(0, n - len(r)) for r in rows]
            tables.append(TableData(
                table_id=f"{source}-{len(tables)}",
                name=f"表格{len(tables) + 1}",
                headers=[cell_to_str(h) for h in headers],
                rows=[[cell_to_str(c) for c in r] for r in rows],
                source=f"{source}:md",
                doc_id=doc_id,
            ))
            continue
        body.append(line)
        i += 1
    return tables, body


class MarkdownParser(DocumentParser):
    """Markdown：正文文本 + GFM 表格 → TableData。"""

    def parse(self, path, doc_id, source, department="", version="",
              effective_date=""):
        text = _read_text(path)
        lines = text.splitlines()
        tables, body = _extract_tables(lines, source, doc_id)
        return ParsedDocument(
            doc_id=doc_id, doc_type="markdown", source=source,
            text="\n".join(body).strip(),
            tables=tables, images=[],
            department=department, version=version,
            effective_date=effective_date,
        )
