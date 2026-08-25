"""CSV 解析器：整份文件作为一张表（第一行表头 + 数据行）。

- 编码：UTF-8（自动去 BOM），容错 GBK；
- 分隔符：csv.Sniffer 探测（逗号/分号/制表符），失败回退逗号；
- 行宽归一：短行补 "N/A"、长行截断，保证 B 路入库行列数一致。
"""
import csv

from ragkb.models import ParsedDocument, TableData
from ragkb.parsers.base import DocumentParser, cell_to_str


def _read_text(path: str) -> str:
    data = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _sniff_dialect(sample: str):
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return None


class CsvParser(DocumentParser):
    """CSV：整文件一张表，第一行为表头，其余为数据行。"""

    def parse(self, path, doc_id, source, department="", version="",
              effective_date="", has_headers=True):
        text = _read_text(path)
        dialect = _sniff_dialect(text[:4096])
        rows = list(csv.reader(text.splitlines(),
                               dialect=dialect or csv.excel))
        # 过滤全空行（纯空串/空白）
        rows = [r for r in rows if any(str(c).strip() for c in r)]
        if not rows:
            tables: list[TableData] = []
        else:
            headers = [cell_to_str(c) for c in rows[0]] if has_headers else []
            data_rows = rows[1:] if has_headers else rows
            # 行宽与表头对齐（短补 N/A、长截断），避免 query 阶段静默丢列
            n = len(headers)
            if n:
                data_rows = [r[:n] + ["N/A"] * max(0, n - len(r))
                             for r in data_rows]
            tables = [TableData(
                table_id=f"{source}-0",
                name="CSV 数据表",
                headers=headers,
                rows=[[cell_to_str(c) for c in r] for r in data_rows],
                source=f"{source}:csv",
                doc_id=doc_id,
            )]
        return ParsedDocument(
            doc_id=doc_id, doc_type="csv", source=source, text="",
            tables=tables, images=[],
            department=department, version=version,
            effective_date=effective_date,
        )
