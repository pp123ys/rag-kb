from openpyxl import load_workbook

from ragkb.models import ParsedDocument, TableData
from ragkb.parsers.base import DocumentParser


class ExcelParser(DocumentParser):
    """Excel：每个 sheet 独立成表，保留表头与行列结构。"""

    def parse(self, path, doc_id, source, department="", version="",
              effective_date=""):
        wb = load_workbook(path, data_only=True)
        tables = []
        for ws in wb.worksheets:
            rows = [[str(c.value).strip() if c.value is not None else "N/A"
                     for c in row] for row in ws.iter_rows()]
            rows = [r for r in rows if any(v != "N/A" for v in r)]
            if not rows:
                continue
            tables.append(TableData(
                table_id=f"{source}-{ws.title}",
                name=ws.title,
                headers=rows[0],
                rows=rows[1:],
                source=f"{source}:{ws.title}",
            ))
        return ParsedDocument(
            doc_id=doc_id, doc_type="excel", source=source, text="",
            tables=tables, images=[],
            department=department, version=version, effective_date=effective_date,
        )
