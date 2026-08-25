from dataclasses import dataclass, field


@dataclass
class TableData:
    """解析出的结构化表格（B 路数据源）。"""
    table_id: str
    name: str
    headers: list[str]
    rows: list[list[str]]
    source: str  # 文件名:页码 或 文件名:Sheet名


@dataclass
class ImageData:
    """解析出的图片。"""
    image_id: str
    data: bytes
    source: str  # 文件名:页码


@dataclass
class ParsedDocument:
    """解析层输出：正文 / 表格 / 图片三通道。"""
    doc_id: str
    doc_type: str  # pdf | excel | email
    source: str    # 原始文件名
    text: str
    tables: list[TableData] = field(default_factory=list)
    images: list[ImageData] = field(default_factory=list)
    department: str = ""
    version: str = ""
    effective_date: str = ""


@dataclass
class Chunk:
    """切块产物：文本 + 全量元数据标签。"""
    chunk_id: str
    doc_id: str
    doc_type: str
    source: str
    text: str
    department: str = ""
    version: str = ""
    effective_date: str = ""
    table_id: str | None = None

    def metadata(self) -> dict:
        """Qdrant payload 用。"""
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "source": self.source,
            "department": self.department,
            "version": self.version,
            "effective_date": self.effective_date,
            "table_id": self.table_id,
        }
