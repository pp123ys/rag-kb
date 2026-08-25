"""本地目录图片存储（嵌入式，免 Docker）。

接口与 MinioImageStore 一致（ensure_bucket/put/get），
由 get_image_store 工厂按配置选择：
- settings.minio_endpoint 非空 → MinioImageStore（远程 MinIO）
- 否则 → LocalImageStore（本地目录 settings.images_dir）
"""
from pathlib import Path


class LocalImageStore:
    """原图存储：图片二进制落盘本地目录，chunk 以 image_id 引用。"""

    def __init__(self, directory: str):
        self._dir = Path(directory)

    def ensure_bucket(self):
        self._dir.mkdir(parents=True, exist_ok=True)

    def put(self, image_id: str, data: bytes):
        self.ensure_bucket()
        # image_id 由解析器生成（doc_id-img-页码-xref），无路径穿越风险；
        # 仍做基本防御：仅取文件名部分
        safe_name = Path(image_id).name
        (self._dir / safe_name).write_bytes(data)

    def get(self, image_id: str) -> bytes:
        safe_name = Path(image_id).name
        return (self._dir / safe_name).read_bytes()
