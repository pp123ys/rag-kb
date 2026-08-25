from io import BytesIO

from minio import Minio


class MinioImageStore:
    """原图存储：图片二进制入 MinIO，chunk 以 image_id 引用。"""

    def __init__(self, settings):
        self._client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=False,
        )
        self._bucket = settings.minio_bucket

    def ensure_bucket(self):
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def put(self, image_id: str, data: bytes):
        self.ensure_bucket()
        # minio put_object 需要可读流（bytes 无 .read()），用 BytesIO 包装
        self._client.put_object(self._bucket, image_id, BytesIO(data), len(data),
                                content_type="image/png")

    def get(self, image_id: str) -> bytes:
        resp = self._client.get_object(self._bucket, image_id)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()
