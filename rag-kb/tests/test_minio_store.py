import pytest

from ragkb.indexers.local_image_store import LocalImageStore
from ragkb.indexers.minio_store import MinioImageStore


def test_local_put_and_get_image(tmp_path):
    # 嵌入式本地目录（免 Docker）
    store = LocalImageStore(str(tmp_path / "images"))
    store.ensure_bucket()
    store.put("i1", b"\x89PNG data")
    assert store.get("i1") == b"\x89PNG data"


def test_local_image_id_sanitized(tmp_path):
    # image_id 含路径分隔符时只取文件名，避免路径穿越
    store = LocalImageStore(str(tmp_path / "images"))
    store.put("../evil", b"x")
    assert store.get("../evil") == b"x"
    assert not (tmp_path / "evil").exists()


@pytest.mark.integration
def test_minio_put_and_get_image(settings):
    store = MinioImageStore(settings)
    store.ensure_bucket()
    store.put("i1", b"\x89PNG data")
    assert store.get("i1") == b"\x89PNG data"
