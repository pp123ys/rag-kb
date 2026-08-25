import pytest

from ragkb.indexers.minio_store import MinioImageStore


@pytest.mark.integration
def test_put_and_get_image(settings):
    store = MinioImageStore(settings)
    store.ensure_bucket()
    store.put("i1", b"\x89PNG data")
    assert store.get("i1") == b"\x89PNG data"
