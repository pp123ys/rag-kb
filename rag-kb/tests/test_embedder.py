import numpy as np
import pytest

from ragkb.embedder.embedder import Embedder


class _FakeEncoder:
    def encode(self, texts, **kw):
        return np.zeros((len(texts), 4), dtype=np.float32)


@pytest.mark.model
def test_real_embedder_dims(settings):
    e = Embedder(model_name=settings.embed_model)
    vec = e.embed(["测试句子"])[0]
    assert vec.shape[0] == 1024


def test_embedder_interface_with_fake():
    e = Embedder(model_name="fake", encoder=_FakeEncoder())
    vecs = e.embed(["a", "b"])
    assert vecs.shape == (2, 4)
