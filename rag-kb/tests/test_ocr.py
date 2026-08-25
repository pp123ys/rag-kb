import pytest

from ragkb.ocr.ocr_engine import OCRUnavailableError, OCRClient


class _FakeOCR:
    def __init__(self, result): self._r = result

    def ocr(self, image_bytes: bytes) -> str:
        if b"empty" in image_bytes:
            return ""
        return self._r


def test_ocr_client_returns_text():
    c = OCRClient(_FakeOCR("报价 99 元"))
    assert c.extract_text(b"\x89PNG") == "报价 99 元"


def test_ocr_client_empty_image_returns_empty_string():
    c = OCRClient(_FakeOCR(""))
    assert c.extract_text(b"empty") == ""


def test_ocr_client_unavailable_raises():
    c = OCRClient(None)
    with pytest.raises(OCRUnavailableError):
        c.extract_text(b"\x89PNG")
