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


def test_ocr_client_unavailable_raises(monkeypatch):
    import sys
    # 强制 paddleocr 导入失败，使测试与本地是否安装 paddleocr 无关
    monkeypatch.setitem(sys.modules, "paddleocr", None)
    c = OCRClient(None)
    with pytest.raises(OCRUnavailableError):
        c.extract_text(b"\x89PNG")


def test_parse_ocr_result_joins_lines():
    from ragkb.ocr.ocr_engine import _parse_ocr_result
    result = [[[[0, 0, 1, 1], ("hello", 0.99)], [[0, 0, 1, 1], ("world", 0.98)]]]
    assert _parse_ocr_result(result) == "hello\nworld"


def test_parse_ocr_result_empty_variants():
    from ragkb.ocr.ocr_engine import _parse_ocr_result
    assert _parse_ocr_result(None) == ""
    assert _parse_ocr_result([]) == ""
    assert _parse_ocr_result([[]]) == ""


def test_parse_ocr_result_skips_malformed_lines():
    from ragkb.ocr.ocr_engine import _parse_ocr_result
    # len-1 行不崩、非元组行跳过、空文本行跳过
    result = [[[1], [0, 0, 1, 1], [[0, 0, 1, 1], ("", 0.5)]]]
    assert _parse_ocr_result(result) == ""
