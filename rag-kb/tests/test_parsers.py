import fitz
import pytest

from ragkb.parsers.pdf_parser import PdfParser


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # 注意：insert_text 默认用 Helvetica，无法编码中文（提取出来是乱码），
    # 必须显式指定内置中文字体 china-s（简体）。
    page.insert_text((72, 72), "产品规格说明。", fontname="china-s")
    page.insert_text((72, 120), "型号 A-100 单价 99 元。", fontname="china-s")
    rect = fitz.Rect(72, 160, 200, 200)
    # 1x1 RGBA PNG（有效字节流；计划原文的 hex 截断了 IEND CRC，
    # 会被 PyMuPDF 以 "premature end of data in png image" 拒绝）
    page.insert_image(rect, stream=bytes.fromhex("89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4890000000B49444154789C6360000200000500017A5EAB3F0000000049454E44AE426082"))
    doc.save(path)
    doc.close()
    return path


def test_pdf_parser_extracts_text_and_image(sample_pdf):
    result = PdfParser().parse(str(sample_pdf), doc_id="d1", source="sample.pdf")
    assert "产品规格说明" in result.text
    assert len(result.images) >= 1
    assert result.images[0].data[:8] == b"\x89PNG\r\n\x1a\n"
