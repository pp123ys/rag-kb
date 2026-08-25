from ragkb.cleaners.cleaner import clean_text


def test_removes_page_footer_and_watermark():
    dirty = "正文第一段。\n第 3 页 / 共 10 页\n本文档仅供内部使用\n正文第二段。"
    out = clean_text(dirty)
    assert "第 3 页 / 共 10 页" not in out
    assert "本文档仅供内部使用" not in out
    assert "正文第一段。" in out


def test_collapses_blank_lines_and_normalizes_whitespace():
    dirty = "第一行。\n\n\n  第二行\t内容。\n"
    out = clean_text(dirty)
    assert "  第二行\t内容" not in out
    assert "\n\n\n" not in out


def test_removes_garbage_chars():
    dirty = "有效内容\x00\x1f\x9d 正常"
    out = clean_text(dirty)
    assert "\x00" not in out
    assert "正常" in out
