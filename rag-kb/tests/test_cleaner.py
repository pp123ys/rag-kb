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
    assert "\x9d" not in out
    assert "正常" in out


def test_watermark_only_matches_whole_line():
    # 水印必须整行匹配：含「机密」子串的内容行保留，独立水印行删除
    dirty = "本协议属于机密文件，不得外传。\n机密\nConfidential"
    out = clean_text(dirty)
    assert "本协议属于机密文件" in out
    assert "机密\n" not in out  # 独立水印行已删
    assert "Confidential" not in out
    assert out == "本协议属于机密文件，不得外传。"


def test_standalone_number_line_removed_as_page_number():
    # 独立数字行视为页码（v1 取舍：2024 这类内容行也会被删）
    out = clean_text("说明文字。\n2024\n继续说明。")
    assert "2024" not in out
    assert "说明文字。" in out
