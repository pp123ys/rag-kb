# tests/test_chunker.py
from ragkb.chunker.chunker import Chunker
from ragkb.models import ParsedDocument


def _doc(text, **kw):
    return ParsedDocument(doc_id="d1", doc_type="pdf", source="a.pdf",
                          text=text, department="销售部", version="v1.0",
                          effective_date="2026-01-15", **kw)


def test_chunks_respect_sentence_boundary():
    text = ("第一句话是介绍。第二句话讲型号。第三句话讲价格。"
            "第四句话讲售后。第五句话讲保修。第六句话讲物流。")
    chunks = Chunker(chunk_target_chars=18, chunk_overlap_chars=0).chunk(_doc(text))
    assert len(chunks) >= 2
    for c in chunks:
        assert c.text.endswith("。")  # 不从句子中间断开


def test_overlap_keeps_previous_context():
    text = ("第一句话是介绍。第二句话讲型号。第三句话讲价格。"
            "第四句话讲售后。第五句话讲保修。第六句话讲物流。")
    chunks = Chunker(chunk_target_chars=18, chunk_overlap_chars=8).chunk(_doc(text))
    if len(chunks) >= 2:
        # 后块开头应包含前块结尾的句子（overlap）
        assert chunks[0].text.split("。")[-2] in chunks[1].text


def test_chunk_carries_metadata():
    chunks = Chunker().chunk(_doc("只有一个句子，长度不长。"))
    c = chunks[0]
    assert c.doc_id == "d1" and c.department == "销售部"
    assert c.version == "v1.0" and c.effective_date == "2026-01-15"
    assert c.metadata()["doc_type"] == "pdf"


# ---------- Fix 1: 英文句号切句 ----------

def test_english_sentence_splitting():
    text = "Hello world. This is a test. It works."
    chunks = Chunker(chunk_target_chars=15, chunk_overlap_chars=0).chunk(_doc(text))
    assert len(chunks) >= 2
    assert chunks[0].text == "Hello world."
    for c in chunks:
        assert len(c.text) <= 800  # 不超过 max_chars
    # 无字符丢失/粘连（不得出现 "wordworld"/"wordword" 类拼接）
    assert "".join(c.text for c in chunks) == text
    assert "wordworld" not in "".join(c.text for c in chunks)


def test_english_abbreviation_periods_not_split():
    # "Mr." / "3.14" 的句号不是句子结束：仅 "." 后跟空白才算，且整句保留
    text = "Mr. Smith is here. It is 3.14 pm."
    chunks = Chunker(chunk_target_chars=100).chunk(_doc(text))
    assert len(chunks) == 1
    assert chunks[0].text == text  # 句号不切词、空格不丢失


def test_abbreviation_not_split_when_fits_target():
    # 即使 target 较紧，缩写后的整句也合为一块，不会出现孤立的 "Mr."
    text = "Mr. Smith is here. He left."
    chunks = Chunker(chunk_target_chars=20, chunk_overlap_chars=0).chunk(_doc(text))
    assert chunks[0].text == "Mr. Smith is here."
    assert all(c.text.strip() != "Mr." for c in chunks)


# ---------- Fix 2: 超长句按逗号/空格硬切 ----------

def test_hard_split_overlong_cjk_no_punctuation():
    # 1000 个汉字、无任何标点/空白：只能按 max_chars 兜底硬切
    text = "汉" * 1000
    chunks = Chunker(chunk_target_chars=100, chunk_overlap_chars=0,
                     chunk_max_chars=800).chunk(_doc(text))
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 800  # 每段 ≤ max_chars
    assert "".join(c.text for c in chunks) == text  # 无字符丢失


def test_hard_split_cuts_at_comma_boundary():
    # 1100 字中文、以逗号分句：硬切必须落在逗号后，不切词
    text = "采购流程分为六个阶段，" * 100  # 1100 字符
    chunks = Chunker(chunk_target_chars=100, chunk_overlap_chars=0,
                     chunk_max_chars=800).chunk(_doc(text))
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 800
    assert all(c.text.endswith("，") for c in chunks[:-1])  # 非末块都以逗号收尾
    assert "".join(c.text for c in chunks) == text


def test_hard_split_english_space_separated():
    text = ("word " * 199) + "word"  # 999 字符，空格分隔
    assert len(text) == 999
    chunks = Chunker(chunk_target_chars=100, chunk_overlap_chars=0,
                     chunk_max_chars=800).chunk(_doc(text))
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 800
    assert "".join(c.text for c in chunks) == text


# ---------- Fix 4: 三通道契约与基本性质 ----------

def test_empty_text_yields_no_chunks():
    assert Chunker().chunk(_doc("")) == []
    assert Chunker().chunk(_doc("   \n\t ")) == []


def test_chunk_ids_are_unique():
    text = "第一句话。第二句话。第三句话。第四句话。第五句话。" * 5
    chunks = Chunker(chunk_target_chars=12).chunk(_doc(text))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) > 1
    assert len(ids) == len(set(ids))


def test_text_chunks_have_no_table_or_image():
    chunks = Chunker().chunk(_doc("只有文字内容，不含表格与图片。"))
    for c in chunks:
        assert c.table_id is None
        assert c.image_id is None
