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
