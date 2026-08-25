from ragkb.models import Chunk, ImageData, ParsedDocument, TableData


def test_parsed_document_holds_channels():
    doc = ParsedDocument(
        doc_id="d1", doc_type="pdf", source="a.pdf", text="正文",
        tables=[TableData(table_id="t1", name="报价表", headers=["型号", "单价"],
                          rows=[["A-100", "99"]], source="a.pdf:3")],
        images=[ImageData(image_id="i1", data=b"\x89PNG", source="a.pdf:2")],
    )
    assert doc.tables[0].headers == ["型号", "单价"]
    assert doc.images[0].image_id == "i1"


def test_chunk_carries_metadata():
    c = Chunk(chunk_id="c1", doc_id="d1", doc_type="pdf", department="销售部",
              version="v2.3", effective_date="2026-01-15", source="a.pdf:3",
              text="……", table_id="t1")
    assert c.version == "v2.3"
    assert c.effective_date == "2026-01-15"
    assert c.table_id == "t1"
