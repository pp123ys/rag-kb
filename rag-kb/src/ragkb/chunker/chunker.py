import re
import uuid

from ragkb.models import Chunk, ParsedDocument

_SENTENCE_END = re.compile(r"(?<=[。！？!?；;])\s*")


class Chunker:
    """语义边界切块：以句子为最小单位，禁止断句，支持 overlap。"""

    def __init__(self, chunk_target_chars: int = 400,
                 chunk_overlap_chars: int = 60, chunk_max_chars: int = 800):
        self.target = chunk_target_chars
        self.overlap = chunk_overlap_chars
        self.max_chars = chunk_max_chars

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        sentences = self._split_sentences(doc.text)
        if not sentences:
            return []
        return self._pack(doc, sentences)

    def _split_sentences(self, text: str) -> list[str]:
        parts = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
        # 超长句（如整段无标点）按长度硬切，保留逗号边界
        out = []
        for p in parts:
            if len(p) <= self.max_chars:
                out.append(p)
            else:
                out.extend(self._hard_split(p))
        return out

    def _hard_split(self, sentence: str) -> list[str]:
        # 仅在逗号/空格处切，避免切词
        segs = re.split(r"(?<=[,，])\s*", sentence)
        buf, out = "", []
        for seg in segs:
            if len(buf) + len(seg) > self.max_chars and buf:
                out.append(buf)
                buf = seg
            else:
                buf += seg
        if buf:
            out.append(buf)
        return out

    def _pack(self, doc: ParsedDocument, sentences: list[str]) -> list[Chunk]:
        chunks, buf = [], ""
        for sent in sentences:
            if buf and len(buf) + len(sent) > self.target:
                chunks.append(self._make_chunk(doc, buf))
                buf = self._overlap_tail(buf)
            buf += sent
        if buf:
            chunks.append(self._make_chunk(doc, buf))
        return chunks

    def _overlap_tail(self, buf: str) -> str:
        """返回 overlap 尾部：上块末尾若干字符，保证上下文连续。

        尾部起点若落在句子中间，则前移到最近的上一个句子结束符之后，
        使下一块从完整句子开始（不切断上一句）。
        """
        if self.overlap <= 0 or not buf:
            return ""
        tail = buf[-self.overlap:]
        start = len(buf) - self.overlap
        prev = list(_SENTENCE_END.finditer(buf, 0, start))
        if prev:
            tail = buf[prev[-1].end():]
        return tail

    def _make_chunk(self, doc: ParsedDocument, text: str) -> Chunk:
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc.doc_id, doc_type=doc.doc_type, source=doc.source,
            text=text, department=doc.department, version=doc.version,
            effective_date=doc.effective_date,
        )
