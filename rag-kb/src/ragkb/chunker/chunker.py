import uuid

from ragkb.models import Chunk, ParsedDocument

# 句子结束符：CJK/ASCII 句读（。！？!?；;），以及后跟空白的英文句号 "."
_TERMINATORS = "。！？!?；;"


class Chunker:
    """语义边界切块：以句子为最小单位，禁止断句，支持 overlap。

    chunk_max_chars 是软上限而非硬上限：句子本身超过 max_chars 时会在
    逗号/空格处硬切（窗口内无边界时按 max_chars 兜底），硬切产物每段
    ≤ max_chars；overlap 开启时，块的尾段会携带上一块的 overlap 尾巴，
    因此末块可能达到 max_chars + overlap。target 只决定块大小的偏好。
    """

    def __init__(self, chunk_target_chars: int = 400,
                 chunk_overlap_chars: int = 60, chunk_max_chars: int = 800):
        self.target = chunk_target_chars
        self.overlap = chunk_overlap_chars
        self.max_chars = chunk_max_chars

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        if not doc.text.strip():
            return []
        sentences = self._split_sentences(doc.text)
        if not sentences:
            return []
        return self._pack(doc, sentences)

    def _sentence_ends(self, text: str, up_to: int | None = None) -> list[int]:
        """扫描句子结束位置（排他下标），逐字符切分，原文逐字节保留。

        - 句读字符 。！？!?；; 之后即为句子结束；
        - 英文句号 "." 仅在其后紧跟空白时才算结束（避免 3.14 / e.g. 被误切）。
        不做 strip、不吞空白：每个字符都恰好归属某个句子，重新拼接即原文。
        """
        ends = []
        n = len(text)
        for i in range(1, n + 1):
            if up_to is not None and i > up_to:
                break
            ch = text[i - 1]
            if ch in _TERMINATORS:
                ends.append(i)
            elif ch == "." and i < n and text[i].isspace():
                ends.append(i)
        return ends

    def _split_sentences(self, text: str) -> list[str]:
        sentences = []
        start = 0
        for end in self._sentence_ends(text):
            sentences.append(text[start:end])
            start = end
        if start < len(text):
            sentences.append(text[start:])
        # 超长句（如整段无标点）按长度硬切，保留逗号/空格边界
        out = []
        for s in sentences:
            if len(s) <= self.max_chars:
                out.append(s)
            else:
                out.extend(self._hard_split(s))
        return out

    def _hard_split(self, sentence: str) -> list[str]:
        """超长句按长度硬切（budget = max_chars）。

        窗口内取最后一个逗号/空格处切断（切在边界之后，不切词）；
        窗口内无逗号/空格时，在 max_chars 处兜底硬切。每段 ≤ max_chars。
        """
        out = []
        start = 0
        n = len(sentence)
        while n - start > self.max_chars:
            window = start + self.max_chars
            cut = -1
            for i in range(window - 1, start, -1):
                if sentence[i].isspace() or sentence[i] in ",，":
                    cut = i + 1
                    break
            if cut < 0:
                cut = window  # 窗口内无逗号/空格：按 max_chars 兜底
            out.append(sentence[start:cut])
            start = cut
        out.append(sentence[start:])
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
        prev = self._sentence_ends(buf, up_to=start)
        if prev:
            tail = buf[prev[-1]:]
        return tail

    def _make_chunk(self, doc: ParsedDocument, text: str) -> Chunk:
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc.doc_id, doc_type=doc.doc_type, source=doc.source,
            text=text, department=doc.department, version=doc.version,
            effective_date=doc.effective_date,
        )
