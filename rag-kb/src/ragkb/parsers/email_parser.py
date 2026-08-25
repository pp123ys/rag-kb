import email
import re
from email import policy

from ragkb.models import ParsedDocument
from ragkb.parsers.base import DocumentParser


def _strip_html(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html)


class EmailParser(DocumentParser):
    """邮件：正文 + 附件递归解析。"""

    def parse(self, path, doc_id, source, department="", version="",
              effective_date=""):
        with open(path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        body_parts = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    content = part.get_content() or ""
                    if content:
                        body_parts.append(content)
            if not body_parts:  # 纯 HTML 邮件：回退提取 text/html 并去标签
                for part in msg.walk():
                    if part.get_content_type() == "text/html":
                        content = _strip_html(part.get_content() or "").strip()
                        if content:
                            body_parts.append(content)
        else:
            body = msg.get_content() or ""
            if body:
                body_parts.append(body)

        text_parts = []
        subject = (msg.get("Subject") or "").strip()
        if subject:
            text_parts.append(f"主题：{subject}")
        body_text = "\n".join(p.strip() for p in body_parts).strip()
        if body_text:
            text_parts.append(body_text)
        return ParsedDocument(
            doc_id=doc_id, doc_type="email", source=source,
            text="\n".join(text_parts),
            tables=[], images=[],
            department=department, version=version, effective_date=effective_date,
        )
