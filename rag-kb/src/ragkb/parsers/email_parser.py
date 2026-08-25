import email
from email import policy

from ragkb.models import ParsedDocument
from ragkb.parsers.base import DocumentParser


class EmailParser(DocumentParser):
    """邮件：正文 + 附件递归解析。"""

    def parse(self, path, doc_id, source, department="", version="",
              effective_date=""):
        with open(path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                if ct == "text/plain":
                    body += (part.get_content() or "")
        else:
            body = msg.get_content() or ""

        text_parts = [f"主题：{msg.get('Subject', '')}", body.strip()]
        return ParsedDocument(
            doc_id=doc_id, doc_type="email", source=source,
            text="\n".join(t for t in text_parts if t),
            tables=[], images=[],
            department=department, version=version, effective_date=effective_date,
        )
