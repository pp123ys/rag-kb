import re

# 页眉页脚 / 页码 / 水印模式（可扩展）
_FOOTER_PATTERNS = [
    re.compile(r"^\s*第\s*\d+\s*页\s*[/／]\s*共\s*\d+\s*页\s*$"),
    re.compile(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$"),
    re.compile(r"本文档仅供内部使用"),
    re.compile(r"机密|Confidential"),
]
_GARBAGE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """规则化清洗管线：去页眉页脚/乱码、合并空行、统一空白。"""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")  # 保留单空行作段落分隔
            continue
        if any(p.search(line) for p in _FOOTER_PATTERNS):
            continue  # 丢弃页眉页脚行
        line = _GARBAGE.sub("", line)
        line = re.sub(r"[ \t\u3000]+", " ", line)  # 统一空白
        lines.append(line)

    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)  # 合并多余空行
    return out.strip()
