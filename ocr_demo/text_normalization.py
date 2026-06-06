import re


def normalize_ocr_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"[-_=*~]{3,}", " ", text)
    text = text.replace("\\n", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    normalized_lines = []
    for raw_line in text.split("\n"):
        line = re.sub(r"[^\S\n]+", " ", raw_line)
        line = re.sub(r",([^\s])", r", \1", line)
        line = re.sub(r"\b(Q|P|TP|TX|TT|H|\u0110)\.\s+", r"\g<1>.", line, flags=re.IGNORECASE)
        line = line.strip()
        if line:
            normalized_lines.append(line)

    return "\n".join(normalized_lines)
