import re


def normalize_ocr_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"[-_=*~]{3,}", " ", text)
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r",([^\s])", r", \1", text)
    text = re.sub(r"\b(Q|P|TP|TX|TT|H|\u0110)\.\s+", r"\g<1>.", text, flags=re.IGNORECASE)
    return text.lower().strip()
