import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = APP_DIR / "outputs"

MODEL_ID = "Ben248/qwen3.5-0.8Bv1"
BASE_MODEL_ID = os.environ.get("QWEN_BASE_MODEL_ID", "").strip() or None

LAYOUT_THRESHOLD = 0.2
LAYOUT_DEVICE = "cpu"
MERGE_Y_THRESH = 15
MERGE_X_THRESH = 10

OCR_INSTRUCTION = """You are a highly precise OCR system. Extract the Vietnamese text from this image exactly as it appears. Strict rules to follow:
- Exact transcription: Preserve all original diacritics, punctuation, abbreviations, and spacing exactly.
- No auto-completion: If a word appears truncated or cut off at the edge of the image, transcribe only the visible letters. Do NOT guess or auto-complete the missing letters.
- Output only text: Return ONLY the raw extracted text. No formatting, no markdown tags, no introductory or concluding remarks"""
