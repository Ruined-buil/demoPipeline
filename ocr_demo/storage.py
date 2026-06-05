import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .config import OUTPUT_ROOT


def make_run_dir() -> Path:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = OUTPUT_ROOT / run_id
    (run_dir / "input").mkdir(parents=True, exist_ok=True)
    (run_dir / "text").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_uploaded_image(image: Image.Image, run_dir: Path) -> Path:
    input_path = run_dir / "input" / "uploaded.png"
    image.convert("RGB").save(input_path)
    return input_path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
