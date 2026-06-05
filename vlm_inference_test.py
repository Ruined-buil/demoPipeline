import argparse
import json
import os
import sys
from pathlib import Path

from ocr_demo.config import BASE_MODEL_ID, MODEL_ID, OCR_INSTRUCTION
from ocr_demo.vlm_inference import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone CPU VLM OCR inference test for one image crop.")
    parser.add_argument("image", type=Path, help="Path to one text crop image.")
    parser.add_argument("--model-id", default=MODEL_ID, help="LoRA adapter or full model id.")
    parser.add_argument(
        "--base-model-id",
        default=os.environ.get("QWEN_BASE_MODEL_ID") or BASE_MODEL_ID,
        help="Base model id for LoRA adapters. Defaults to QWEN_BASE_MODEL_ID or adapter_config.json.",
    )
    parser.add_argument("--instruction", default=OCR_INSTRUCTION, help="OCR prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    result = run_inference(
        image_path=args.image,
        model_id=args.model_id,
        base_model_id=args.base_model_id,
        instruction=args.instruction,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
