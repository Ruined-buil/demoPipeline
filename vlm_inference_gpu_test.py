import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from ocr_demo.config import BASE_MODEL_ID, MODEL_ID, OCR_INSTRUCTION
from ocr_demo.text_normalization import normalize_ocr_text
from ocr_demo.vlm_inference import infer_base_model_id, load_with_local_retry, resolve_local_snapshot_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone GPU VLM OCR inference test for one image crop.")
    parser.add_argument("image", type=Path, help="Path to one text crop image.")
    parser.add_argument("--model-id", default=MODEL_ID, help="LoRA adapter or full model id.")
    parser.add_argument(
        "--base-model-id",
        default=os.environ.get("QWEN_BASE_MODEL_ID") or BASE_MODEL_ID,
        help="Base model id for LoRA adapters. Defaults to QWEN_BASE_MODEL_ID or adapter_config.json.",
    )
    parser.add_argument("--instruction", default=OCR_INSTRUCTION, help="OCR prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda:0", help="Torch CUDA device, for example cuda:0.")
    return parser.parse_args()


def ensure_cuda(device: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise RuntimeError(f"Expected a CUDA device, got {device!r}.")

    if torch_device.index is not None and torch_device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {device!r}, but only {torch.cuda.device_count()} CUDA device(s) are available."
        )

    return torch_device


def load_model_and_processor_gpu(model_id: str, base_model_id: str | None, device: torch.device):
    resolved_base_model_id = base_model_id or infer_base_model_id(model_id)
    if not resolved_base_model_id:
        raise RuntimeError(
            "No base model id was provided and adapter_config.json did not contain base_model_name_or_path."
        )

    base_model = load_with_local_retry(
        AutoModelForImageTextToText,
        resolved_base_model_id,
        dtype="auto",
        trust_remote_code=True,
    )
    adapter_path = str(resolve_local_snapshot_dir(model_id) or model_id)
    model = PeftModel.from_pretrained(base_model, adapter_path, local_files_only=Path(adapter_path).exists())
    model.to(device)
    model.eval()

    processor = load_with_local_retry(AutoProcessor, model_id, trust_remote_code=True)
    return model, processor, resolved_base_model_id


def build_inputs(processor, image: Image.Image, instruction: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[prompt], images=[image], return_tensors="pt")


def move_inputs_to_device(inputs, device: torch.device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def run_inference_gpu(
    image_path: Path,
    model_id: str,
    base_model_id: str | None,
    instruction: str,
    max_new_tokens: int,
    device_name: str,
) -> dict[str, str]:
    device = ensure_cuda(device_name)
    model, processor, resolved_base_model_id = load_model_and_processor_gpu(model_id, base_model_id, device)
    image = Image.open(image_path).convert("RGB")
    inputs = move_inputs_to_device(build_inputs(processor, image, instruction), device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_length:]
    raw_text = processor.decode(generated_tokens, skip_special_tokens=True)

    return {
        "model_id": model_id,
        "base_model_id": resolved_base_model_id,
        "image_path": str(image_path),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "raw_text": raw_text,
        "normalized_text": normalize_ocr_text(raw_text),
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    result = run_inference_gpu(
        image_path=args.image,
        model_id=args.model_id,
        base_model_id=args.base_model_id,
        instruction=args.instruction,
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
