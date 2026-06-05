import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL_ID = "PhYen/qwen_lora_base"
DEFAULT_INSTRUCTION = "Write the text representation for this image."
_MODEL_CACHE: dict[tuple[str, str | None], tuple[Any, Any, str]] = {}


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


def resolve_local_snapshot_dir(model_id: str) -> Path | None:
    if not model_id or "/" not in model_id:
        path = Path(model_id)
        return path if path.exists() else None

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return None

    namespace, name = model_id.split("/", 1)
    repo_cache = Path(HF_HUB_CACHE) / f"models--{namespace}--{name}"
    ref_path = repo_cache / "refs" / "main"
    if not ref_path.exists():
        return None

    revision = ref_path.read_text(encoding="utf-8").strip()
    snapshot_dir = repo_cache / "snapshots" / revision
    return snapshot_dir if snapshot_dir.exists() else None


def load_with_local_retry(loader: Any, model_id: str, **kwargs):
    local_dir = resolve_local_snapshot_dir(model_id)
    if local_dir:
        local_kwargs = dict(kwargs)
        local_kwargs["local_files_only"] = True
        try:
            return loader.from_pretrained(str(local_dir), **local_kwargs)
        except Exception as local_first_exc:
            try:
                return loader.from_pretrained(model_id, **kwargs)
            except Exception as remote_exc:
                raise RuntimeError(
                    f"Could not load {model_id}. "
                    f"local error: {type(local_first_exc).__name__}: {local_first_exc}; "
                    f"remote/cache error: {type(remote_exc).__name__}: {remote_exc}"
                ) from remote_exc

    try:
        return loader.from_pretrained(model_id, **kwargs)
    except Exception as first_exc:
        local_id = str(local_dir or model_id)
        local_kwargs = dict(kwargs)
        local_kwargs["local_files_only"] = True
        try:
            return loader.from_pretrained(local_id, **local_kwargs)
        except Exception as local_exc:
            raise RuntimeError(
                f"Could not load {model_id}. "
                f"remote/cache error: {type(first_exc).__name__}: {first_exc}; "
                f"local error: {type(local_exc).__name__}: {local_exc}"
            ) from local_exc


def infer_base_model_id(adapter_model_id: str) -> str | None:
    adapter_dir = resolve_local_snapshot_dir(adapter_model_id)
    if not adapter_dir:
        return None

    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        return None

    with adapter_config.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("base_model_name_or_path")


def load_model_and_processor(model_id: str, base_model_id: str | None):
    base_model_id = base_model_id or infer_base_model_id(model_id)
    if not base_model_id:
        raise RuntimeError(
            "No base model id was provided and adapter_config.json did not contain base_model_name_or_path."
        )

    cache_key = (model_id, base_model_id)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    base_model = load_with_local_retry(
        AutoModelForImageTextToText,
        base_model_id,
        dtype="auto",
        trust_remote_code=True,
    )

    adapter_path = str(resolve_local_snapshot_dir(model_id) or model_id)
    model = PeftModel.from_pretrained(base_model, adapter_path, local_files_only=Path(adapter_path).exists())
    model.to("cpu")
    model.eval()

    processor = load_with_local_retry(AutoProcessor, model_id, trust_remote_code=True)
    _MODEL_CACHE[cache_key] = (model, processor, base_model_id)
    return _MODEL_CACHE[cache_key]


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
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    return inputs


def move_inputs_to_cpu(inputs):
    if hasattr(inputs, "to"):
        return inputs.to("cpu")
    return {key: value.to("cpu") if hasattr(value, "to") else value for key, value in inputs.items()}


def run_inference(
    image_path: Path,
    model_id: str = DEFAULT_MODEL_ID,
    base_model_id: str | None = None,
    instruction: str = DEFAULT_INSTRUCTION,
    max_new_tokens: int = 128,
) -> dict[str, str]:
    model, processor, resolved_base_model_id = load_model_and_processor(model_id, base_model_id)
    image = Image.open(image_path).convert("RGB")
    inputs = move_inputs_to_cpu(build_inputs(processor, image, instruction))

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
        "raw_text": raw_text,
        "normalized_text": normalize_ocr_text(raw_text),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone CPU VLM OCR inference test for one image crop.")
    parser.add_argument("image", type=Path, help="Path to one text crop image.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="LoRA adapter or full model id.")
    parser.add_argument(
        "--base-model-id",
        default=os.environ.get("QWEN_BASE_MODEL_ID"),
        help="Base model id for LoRA adapters. Defaults to QWEN_BASE_MODEL_ID or adapter_config.json.",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="OCR prompt.")
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
