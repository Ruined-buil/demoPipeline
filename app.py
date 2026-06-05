import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
from PIL import Image


os.environ.setdefault("DISABLE_MODELSCOPE", "1")
os.environ.setdefault("PADDLEX_DISABLE_MODELSCOPE", "1")

APP_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = APP_DIR / "outputs"
MODEL_ID = "PhYen/qwen_lora_base"
BASE_MODEL_ID = os.environ.get("QWEN_BASE_MODEL_ID", "").strip()
LAYOUT_THRESHOLD = 0.3
LAYOUT_DEVICE = "cpu"
MERGE_Y_THRESH = 15
MERGE_X_THRESH = 10

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from Scripts.paddle_layout_single_image import (  # noqa: E402
    clamp_bbox,
    draw_annotated_image,
    merge_fragmented_boxes,
)


OCR_INSTRUCTION = """You are a highly precise OCR system. Extract the Vietnamese text from this image exactly as it appears. Strict rules to follow:
- Exact transcription: Preserve all original diacritics, punctuation, abbreviations, and spacing exactly.
- No auto-completion: If a word appears truncated or cut off at the edge of the image, transcribe only the visible letters. Do NOT guess or auto-complete the missing letters.
- Output only text: Return ONLY the raw extracted text. No formatting, no markdown tags, no introductory or concluding remarks"""


@dataclass
class VLMResources:
    model: Any
    tokenizer: Any
    processor_id: str
    processor: Any | None = None
    processor_load_error: str | None = None


_layout_model = None
_vlm_resources: VLMResources | None = None


def from_pretrained_with_local_retry(loader, model_id: str, **kwargs):
    try:
        return loader.from_pretrained(model_id, **kwargs)
    except Exception as first_exc:
        local_kwargs = dict(kwargs)
        local_kwargs["local_files_only"] = True
        local_model_id = str(resolve_local_snapshot_dir(model_id) or model_id)
        try:
            return loader.from_pretrained(local_model_id, **local_kwargs)
        except Exception as local_exc:
            raise RuntimeError(
                f"remote/cache load failed: {type(first_exc).__name__}: {first_exc}; "
                f"local cache load failed: {type(local_exc).__name__}: {local_exc}"
            ) from local_exc


def resolve_local_snapshot_dir(model_id: str) -> Path | None:
    if not model_id or "/" not in model_id:
        return None

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


def normalize_ocr_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"[-_=*~]{3,}", " ", text)
    text = text.replace("\\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r",([^\s])", r", \1", text)
    text = re.sub(r"\b(Q|P|TP|TX|TT|H|Đ)\.\s+", r"\g<1>.", text, flags=re.IGNORECASE)
    return text.lower().strip()


def get_layout_model():
    global _layout_model
    if _layout_model is None:
        from paddleocr import LayoutDetection

        _layout_model = LayoutDetection(
            model_name="PP-DocLayoutV2",
            device=LAYOUT_DEVICE,
            threshold=LAYOUT_THRESHOLD,
        )
    return _layout_model


def get_vlm_resources() -> VLMResources:
    global _vlm_resources
    if _vlm_resources is not None:
        return _vlm_resources

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Could not import VLM dependencies. Install torch, transformers, accelerate, pillow, and huggingface-hub."
        ) from exc

    load_errors = []
    try:
        try:
            model = from_pretrained_with_local_retry(
                AutoModel,
                MODEL_ID,
                dtype="auto",
                trust_remote_code=True,
            )
        except TypeError:
            model = from_pretrained_with_local_retry(
                AutoModel,
                MODEL_ID,
                torch_dtype="auto",
                trust_remote_code=True,
            )
        tokenizer = from_pretrained_with_local_retry(AutoTokenizer, MODEL_ID, trust_remote_code=True)
        model.to("cpu")
        model.eval()
        torch.set_grad_enabled(False)
        processor_id = BASE_MODEL_ID or MODEL_ID
        _vlm_resources = VLMResources(model=model, tokenizer=tokenizer, processor_id=processor_id)
        return _vlm_resources
    except Exception as exc:
        load_errors.append(f"standalone load failed: {type(exc).__name__}: {exc}")

    if BASE_MODEL_ID:
        try:
            from peft import PeftModel

            try:
                base_model = from_pretrained_with_local_retry(
                    AutoModel,
                    BASE_MODEL_ID,
                    dtype="auto",
                    trust_remote_code=True,
                )
            except TypeError:
                base_model = from_pretrained_with_local_retry(
                    AutoModel,
                    BASE_MODEL_ID,
                    torch_dtype="auto",
                    trust_remote_code=True,
                )
            try:
                model = PeftModel.from_pretrained(base_model, MODEL_ID)
            except Exception as first_exc:
                local_adapter_id = str(resolve_local_snapshot_dir(MODEL_ID) or MODEL_ID)
                try:
                    model = PeftModel.from_pretrained(base_model, local_adapter_id, local_files_only=True)
                except Exception as local_exc:
                    raise RuntimeError(
                        f"remote/cache adapter load failed: {type(first_exc).__name__}: {first_exc}; "
                        f"local adapter load failed: {type(local_exc).__name__}: {local_exc}"
                    ) from local_exc
            try:
                tokenizer = from_pretrained_with_local_retry(AutoTokenizer, MODEL_ID, trust_remote_code=True)
            except Exception:
                tokenizer = from_pretrained_with_local_retry(AutoTokenizer, BASE_MODEL_ID, trust_remote_code=True)
            model.to("cpu")
            model.eval()
            torch.set_grad_enabled(False)
            _vlm_resources = VLMResources(model=model, tokenizer=tokenizer, processor_id=BASE_MODEL_ID)
            return _vlm_resources
        except Exception as exc:
            load_errors.append(f"PEFT adapter load failed: {type(exc).__name__}: {exc}")

    base_hint = (
        f" QWEN_BASE_MODEL_ID is currently set to {BASE_MODEL_ID!r}."
        if BASE_MODEL_ID
        else " If this repo is a LoRA adapter, set QWEN_BASE_MODEL_ID to the base model, for example: $env:QWEN_BASE_MODEL_ID='unsloth/Qwen3.5-0.8B'."
    )
    raise RuntimeError(f"Could not load {MODEL_ID}. {' | '.join(load_errors)}.{base_hint}")


def ensure_processor(resources: VLMResources):
    if resources.processor is not None:
        return resources.processor
    if resources.processor_load_error:
        raise RuntimeError(resources.processor_load_error)

    from transformers import AutoProcessor

    attempted_ids = []
    load_errors = []
    for processor_id in (resources.processor_id, BASE_MODEL_ID, MODEL_ID):
        if not processor_id or processor_id in attempted_ids:
            continue
        attempted_ids.append(processor_id)
        try:
            resources.processor = from_pretrained_with_local_retry(
                AutoProcessor,
                processor_id,
                trust_remote_code=True,
            )
            resources.processor_id = processor_id
            return resources.processor
        except Exception as exc:
            load_errors.append(f"{processor_id}: {type(exc).__name__}: {exc}")

    resources.processor_load_error = (
        "Could not load AutoProcessor; processor-based image input is unavailable. "
        + " | ".join(load_errors)
    )
    raise RuntimeError(resources.processor_load_error)


def move_inputs_to_cpu(inputs):
    if hasattr(inputs, "to"):
        return inputs.to("cpu")

    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to("cpu") if hasattr(value, "to") else value
    return moved


def decode_generated(resources: VLMResources, inputs, outputs) -> str:
    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    input_length = input_ids.shape[-1]
    generated = outputs[0][input_length:] if len(outputs.shape) > 1 else outputs[input_length:]

    decoder = resources.processor if resources.processor is not None and hasattr(resources.processor, "decode") else resources.tokenizer
    return decoder.decode(generated, skip_special_tokens=True)


def build_messages(image: Image.Image):
    del image
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": OCR_INSTRUCTION},
            ],
        }
    ]


def generate_with_tokenizer(resources: VLMResources, image: Image.Image) -> str:
    tokenizer = resources.tokenizer
    messages = build_messages(image)
    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    else:
        input_text = OCR_INSTRUCTION

    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    inputs = move_inputs_to_cpu(inputs)

    outputs = resources.model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True,
        do_sample=False,
    )
    return decode_generated(resources, inputs, outputs)


def generate_with_processor(resources: VLMResources, image: Image.Image) -> str:
    processor = ensure_processor(resources)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": OCR_INSTRUCTION},
            ],
        }
    ]

    if hasattr(processor, "apply_chat_template"):
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    else:
        inputs = processor(text=[OCR_INSTRUCTION], images=[image], return_tensors="pt")

    inputs = move_inputs_to_cpu(inputs)
    outputs = resources.model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True,
        do_sample=False,
    )
    return decode_generated(resources, inputs, outputs)


def run_vlm_on_crop(crop_path: Path) -> str:
    from vlm_inference_test import run_inference

    result = run_inference(
        image_path=crop_path,
        model_id=MODEL_ID,
        base_model_id=BASE_MODEL_ID or None,
        instruction=OCR_INSTRUCTION,
        max_new_tokens=128,
    )
    return result["raw_text"]


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


def extract_result_boxes(results) -> list[dict[str, Any]]:
    if isinstance(results, list):
        result = results[0] if results else {}
    elif isinstance(results, dict):
        result = results
    else:
        try:
            result_list = list(results)
            result = result_list[0] if result_list else {}
        except TypeError:
            result = results

    if isinstance(result, dict):
        return result.get("boxes", [])
    if hasattr(result, "get"):
        return result.get("boxes", [])
    return []


def sorted_text_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text_regions = []
    for region in regions:
        label = str(region.get("label", "")).lower()
        bbox = region.get("bbox_safe") or region.get("coordinate")
        if label == "text" and bbox and len(bbox) == 4:
            text_regions.append(region)
    return sorted(text_regions, key=lambda item: (item["bbox_safe"][1], item["bbox_safe"][0]))


def detect_text_crops(input_path: Path, run_dir: Path):
    image = cv2.imread(str(input_path))
    if image is None:
        raise RuntimeError("OpenCV could not read the uploaded image.")

    height, width = image.shape[:2]
    layout_model = get_layout_model()
    results = layout_model.predict(str(input_path), batch_size=1, layout_nms=True)
    boxes = extract_result_boxes(results)
    merged_boxes = merge_fragmented_boxes(boxes, y_thresh=MERGE_Y_THRESH, x_thresh=MERGE_X_THRESH)

    safe_regions = []
    for region in merged_boxes:
        label = str(region.get("label", "")).lower()
        if label != "text":
            continue

        bbox = region.get("coordinate")
        if not bbox or len(bbox) != 4:
            continue

        safe_bbox = clamp_bbox(bbox, width, height)
        if safe_bbox is None:
            continue

        region_meta = dict(region)
        region_meta["label"] = label
        region_meta["bbox_safe"] = safe_bbox
        safe_regions.append(region_meta)

    safe_regions = sorted_text_regions(safe_regions)
    crop_paths = []
    for idx, region in enumerate(safe_regions):
        x1, y1, x2, y2 = region["bbox_safe"]
        crop = image[y1:y2, x1:x2]
        crop_path = run_dir / "text" / f"text_{idx:03d}.jpg"
        cv2.imwrite(str(crop_path), crop)
        region["crop_id"] = crop_path.name
        region["crop_path"] = str(crop_path.resolve())
        crop_paths.append(str(crop_path))

    annotated = draw_annotated_image(image, safe_regions)
    annotated_path = run_dir / "annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated)

    metadata = {
        "input_path": str(input_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "layout_model": "PP-DocLayoutV2",
        "device": LAYOUT_DEVICE,
        "threshold": LAYOUT_THRESHOLD,
        "merge_y_thresh": MERGE_Y_THRESH,
        "merge_x_thresh": MERGE_X_THRESH,
        "regions_data": safe_regions,
    }
    with (run_dir / "layout_result.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return str(annotated_path), crop_paths, safe_regions


def process_image(image: Image.Image | None):
    if image is None:
        return None, [], "", "Upload an image before running OCR."

    run_dir = make_run_dir()
    try:
        input_path = save_uploaded_image(image, run_dir)
        annotated_path, crop_paths, regions = detect_text_crops(input_path, run_dir)
    except Exception as exc:
        return None, [], "", f"Layout detection failed: {exc}"

    if not crop_paths:
        return annotated_path, [], "", f"No text regions detected. Outputs saved to {run_dir}."

    raw_predictions = []
    normalized_predictions = []
    failed_count = 0

    for crop_path in crop_paths:
        try:
            raw_text = run_vlm_on_crop(Path(crop_path))
            clean_text = normalize_ocr_text(raw_text)
            raw_predictions.append({"crop_path": crop_path, "raw_text": raw_text, "normalized_text": clean_text})
            if clean_text:
                normalized_predictions.append(clean_text)
        except Exception as exc:
            failed_count += 1
            raw_predictions.append({"crop_path": crop_path, "error": str(exc)})

    final_text = normalize_ocr_text(" ".join(normalized_predictions))
    with (run_dir / "ocr_result.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_id": MODEL_ID,
                "base_model_id": BASE_MODEL_ID or "adapter_config.json",
                "vlm_module": "vlm_inference_test.run_inference",
                "num_text_regions": len(regions),
                "num_failed_crops": failed_count,
                "predictions": raw_predictions,
                "final_text": final_text,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    status = f"Processed {len(crop_paths)} text crop(s). Outputs saved to {run_dir}."
    if failed_count:
        status += f" {failed_count} crop(s) failed VLM inference."
    return annotated_path, crop_paths, final_text, status


def clear_outputs():
    return None, None, [], "", ""


def build_app():
    with gr.Blocks(title="CPU OCR Demo") as demo:
        gr.Markdown("# CPU OCR Demo")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Uploaded image", type="pil", sources=["upload"])
                with gr.Row():
                    run_button = gr.Button("Run OCR", variant="primary")
                    clear_button = gr.Button("Clear")
                status_output = gr.Textbox(label="Status", interactive=False, lines=3)
            with gr.Column():
                annotated_output = gr.Image(label="Paddle annotated preview", type="filepath")
                crop_gallery = gr.Gallery(
                    label="Detected text crops",
                    columns=3,
                    object_fit="contain",
                    height=320,
                )

        final_text = gr.Textbox(label="Final normalized OCR text", interactive=False, lines=8)

        run_button.click(
            fn=process_image,
            inputs=image_input,
            outputs=[annotated_output, crop_gallery, final_text, status_output],
        )
        clear_button.click(
            fn=clear_outputs,
            inputs=None,
            outputs=[image_input, annotated_output, crop_gallery, final_text, status_output],
        )

    return demo


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(share=False)
