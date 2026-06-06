from pathlib import Path

from PIL import Image

from .config import BASE_MODEL_ID, MODEL_ID, OCR_INSTRUCTION
from .layout_detection import detect_text_crops
from .storage import make_run_dir, save_uploaded_image, write_json
from .text_normalization import normalize_ocr_text
from .vlm_inference import run_inference


def run_vlm_on_crop(crop_path: Path) -> str:
    result = run_inference(
        image_path=crop_path,
        model_id=MODEL_ID,
        base_model_id=BASE_MODEL_ID,
        instruction=OCR_INSTRUCTION,
        max_new_tokens=128,
    )
    return result["raw_text"]


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
    raw_output_lines = []
    normalized_predictions = []
    failed_count = 0

    for crop_path in crop_paths:
        try:
            raw_text = run_vlm_on_crop(Path(crop_path))
            clean_text = normalize_ocr_text(raw_text)
            raw_predictions.append({"crop_path": crop_path, "raw_text": raw_text, "normalized_text": clean_text})
            if raw_text.strip():
                raw_output_lines.append(raw_text.strip())
            if clean_text:
                normalized_predictions.append(clean_text)
        except Exception as exc:
            failed_count += 1
            raw_predictions.append({"crop_path": crop_path, "error": str(exc)})

    final_raw_text = "\n\n".join(raw_output_lines)
    final_normalized_text = "\n".join(normalized_predictions)
    write_json(
        run_dir / "ocr_result.json",
        {
            "model_id": MODEL_ID,
            "base_model_id": BASE_MODEL_ID or "adapter_config.json",
            "vlm_module": "ocr_demo.vlm_inference.run_inference",
            "num_text_regions": len(regions),
            "num_failed_crops": failed_count,
            "predictions": raw_predictions,
            "final_raw_text": final_raw_text,
            "final_normalized_text": final_normalized_text,
        },
    )

    status = f"Processed {len(crop_paths)} text crop(s). Outputs saved to {run_dir}."
    if failed_count:
        status += f" {failed_count} crop(s) failed VLM inference."
    return annotated_path, crop_paths, final_raw_text, status
