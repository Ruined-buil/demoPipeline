from pathlib import Path

from PIL import Image

from .config import BASE_MODEL_ID, MODEL_ID, OCR_INSTRUCTION
from .storage import make_run_dir, save_uploaded_image, write_json
from .text_normalization import normalize_ocr_text
from .vlm_inference import run_inference


def process_crop_image(image: Image.Image | None):
    if image is None:
        return None, "", "Upload a cropped image before running OCR."

    run_dir = make_run_dir()
    try:
        input_path = save_uploaded_image(image, run_dir)
        result = run_inference(
            image_path=input_path,
            model_id=MODEL_ID,
            base_model_id=BASE_MODEL_ID,
            instruction=OCR_INSTRUCTION,
            max_new_tokens=128,
        )
    except Exception as exc:
        return None, "", f"VLM inference failed: {exc}"

    write_json(
        run_dir / "ocr_result.json",
        {
            "model_id": MODEL_ID,
            "base_model_id": BASE_MODEL_ID or "adapter_config.json",
            "vlm_module": "ocr_demo.vlm_inference.run_inference",
            "input_path": str(input_path.resolve()),
            "raw_text": result["raw_text"],
            "final_raw_text": result["raw_text"],
            "final_normalized_text": normalize_ocr_text(result["raw_text"]),
        },
    )

    status = f"Processed 1 cropped image. Outputs saved to {run_dir}."
    return str(input_path), normalize_ocr_text(result["raw_text"]), status
