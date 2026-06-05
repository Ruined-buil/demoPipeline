# Gradio OCR Demo Plan

## Goal

Build a CPU-only local Gradio demo that accepts one uploaded image, detects text regions with PaddleOCR layout detection, runs each detected text crop through `PhYen/qwen_lora_base`, and displays only normalized OCR text as the final result.

## Flow

1. User uploads a single image in the Gradio UI.
2. The app saves the uploaded image under `demoPipeline/outputs/<run_id>/input/`.
3. PaddleOCR `LayoutDetection` runs on CPU with:
   - `model_name="PP-DocLayoutV2"`
   - `threshold=0.3`
   - `device="cpu"`
4. Detected layout boxes are merged with `merge_fragmented_boxes(..., y_thresh=15, x_thresh=10)`.
5. The app keeps only boxes where `label == "text"` and drops `table` plus every other label.
6. Text boxes are clamped to image bounds, sorted top-to-bottom and left-to-right, cropped, and saved under `demoPipeline/outputs/<run_id>/text/`.
7. Each text crop is passed independently to `PhYen/qwen_lora_base`.
8. Raw crop predictions are normalized and joined into the final text output.
9. The UI shows:
   - Uploaded image preview.
   - Paddle annotated preview.
   - Detected text crop gallery.
   - Final normalized OCR text.
   - Basic status or error output.

## Dependencies

Install from `demoPipeline/requirements.txt`:

```powershell
demoPipeline\venv\Scripts\python.exe -m pip install -r demoPipeline\requirements.txt
```

The demo depends on PaddleOCR layout detection, OpenCV, Gradio, Pillow, PyTorch, Transformers, Accelerate, and Hugging Face Hub.

## Commands

Check imports:

```powershell
demoPipeline\venv\Scripts\python.exe -c "import gradio, torch, transformers, paddleocr, cv2"
```

Check syntax:

```powershell
demoPipeline\venv\Scripts\python.exe -m py_compile demoPipeline\app.py
```

Run the demo:

```powershell
demoPipeline\venv\Scripts\python.exe demoPipeline\app.py
```

The app launches locally with `share=False` and Gradio queueing enabled for one request at a time.

## UI Behavior

- The image upload component is the uploaded image preview.
- `Run OCR` executes layout detection and VLM inference.
- `Clear` resets the image, annotated preview, crop gallery, text output, and status.
- If no text regions are detected, the app returns an annotated preview, an empty crop gallery, an empty OCR text result, and a clear status message.
- If one crop fails VLM inference, the app continues with remaining crops and reports the failed crop count.
- If model loading fails, the app reports a dependency or Hugging Face model access error.

## Normalization

The OCR text normalization follows the notebook strategy:

- Remove model special tags like `<|...|>`.
- Remove separator noise such as repeated `-`, `_`, `=`, `*`, or `~`.
- Replace literal `\n` sequences with spaces.
- Collapse real whitespace to a single space.
- Insert missing spaces after commas.
- Fix Vietnamese address abbreviation spacing such as `Q. Gò Vấp` to `Q.Gò Vấp`.
- Lowercase and strip.

## Assumptions

- `PhYen/qwen_lora_base` is public or accessible in the local Hugging Face environment.
- CPU inference is acceptable even if slow.
- The demo uses individual Paddle text crops, not the full uploaded image.
- Public Gradio sharing remains disabled by default.

## Test Plan

1. Import check:
   - `demoPipeline\venv\Scripts\python.exe -c "import gradio, torch, transformers, paddleocr, cv2"`
2. Static syntax check:
   - `demoPipeline\venv\Scripts\python.exe -m py_compile demoPipeline\app.py`
3. Smoke test:
   - Run `demoPipeline\venv\Scripts\python.exe demoPipeline\app.py`.
   - Upload one receipt image.
   - Confirm text crops are produced and table crops are not shown.
   - Confirm the annotated preview appears.
   - Confirm normalized lowercase OCR text appears.
4. Edge cases:
   - No text detected.
   - Unreadable image.
   - Multiple text regions.
   - CPU-only environment with no CUDA.
