import os
from pathlib import Path
from typing import Any

import cv2

from .config import LAYOUT_DEVICE, LAYOUT_THRESHOLD, MERGE_X_THRESH, MERGE_Y_THRESH
from .storage import write_json


os.environ.setdefault("DISABLE_MODELSCOPE", "1")
os.environ.setdefault("PADDLEX_DISABLE_MODELSCOPE", "1")

from Scripts.paddle_layout_single_image import (  # noqa: E402
    clamp_bbox,
    draw_annotated_image,
    merge_fragmented_boxes,
)


_layout_model = None


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

    write_json(
        run_dir / "layout_result.json",
        {
            "input_path": str(input_path.resolve()),
            "run_dir": str(run_dir.resolve()),
            "layout_model": "PP-DocLayoutV2",
            "device": LAYOUT_DEVICE,
            "threshold": LAYOUT_THRESHOLD,
            "merge_y_thresh": MERGE_Y_THRESH,
            "merge_x_thresh": MERGE_X_THRESH,
            "regions_data": safe_regions,
        },
    )

    return str(annotated_path), crop_paths, safe_regions
