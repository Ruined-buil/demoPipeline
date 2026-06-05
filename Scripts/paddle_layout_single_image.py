import argparse
import json
import os
from pathlib import Path

import cv2

os.environ.setdefault("DISABLE_MODELSCOPE", "1")
os.environ.setdefault("PADDLEX_DISABLE_MODELSCOPE", "1")

from paddleocr import LayoutDetection


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PaddleOCR PP-DocLayoutV2 on a single image, merge fragmented text "
            "regions, save crops, metadata, and an annotated preview."
        )
    )
    parser.add_argument("input_image", help="Path to the input image.")
    parser.add_argument(
        "--output-dir",
        default=str(Path("demoPipeline") / "outputs"),
        help="Base output directory. A subfolder named after the input image stem is created.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Execution device for LayoutDetection.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Detection threshold passed to PP-DocLayoutV2.",
    )
    parser.add_argument(
        "--y-thresh",
        type=int,
        default=15,
        help="Vertical distance threshold when merging fragmented text boxes.",
    )
    parser.add_argument(
        "--x-thresh",
        type=int,
        default=10,
        help="Horizontal gap threshold when merging fragmented text boxes.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import paddle

        if paddle.is_compiled_with_cuda():
            return "gpu"
    except Exception:
        pass

    return "cpu"


def merge_fragmented_boxes(boxes, y_thresh: int = 20, x_thresh: int = 10):
    text_boxes = [box for box in boxes if box.get("label") == "text"]
    other_boxes = [box for box in boxes if box.get("label") != "text"]

    if not text_boxes:
        return boxes

    text_boxes = sorted(text_boxes, key=lambda box: (box["coordinate"][1], box["coordinate"][0]))
    merged_groups = []

    for current_box in text_boxes:
        c_x1, c_y1, c_x2, c_y2 = current_box["coordinate"]
        matched_group_idx = -1

        for i in range(len(merged_groups) - 1, -1, -1):
            merged_box = merged_groups[i]
            m_x1, m_y1, m_x2, m_y2 = merged_box["coordinate"]

            vertical_dist = c_y1 - m_y2
            horizontal_overlap = min(c_x2, m_x2) - max(c_x1, m_x1)

            if vertical_dist <= y_thresh and horizontal_overlap >= -x_thresh:
                matched_group_idx = i
                break

        if matched_group_idx != -1:
            target_box = merged_groups[matched_group_idx]
            target_box["coordinate"][0] = min(target_box["coordinate"][0], c_x1)
            target_box["coordinate"][1] = min(target_box["coordinate"][1], c_y1)
            target_box["coordinate"][2] = max(target_box["coordinate"][2], c_x2)
            target_box["coordinate"][3] = max(target_box["coordinate"][3], c_y2)
            target_box["score"] = (target_box.get("score", 0.0) + current_box.get("score", 0.0)) / 2.0
        else:
            copied_box = dict(current_box)
            copied_box["coordinate"] = list(current_box["coordinate"])
            merged_groups.append(copied_box)

    return merged_groups + other_boxes


def clamp_bbox(bbox, width: int, height: int):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def draw_annotated_image(image, regions):
    annotated = image.copy()
    color_map = {
        "text": (0, 200, 0),
        "table": (0, 140, 255),
    }

    for idx, region in enumerate(regions):
        label = region.get("label", "").lower()
        bbox = region.get("bbox_safe")
        if label not in color_map or not bbox:
            continue

        x1, y1, x2, y2 = bbox
        color = color_map[label]
        score = region.get("score")
        score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
        tag = f"{idx}:{label}:{score_text}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            tag,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def build_output_paths(base_output_dir: Path, image_path: Path):
    run_dir = base_output_dir / image_path.stem
    paths = {
        "run_dir": run_dir,
        "text_dir": run_dir / "text",
        "table_dir": run_dir / "table",
        "metadata_json": run_dir / "layout_result.json",
        "annotated_image": run_dir / f"{image_path.stem}_annotated.jpg",
    }

    for key in ("run_dir", "text_dir", "table_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    return paths


def process_single_image(
    input_image: Path,
    output_dir: Path,
    device: str,
    threshold: float,
    y_thresh: int,
    x_thresh: int,
):
    if not input_image.exists():
        raise FileNotFoundError(f"Input image does not exist: {input_image}")
    if input_image.suffix.lower() not in VALID_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image extension: {input_image.suffix}")

    image = cv2.imread(str(input_image))
    if image is None:
        raise RuntimeError(f"OpenCV could not read image: {input_image}")

    image_height, image_width = image.shape[:2]
    resolved_device = resolve_device(device)

    model = LayoutDetection(
        model_name="PP-DocLayoutV2",
        device=resolved_device,
        threshold=threshold,
    )

    results = model.predict(str(input_image), batch_size=1, layout_nms=True)
    result = results[0] if isinstance(results, list) and results else results
    regions = result.get("boxes", [])
    optimized_regions = merge_fragmented_boxes(regions, y_thresh=y_thresh, x_thresh=x_thresh)

    paths = build_output_paths(output_dir, input_image)

    metadata = {
        "image_path": str(input_image.resolve()),
        "image_name": input_image.name,
        "width": image_width,
        "height": image_height,
        "device": resolved_device,
        "threshold": threshold,
        "regions_data": [],
    }

    crop_counter = 0

    for region in optimized_regions:
        label = region.get("label", "").lower()
        if label not in {"text", "table"}:
            continue

        bbox = region.get("coordinate")
        if not bbox or len(bbox) != 4:
            continue

        safe_bbox = clamp_bbox(bbox, image_width, image_height)
        if safe_bbox is None:
            continue

        x1, y1, x2, y2 = safe_bbox
        crop = image[y1:y2, x1:x2]
        crop_filename = f"{input_image.stem}_{label}_{crop_counter:03d}.jpg"
        crop_dir = paths["text_dir"] if label == "text" else paths["table_dir"]
        crop_path = crop_dir / crop_filename
        cv2.imwrite(str(crop_path), crop)

        region_meta = dict(region)
        region_meta["crop_id"] = crop_filename
        region_meta["crop_path"] = str(crop_path.resolve())
        region_meta["bbox_safe"] = safe_bbox
        metadata["regions_data"].append(region_meta)
        crop_counter += 1

    annotated = draw_annotated_image(image, metadata["regions_data"])
    cv2.imwrite(str(paths["annotated_image"]), annotated)

    with paths["metadata_json"].open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return {
        "run_dir": str(paths["run_dir"].resolve()),
        "metadata_json": str(paths["metadata_json"].resolve()),
        "annotated_image": str(paths["annotated_image"].resolve()),
        "text_crops": str(paths["text_dir"].resolve()),
        "table_crops": str(paths["table_dir"].resolve()),
        "num_regions": len(metadata["regions_data"]),
        "device": resolved_device,
    }


def main():
    args = parse_args()
    summary = process_single_image(
        input_image=Path(args.input_image),
        output_dir=Path(args.output_dir),
        device=args.device,
        threshold=args.threshold,
        y_thresh=args.y_thresh,
        x_thresh=args.x_thresh,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
