from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple


VOC_ROOT = Path("../VOC2028/VOC2028")
OUTPUT_ROOT = Path("data/real")
CLASS_MAP = {"person": 0, "hat": 1}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def convert_box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: float,
    image_height: float,
) -> Tuple[float, float, float, float]:
    xmin = clamp(xmin, 0.0, image_width)
    xmax = clamp(xmax, 0.0, image_width)
    ymin = clamp(ymin, 0.0, image_height)
    ymax = clamp(ymax, 0.0, image_height)
    center_x = ((xmin + xmax) / 2.0) / image_width
    center_y = ((ymin + ymax) / 2.0) / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    return center_x, center_y, width, height


def convert_annotation(xml_path: Path) -> Tuple[List[str], Dict[str, int]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing image size in {xml_path}")

    image_width = float(size.findtext("width", "0"))
    image_height = float(size.findtext("height", "0"))
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image dimensions in {xml_path}")

    labels: List[str] = []
    skipped: Dict[str, int] = {}
    for object_node in root.findall("object"):
        class_name = object_node.findtext("name", "").strip().lower()
        if class_name not in CLASS_MAP:
            skipped[class_name] = skipped.get(class_name, 0) + 1
            continue

        box = object_node.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        center_x, center_y, width, height = convert_box(
            xmin, ymin, xmax, ymax, image_width, image_height
        )
        if width <= 0 or height <= 0:
            continue
        labels.append(
            f"{CLASS_MAP[class_name]} {center_x:.6f} {center_y:.6f} "
            f"{width:.6f} {height:.6f}"
        )

    return labels, skipped


def main() -> None:
    image_source = VOC_ROOT / "JPEGImages"
    annotation_source = VOC_ROOT / "Annotations"
    image_destination = OUTPUT_ROOT / "images"
    label_destination = OUTPUT_ROOT / "labels"
    image_destination.mkdir(parents=True, exist_ok=True)
    label_destination.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped_objects: Dict[str, int] = {}
    for image_path in sorted(image_source.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        xml_path = annotation_source / f"{image_path.stem}.xml"
        if not xml_path.exists():
            continue

        labels, skipped = convert_annotation(xml_path)
        target_image = image_destination / image_path.name
        target_label = label_destination / f"{image_path.stem}.txt"
        shutil.copy2(image_path, target_image)
        target_label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        converted += 1
        for class_name, count in skipped.items():
            skipped_objects[class_name] = skipped_objects.get(class_name, 0) + count

    report = {
        "source": str(VOC_ROOT),
        "output": str(OUTPUT_ROOT),
        "class_map": {"0": "worker", "1": "helmet"},
        "converted_images": converted,
        "skipped_objects": skipped_objects,
    }
    report_path = OUTPUT_ROOT / "voc_conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()