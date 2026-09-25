from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List
import json


@dataclass
class SyntheticGenerationConfig:
    output_dir: str = "data/synthetic/generated"
    scenes: int = 50
    workers_per_scene: int = 3
    lighting_variations: List[str] | None = None
    background_types: List[str] | None = None

    def __post_init__(self) -> None:
        self.lighting_variations = self.lighting_variations or ["daylight", "overcast", "dim", "golden_hour"]
        self.background_types = self.background_types or ["warehouse", "factory_floor", "loading_bay", "assembly_line"]


def generate_synthetic_manifest(config: SyntheticGenerationConfig) -> List[dict]:
    """Construct a synthetic-image manifest instead of rendering actual images.

    This is a lightweight, deterministic placeholder for the research pipeline.
    """
    manifest: List[dict] = []
    for scene_idx in range(config.scenes):
        image_name = f"synthetic_scene_{scene_idx:04d}.jpg"
        scene = {
            "image_id": scene_idx,
            "image_name": image_name,
            "file_path": str(Path(config.output_dir) / image_name),
            "lighting": config.lighting_variations[scene_idx % len(config.lighting_variations)],
            "background": config.background_types[scene_idx % len(config.background_types)],
            "workers": config.workers_per_scene,
            "synthetic": True,
            "annotations": [],
        }
        for obj_idx in range(config.workers_per_scene):
            scene["annotations"].append({
                "class_name": "worker",
                "bbox": [0.12 + obj_idx * 0.18, 0.18 + (obj_idx % 2) * 0.2, 0.15, 0.2],
                "occluded": obj_idx % 3 == 0,
            })
        manifest.append(scene)
    return manifest


def save_manifest(manifest: List[dict], output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
