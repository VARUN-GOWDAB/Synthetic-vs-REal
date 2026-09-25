from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json


def build_yaml(train_dir: str, val_dir: str, test_dir: str, classes: List[str]) -> str:
    lines = [
        f"train: {train_dir}",
        f"val: {val_dir}",
        f"test: {test_dir}",
        "",
        f"nc: {len(classes)}",
        "names: [" + ", ".join(f"'{cls}'" for cls in classes) + "]",
    ]
    return "\n".join(lines) + "\n"


def generate_experiment_yaml(base_dir: str | Path = "data/processed") -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    base = Path(base_dir)
    experiments = [
        "100_real",
        "75_real_25_synthetic",
        "50_real_50_synthetic",
        "25_real_75_synthetic",
        "100_synthetic",
    ]
    classes = ["worker", "helmet"]

    for experiment in experiments:
        train_dir = str(base / experiment / "train" / "images")
        val_dir = str(base / "real" / "val" / "images")
        test_dir = str(base / "real" / "test" / "images")
        yaml_content = build_yaml(train_dir, val_dir, test_dir, classes)
        yaml_path = base / experiment / "data.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(yaml_content, encoding="utf-8")
        results.append({
            "experiment": experiment,
            "yaml_path": str(yaml_path),
            "train_dir": train_dir,
            "val_dir": val_dir,
            "test_dir": test_dir,
        })

    summary_path = base / "experiment_yaml_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    result = generate_experiment_yaml()
    print(json.dumps(result, indent=2))
