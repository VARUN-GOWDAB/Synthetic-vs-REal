from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_runs(run_file: str | Path) -> List[Dict[str, Any]]:
    with Path(run_file).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_run_directories(run_file: str | Path, base_dir: str | Path = "experiments") -> List[Dict[str, Any]]:
    runs = load_runs(run_file)
    base = Path(base_dir)

    created: List[Dict[str, Any]] = []
    for run in runs:
        run_dir = base / run["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "name": run["name"],
            "real_ratio": run["real_ratio"],
            "synthetic_ratio": run["synthetic_ratio"],
            "real_count": run["real_count"],
            "synthetic_count": run["synthetic_count"],
            "train_path": run["train_path"],
            "val_path": run["val_path"],
            "test_path": run["test_path"],
            "status": "prepared",
        }
        manifest_path = run_dir / "run_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        created.append(metadata)

    return created


if __name__ == "__main__":
    created = prepare_run_directories("data/processed/experiment_runs.json")
    print(json.dumps(created, indent=2))
