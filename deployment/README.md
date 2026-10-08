# Active dashboard models

This folder contains the **eight completed Google Colab models** used by the dashboard. Both Blender-containing mixtures and all three additional AI/real mixtures are installed.

## Contents

| File or folder | Purpose |
| --- | --- |
| [registry.json](registry.json) | Model IDs, display names, source ratios, checkpoint paths, SHA-256 hashes and saved metrics |
| `models/<experiment_id>.pt` | Best checkpoint for each of the eight experiments |
| `evaluation/<experiment_id>.json` | Original per-model Colab test metric record |
| [evaluation/dataset_manifest.json](evaluation/dataset_manifest.json) | Frozen image memberships and hashes |
| [evaluation/run_specification.json](evaluation/run_specification.json) | Recorded training settings and manifest checksum |
| `evaluation/` remaining metadata | Completion status, environment, initialization and baseline-reuse evidence |

The original checkpoints also remain in [results/colab_run](../results/colab_run/README.md). This deployment copy provides stable paths for inference; the results copy preserves the run's exported layout. All models use 300 training images, 7 shared real validation images and 22 shared real test images.

## Launch or verify

From the repository root on Linux/macOS:

```bash
python3 run_dashboard.py --setup  # First setup on a new computer
python3 run_dashboard.py          # Subsequent launches
python3 run_dashboard.py --check  # Verify hashes, load models and run CPU predictions
```

Open [the dashboard](http://127.0.0.1:8765). On Windows use `python` instead of `python3`. The launcher selects the inference environment automatically; see the [deployment guide](../docs/deployment.md) for setup and portability.

Datasets, Blender and full training histories are not required for inference. The dashboard shows saved Colab scores; launching it does not rerun research evaluation. Those scores are preliminary because the holdouts are small and correlated and labels are incompletely reviewed.

## Import a future completed export

The current eight models are already installed. Run the importer only when intentionally replacing them:

```bash
python3 scripts/import_colab_deployment.py /absolute/path/to/completed_run
```

The argument must be the extracted folder containing the root manifest, run specification, completion status and experiment folders with `weights/best.pt` and `test_metrics.json`. A ZIP or the enclosing download folder is not the input.

Before changing the active deployment, the importer validates completion, the manifest checksum, class mapping, experiment names/source counts, split counts, shared holdouts, finite metrics and checkpoint hashes. It then copies the checkpoints and available evaluation metadata and updates the registry. It does not retrain models or compute fresh test scores. Preserve a copy of a deployment you want to retain before replacing it.
