> Active training: corrected labels v3. See [annotation review](synthetic_vs_real_cv/reports/annotation_corrections_v3.md). Watch progress with `./scripts/watch_training.ps1`. Completed v2 models remain available; corrected models are added automatically after evaluation.

# Factory worker and PPE detection

**Run on another computer without training:** clone this repository and run `python run_dashboard.py --setup` (`python3` on Linux/macOS). Open http://127.0.0.1:8765. The older real models have been removed; export the replacement models after training before using detection on another computer. See [deployment instructions](DEPLOYMENT.md).

For later launches, run `python run_dashboard.py`. The portable launcher needs no datasets and runs inference only. The [research dashboard guide](synthetic_vs_real_cv/reports/synthreal_dashboard.md) describes the separate local training workspace.

YOLO dataset and research workspace comparing real, AI-generated, and 3D-rendered images.

Classes: **0 worker/person, 1 safety helmet/hard hat, 2 safety/high-visibility vest**.

## Where things belong

- `synthetic_vs_real_cv/data/` — current datasets and annotation manifests.
- `synthetic_vs_real_cv/src/` — data preparation, training, evaluation, and generation code.
- `synthetic_vs_real_cv/configs/` — training and research configurations.
- `synthetic_vs_real_cv/results/` — current annotation audits and future training outputs.
- `synthetic_vs_real_cv/reports/` — methodology and training instructions.
- `synthetic_vs_real_cv/archive/` — retired demo outputs and annotation intermediates.
- `assets/blender/` — worker, helmet, and vest source assets.
- `docs/research/` — research proposal and original repository analysis.
- `weights/` — local pretrained models, including YOLO-World and CLIP.
- `.venv/` and `Ultralytics/` — local Python environment and runtime configuration.

## Current datasets

| Source | Image-label pairs | Unique images selected |
| --- | ---: | ---: |
| Real safety | 500 | 500 |
| AI-generated | 680 | 656 |
| 3D-rendered | 400 | 400 |

The replacement real dataset has fixed 400/50/50 train/validation/test splits. Five comparison experiments use 400 training images each and the same real holdouts. Real annotations are supplied labels remapped to the three project classes; synthetic labels are model-assisted. Neither is exhaustive manually verified ground truth. See [replacement report](synthetic_vs_real_cv/reports/real_dataset_replacement.md).

Start with [the dataset guide](synthetic_vs_real_cv/data/README.md) and [the training guide](synthetic_vs_real_cv/reports/real_3class_training.md). The current entry point is `synthetic_vs_real_cv/src/training/train_real_3class.py`; use `--help` for options. Older prototype and two-class scripts remain for research history and must not be treated as the current three-class training setup.

See [cleanup details](synthetic_vs_real_cv/reports/folder_organization.md).

The next-stage [comparison workflow](synthetic_vs_real_cv/reports/comparison_training.md) prepares five sequential YOLOv8n runs. Its live status is `synthetic_vs_real_cv/results/runs/comparison_v2/queue_status.json`; completed test metrics appear only after each run finishes.
