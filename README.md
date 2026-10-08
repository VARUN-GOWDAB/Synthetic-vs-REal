# Synthetic vs. Real: Worker and PPE Detection

This college research project compares real photographs, AI-generated images and Blender renders for detecting workers and personal protective equipment with YOLOv8n. The three classes are **0 worker/person, 1 helmet/hard hat, and 2 safety/high-visibility vest**.

**Current status:** all eight 300-image experiments have finished in Google Colab. Their trained models are installed locally, the datasets are already available, and no dataset preparation or retraining is needed to use the dashboard. The dashboard supports image comparisons, webcam/local-video inference and prototype PPE alerts.

## Run the dashboard

Run these commands from the repository root. The launcher requires Python 3.10 or newer and recommends 3.11 or 3.12.

```bash
# First setup on a new computer: install the inference dependencies and launch
python3 run_dashboard.py --setup

# Later launches, including this computer's existing environment
python3 run_dashboard.py

# Check all eight checkpoint hashes, model loading and small CPU predictions
python3 run_dashboard.py --check
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Stop the server with Ctrl+C. If the port is occupied, launch with `--port 8766` and open that port instead. On Windows, replace `python3` with `python`; the [PowerShell launcher](start_dashboard.ps1) is also available.

The launcher selects `.venv-inference/` when available, otherwise the existing `.venv/`. `--setup` installs the [inference requirements](requirements-inference.txt) into a dedicated environment; `--current-env` uses your current Python interpreter. Model inference needs the deployment files, but does not need the datasets or Blender. See the [deployment guide](docs/deployment.md) for portability details.

## Understand the completed study

Every model used 300 distinct training images, the same **7 real validation images and 22 real test images**, YOLOv8n, 50 epochs, image size 640, batch 8 and seed 42. Validation selects the checkpoint; the saved test metrics report its performance on the real test set.

| Completed experiment | Real images | AI images | Blender images |
| --- | ---: | ---: | ---: |
| Real only | 300 | 0 | 0 |
| AI only | 0 | 300 | 0 |
| Blender only | 0 | 0 | 300 |
| AI 50% / Blender 25% / real 25% | 75 | 150 | 75 |
| Real 50% / AI 25% / Blender 25% | 150 | 75 | 75 |
| AI 50% / real 50% | 150 | 150 | 0 |
| Real 75% / AI 25% | 225 | 75 | 0 |
| AI 75% / real 25% | 75 | 225 | 0 |

Start with the [eight-model comparison](results/colab_analysis_8_models/README.md) for scores and interpretation. The highest recorded overall test mAP50–95 is **56.34%**, from the 50% real / 25% AI / 25% Blender model. This is a preliminary result: labels are not exhaustively reviewed, holdouts are small and correlated, and the three additional mixtures were selected after observing earlier test scores. The test set is therefore no longer an untouched final evaluation.

## Find the files you need

```text
.
├── assets/blender/       # Optional source assets for synthetic scene generation
├── colab/                # Existing notebooks and locally stored upload ZIPs
├── data/                 # Local datasets; excluded from Git (ZIPs in Drive)
├── deployment/           # Eight active models, registry and evaluation evidence
├── docs/                 # Deployment and cleanup guides
│   └── research/         # Proposal and research workflow documentation
├── results/              # Completed Colab outputs and analyses only
│   ├── colab_run/        # Imported eight-model run
│   ├── colab_baselines/  # Original five-model export reused in the expanded run
│   ├── colab_analysis_8_models/
│   └── colab_analysis_300/ # Blender-only training diagnosis
├── scripts/              # Colab helpers and completed-export importer
├── src/
│   ├── dashboard/        # Server, PPE tracking and static frontend
│   ├── evaluation/       # Evaluation utilities
│   ├── synthetic/        # Optional Blender generation code
│   └── analysis/         # Research analysis utilities
├── tests/                # Software integrity and behavior checks
├── run_dashboard.py
├── start_dashboard.ps1
├── requirements.txt      # Broader research dependencies
├── requirements-inference.txt
├── requirements-colab.txt
├── README.md
└── .gitignore
```

| What you want to do | Read this |
| --- | --- |
| Understand the existing datasets and class labels | [Data guide](data/README.md) |
| Inspect scores, checkpoints or training plots | [Results guide](results/README.md) |
| Understand the dashboard's model files | [Deployment contents](deployment/README.md) |
| Reproduce or resume training in Colab | [Colab guide](colab/README.md) |
| Inspect optional Blender assets | [Asset guide](assets/blender/README.md) |
| Review the reorganization and validation | [Cleanup report](docs/cleanup_report.md) |

## Optional: reproduce in Colab or import another run

Use the existing matching notebook/ZIP pair described in the [Colab guide](colab/README.md). The notebooks validate the frozen data and resolve paths at runtime. Local dataset preparation and bundle generators have been retired. Dataset folders and ZIP bundles are excluded from Git because the datasets are stored in Google Drive. A normal clone includes the data guide and frozen evaluation manifests, but not the dataset images or labels. Use the matching ZIP already in Drive for Colab; download it only if you need a local copy.

To deliberately replace the active deployment with another completed export:

```bash
python3 scripts/import_colab_deployment.py /absolute/path/to/completed_run
```

Pass the extracted run folder containing `dataset_manifest.json`, `run_specification.json`, `status.json` and the experiment folders, rather than its enclosing download folder or ZIP. The importer checks completion, manifest/checkpoint hashes, split counts, shared holdouts and metric consistency before installing models. See the [deployment README](deployment/README.md) for details.

## Software checks and generated files

In a Python environment with the inference dependencies installed:

```bash
python3 -m unittest discover -s tests -v
```

These are software tests for data integrity, resume/baseline reuse, launcher selection and PPE behavior. They do not train models or produce a replacement research evaluation.

Both the launcher and `python -m src.dashboard.server` read [deployment/registry.json](deployment/registry.json). Runtime model cache files live under `.runtime/`. Best Colab checkpoints, metrics and plots are allowed by Git; dataset folders, virtual environments, caches, ZIP bundles and intermediate/last/initial checkpoints are ignored. Ignored files can exist locally without being included in a GitHub checkout.
