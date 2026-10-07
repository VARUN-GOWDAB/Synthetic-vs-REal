# Synthetic vs. Real: Worker and PPE Detection

A computer vision research project comparing real photographs, AI-generated images,
and Blender renders for YOLO object detection in industrial scenes. The SynthReal
web dashboard supports image comparisons, webcam and local-video inference, and
prototype alerts for missing personal protective equipment (PPE).

The three detection classes are **0: worker/person**, **1: safety helmet/hard hat**,
and **2: safety/high-visibility vest**.

## Train on Google Colab GPU

The current Colab comparison uses **300 unique training images per experiment**
with existing labels, without further annotation review as requested. Known
rejected images and missing files are excluded. Each experiment shares the same
7 real validation images and 22 real test images. The small holdouts and unresolved
training annotations mean the results are preliminary.

| Experiment | Real | AI-generated | 3D-rendered | Total |
| --- | ---: | ---: | ---: | ---: |
| Real only | 300 | 0 | 0 | 300 |
| AI only | 0 | 300 | 0 | 300 |
| Rendered only | 0 | 0 | 300 | 300 |
| AI 50% + rendered 25% + real 25% | 75 | 150 | 75 | 300 |
| Real 50% + AI 25% + rendered 25% | 150 | 75 | 75 | 300 |

1. Upload `colab/synthreal_300.zip` to Google Drive under `MyDrive/SynthReal/`.
2. Upload [Train_SynthReal_on_Colab.ipynb](colab/Train_SynthReal_on_Colab.ipynb)
   to [Google Colab](https://colab.research.google.com/).
3. Select **Runtime → Change runtime type → GPU** and run the cells in order.

Training runs on Colab; checkpoints and metrics are saved to Google Drive.
See the [Colab guide](colab/README.md) for resume instructions and rebuilding
this locally generated ZIP. No training has been run for this comparison yet.

## Quick start

Install Python **3.10 or newer**; Python **3.11 or 3.12** is recommended by the
launcher. Run these commands from the repository root:

```bash
# Linux / macOS
python3 run_dashboard.py --setup
```

```powershell
# Windows
python run_dashboard.py --setup
```

Open [the dashboard](http://127.0.0.1:8765). The first launch creates
`.venv-inference/`, installs the dependencies in `requirements-inference.txt`
(internet required), verifies the bundled checkpoint checksums, and starts the
local server. Training datasets and Blender are not required for inference.

For subsequent launches, omit `--setup`:

```bash
python3 run_dashboard.py
```

Stop the server with **Ctrl+C**. Use `--port 8766` if the default port is occupied.
On Windows, substitute `python` for `python3` in the examples below.

```bash
# Verify bundled checksums and run a small inference check, then exit
python3 run_dashboard.py --check

# Use an existing environment with inference dependencies installed
python3 run_dashboard.py --current-env
```

See [deployment instructions](DEPLOYMENT.md) for more options.

## Included models and legacy research configuration

`deployment/registry.json` includes two evaluated **dataset-version-2** checkpoints:
`real_only_400` and `ai_only_400`. Both are bundled in `deployment/models/` with
saved metrics and SHA-256 checksums. They can be used immediately for inference;
they do not represent results from the current v4 dataset.

The research workspace selects `configs/comparison_v4.json` through
`configs/active_dataset.json`. **The legacy V4 runner is blocked by incomplete annotation
review** (`ready_for_training: false`): the saved review status records 612 images
requiring closer review. The configuration plans five runs, each using 400 training
images and shared real holdouts of 50 validation and 50 test images:

| Experiment | Real | AI-generated | 3D-rendered |
| --- | ---: | ---: | ---: |
| Real only | 400 | 0 | 0 |
| AI only | 0 | 400 | 0 |
| AI 50% + rendered 25% + real 25% | 100 | 200 | 100 |
| Rendered only | 0 | 0 | 400 |
| Real 50% + AI 25% + rendered 25% | 200 | 100 | 100 |

Real labels are remapped source annotations; synthetic labels are model-assisted
with partial manual corrections. Saved scores should be interpreted with these
annotation limitations. See [the v4 review report](synthetic_vs_real_cv/reports/full_visual_review_v4.md).

## Repository layout

```text
.
├── run_dashboard.py             # Portable inference launcher
├── requirements-inference.txt   # Inference dependencies
├── deployment/                  # Bundled checkpoints and metrics
├── scripts/                     # Colab preparation, model export, status tools
├── colab/                       # GPU notebook, upload guide, generated data ZIP
├── assets/blender/              # Source assets for synthetic generation
├── docs/research/               # Proposal and research background
└── synthetic_vs_real_cv/
    ├── configs/                 # Active and historical experiment settings
    ├── data/                    # Images, YOLO labels, split lists, manifests
    ├── src/                     # Dashboard, data, training, evaluation, generation
    ├── tests/                   # Dashboard and PPE tracking tests
    ├── results/                 # Annotation audits and local training outputs
    ├── reports/                 # Methodology, reviews, and workflow notes
    ├── experiments/             # Original study design notes
    └── archive/                 # Historical cleanup manifests
```

## Working with the research code

Use a separate environment for research dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r synthetic_vs_real_cv/requirements.txt
cd synthetic_vs_real_cv
python -m src.training.run_comparison --help
```

On Windows, activate with `.venv\Scripts\Activate.ps1` before changing directories.
The training runner uses the active configuration by default, checks frozen data
hashes and shared holdouts, and refuses training while review is incomplete.

The legacy 400-image workflow is **not a portable training bundle**: saved configurations and
split manifests contain paths from the original Windows machine, prepared
`data/processed/` splits and local `weights/` are excluded from Git, and the
3D source currently has 394 images with 400 labels (six images are missing).
Restore the missing source images, finish review, and regenerate local paths,
prepared splits, and their audit hashes before attempting training. Older
prototype and two-class scripts are historical workflows rather than current
three-class instructions. The separate Colab bundle above supplies portable paths
and the requested 300-image experiments.

The research dashboard can be launched with `python -m src.dashboard.server`
from `synthetic_vs_real_cv/` in the research environment. It reads local run
artifacts; the root launcher selects the bundled models for portable inference.

Run the existing dashboard unit tests from the research directory:

```bash
cd synthetic_vs_real_cv
python -m unittest discover -s tests -v
```

Export additional completed, evaluated models from the repository root using the
research environment:

```bash
python scripts/export_deployment.py
```

The exporter retains existing bundled models and adds completed runs from the
active configuration. Commit the resulting deployment files when sharing them.

## Data and project housekeeping

See [the dataset guide](synthetic_vs_real_cv/data/README.md) for source counts and
format details, and [the research workspace guide](synthetic_vs_real_cv/README.md)
for the current workflow. Preserve datasets, annotations, review decisions,
research documents, Blender assets, and bundled model checkpoints.

Runtime logs, process ID files, Python caches, generated model registries, local
environments, and training outputs are ignored by Git. Historical runtime logs,
process IDs, and the generated model registry were removed during this cleanup;
annotation audits and research history are retained.
