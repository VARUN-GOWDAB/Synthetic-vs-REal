> Current dataset replacement: use `configs/comparison_v2.json` and `data/real_safety_500` (400/50/50). Older counts, checkpoints and commands below describe historical runs. See [replacement report](real_dataset_replacement.md).

# Worker, helmet and vest training

The current dataset is `data/real_safety_600`: 600 images and matching YOLO labels,
selected by visual contact-sheet screening of 1,240 ranked safety candidates.
Class IDs: 0 worker/person, 1 safety helmet/hard hat, 2 safety/high-visibility vest.

There are 1,108 worker boxes, 1,052 helmet boxes and 149 vest boxes. Vests occur in
94 images. Splits: 420 training, 90 validation and 90 test images, stratified by vest
presence, seed 42. Every split contains all three classes.

Worker and vest labels originated from YOLO-World predictions. Source SHWD class 0
was an unhelmeted head and was replaced by whole-person predictions. Class 1 retains
SHWD box extents, which can include the face. Screening rejected obvious missed
objects, classroom images, unsuitable graphics and repeated scenes. Sixty-one false
vest boxes were removed from 52 images. This is contact-sheet screening, not exhaustive
manual ground truth; independently review test labels before reporting accuracy.
Exact duplicates were removed and near duplicates filtered by dHash, but scene-level
separation is not guaranteed.

The user requested deletion of the other real-data images after the 600-image copy
was verified. The old source, draft, sample folders and obsolete three-class split
are removed by the cleanup step. Audit records remain in `results/curation_600`.
Historical prototype experiments and synthetic data are separate from this dataset.

## Validate configuration without training

From the repository root in PowerShell:

```powershell
.venv/Scripts/python.exe synthetic_vs_real_cv/src/training/train_real_3class.py --prepare-only
```

The script defaults to the new dataset and reuses its prepared `data.yaml`.

## Train after accepting the annotations

```powershell
.venv/Scripts/python.exe synthetic_vs_real_cv/src/training/train_real_3class.py --annotations-reviewed --device cpu
```

Training has not started. The user's earlier choice was to review automatic labels
before training. Do not bypass this with `--allow-pseudo-labels` for this workflow.

Defaults: YOLOv8n, 640 pixels, 50 epochs, batch 16, AdamW, lr0=0.01,
patience 20, seed 42, workers 0. Results and weights go under `results/runs`.
The best validation checkpoint is evaluated on the held-out test split.

For real-versus-synthetic comparisons, use identical class definitions, starting
weights, hyperparameters and the same independently reviewed real test set.
Fine-tuning synthetic-trained weights is a separate experiment. Scores measured
against model-generated labels are not independent ground-truth scores.

Old annotation/ranking scripts are retained as provenance; their original inputs
are intentionally deleted and they cannot regenerate the discarded data.
