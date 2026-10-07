# Synthetic vs. Real research workspace

This workspace studies how real, AI-generated, and Blender-rendered training
images affect worker, helmet, and vest detection on shared real-world holdouts.
See [the main README](../README.md) for installation, inference, and test commands.

## Current Colab workflow

Use the [Colab guide](../colab/README.md) and notebook in `../colab/` for the
requested **300 training images per experiment**. This workflow uses existing
labels without further review, preserves known exclusions, and shares 7 real
validation and 22 real test images. Its frozen manifest and selection report are
in `results/colab_preparation_300/`. Source data and old review decisions are
unchanged. These small holdouts and unresolved labels limit interpretation.

## Legacy configuration

`configs/active_dataset.json` selects `comparison_v4.json` and the
`real_safety_500_v4` dataset. The real split is 400 training, 50 validation, and
50 test images. Five planned comparisons use 400 training images each; their
source ratios are defined in the active configuration.

V4 annotations remain partially reviewed. The saved review status records 612
images needing closer review, and `ready_for_training` is false. The training
runner refuses to start until the review is resolved. Bundled inference models
in `../deployment/` come from dataset version 2.

## Directories

- `configs/`: active configuration and historical experiment settings.
- `data/`: source images, YOLO labels, split lists, and annotation manifests.
- `src/`: data preparation, synthetic generation, training, evaluation, analysis,
  prototype utilities, and the dashboard.
- `tests/`: dashboard numeric validation and PPE association/tracking tests.
- `results/`: saved audits, review decisions, and local training outputs.
- `reports/`: methodology, historical training notes, and annotation reviews.
- `experiments/`: original study design notes.
- `notebooks/`: reserved for future exploratory notebooks.
- `archive/`: historical cleanup manifests; archive ZIPs are not tracked.

## Legacy local research workflow

1. Inspect the active configuration and [dataset guide](data/README.md).
2. Restore missing 3D images and finish the [v4 annotation review](reports/full_visual_review_v4.md).
3. Regenerate prepared splits, audit hashes, and paths for the local machine.
4. Supply pretrained weights in `../weights/` and install research dependencies
   from `requirements.txt` in a dedicated environment.
5. Run `python -m src.training.run_comparison --check-only` from this directory
   to validate the prepared inputs before training.
6. Train with `python -m src.training.run_comparison`, evaluate on the fixed real
   test set, and export completed models using `../scripts/export_deployment.py`.

The current checkout contains historical Windows paths and excludes generated
splits and training runs, so these commands require the preparation above.
Prototype and older two-class scripts remain as research history. Historical
reports may describe earlier datasets; the active configuration governs the
current workflow. Source and model-assisted annotations are not exhaustive
manually verified ground truth.
