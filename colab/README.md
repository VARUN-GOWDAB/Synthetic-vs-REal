# Train the 300-image comparison on Google Colab

Use `Train_SynthReal_on_Colab.ipynb` with `synthreal_300_ai_real.zip`. The ZIP contains
929 unique images and matching labels: 300 from each source, plus 7 real
validation images and 22 real test images. Files are reused across experiments;
there are **300 distinct training images in each experiment**, with no duplicate
image padding or validation/test images counted as training.

## Already uploaded the original dataset?

Use the small update instead of uploading the full dataset again:

1. Upload `synthreal_experiment_update.zip` (about 211 KiB) into `MyDrive/SynthReal/`.
2. Open `Update_Existing_Dataset_on_Colab.ipynb` in Colab, select GPU, and run all cells.
3. Keep the original `synthreal_300.zip` and completed run folder in Drive.

The notebook reuses `/content/synthreal_5ea8644bd890` if present. After a runtime
reset, it reads the original ZIP from Drive. The five completed checkpoints come
from `MyDrive/SynthReal/runs/existing_labels_300_seed42_5ea8644b/`.
If you moved either item, adjust `OLD_DATA_ZIP` or `PREVIOUS_RUN` in step 3.
Original files remain intact. A new run folder stores the expanded comparison;
only the three new AI/real mixtures train with the default settings.

Regenerate this small update after rebuilding the main bundle with:

```bash
python3 scripts/create_colab_update.py
```

## Start training

1. Create a `SynthReal` folder in Google Drive's **My Drive**.
2. Upload the locally generated `colab/synthreal_300_ai_real.zip` into that folder.
3. Open [Google Colab](https://colab.research.google.com/), select **Upload
   notebook**, and choose `colab/Train_SynthReal_on_Colab.ipynb` from this computer.
4. Select **Runtime → Change runtime type → GPU**, then save the setting.
5. Run the cells from top to bottom and authorize the Drive connection.

No local model training is required. The notebook checks for a GPU and stops if
one is unavailable. Colab GPU availability and session limits vary; see the
[Colab FAQ](https://research.google.com/colaboratory/faq.html).

The defaults are YOLOv8n, 50 epochs, image size 640, batch 8, and seed 42.
The active comparison has eight models:

| Experiment | Real | AI | Blender |
| --- | ---: | ---: | ---: |
| Real only (completed baseline) | 300 | 0 | 0 |
| AI only (completed baseline) | 0 | 300 | 0 |
| Blender only (completed baseline) | 0 | 0 | 300 |
| AI 50% + rendered 25% + real 25% (completed) | 75 | 150 | 75 |
| Real 50% + AI 25% + rendered 25% (completed) | 150 | 75 | 75 |
| AI 50% + real 50% (new) | 150 | 150 | 0 |
| Real 75% + AI 25% (new) | 225 | 75 | 0 |
| AI 75% + real 25% (new) | 75 | 225 | 0 |

Both original Blender-containing mixtures are included in the current
comparison. All five previously completed results are also preserved in
`synthetic_vs_real_cv/archive/comparison_300_blender_mixtures_2026-10-07/`.
The five completed checkpoints and metrics are included in the dataset ZIP;
only the three new mixtures train by default. The helper verifies identical
baseline memberships, per-image hashes, settings, initialization, and checkpoint
hashes before copying them. Reused metrics are never described as new evaluations.
All eight models use the same training settings and real holdouts.

## Results and resuming

Checkpoints, plots, metrics, and the dataset manifest are stored under:

```text
MyDrive/SynthReal/runs/ai_real_mixtures_300_seed42_<archive hash>/
```

Use the updated notebook/ZIP pair, not the old `synthreal_300.zip`. The new run
name keeps previous results separate. Do not resume the old five-model folder.
Imported completed models contain `best.pt` and metrics only; the new mixture runs also
save `last.pt`, training history, and plots.

After a disconnect, reconnect to a GPU and rerun the cells. Completed experiments
are skipped; unfinished experiments resume from their saved `weights/last.pt`.
At least one completed epoch must have saved a checkpoint. If a session stops
before that first checkpoint, use a new run name. Periodic epoch checkpoints are
also retained. Training follows the [Ultralytics resume workflow](https://docs.ultralytics.com/modes/train/).

Do not change the dataset, epochs, batch size, or other settings while resuming.
To run different settings, set a new `RUN_NAME`. If batch 8 exceeds GPU memory,
use batch 4 in a new run folder and set `REUSE_BASELINES = False` to retrain all
eight experiments under the new settings. Any setting change requires disabling
baseline reuse. Keep `REUSE_BASELINES = True` with the supplied defaults.

The final cells display `comparison_metrics.csv` and create a ZIP of the eight
best models and metrics in Drive. Test metrics are computed after training;
validation is used for model selection. The new mixtures have not been trained yet. The five existing models reuse their original
completed Colab runs; their provenance is saved in `reused_baselines.json`.

## Label policy and limitations

Further annotation review was skipped at the user's request. Existing labels
are packaged unchanged. Missing files and previously rejected images are
excluded. Of the 929 selected images, 205 have an additional visual-screen
record, 196 inherit an earlier approved review, and 528 are not visually
approved. File checks and valid YOLO coordinates do not establish label accuracy.

The 7 validation and 22 test images retain their original split assignments.
Known exclusions and recorded scene groups are respected. Original scene groups
are heuristic, so visual overlap in the expanded, unreviewed pool may remain.
Repeated frames within the small holdouts also limit statistical confidence.
Treat results as preliminary and do not compare directly with bundled v2 scores.
The full original v4 review remains incomplete.

## Rebuild locally

The ZIP is generated and ignored by Git; it must be rebuilt after cloning.
The archived completed export supplies the five completed checkpoints for reuse. In a
Python environment with Pillow installed, run from the repository root:

```bash
python3 scripts/prepare_colab.py
python3 scripts/create_colab_notebook.py
python3 -m unittest discover -s tests -v
```

Defaults are 300 images and `--review-policy existing_labels`. The manifest,
per-image label provenance, source ratios, checksums, and complete exclusion list
are saved in `synthetic_vs_real_cv/results/colab_preparation_300/`.
The original 100-image reviewed selection remains as historical audit data in
`results/annotation_review_v5/`; it is not the current training configuration.

Rebuilding the ZIP changes its checksum. Always regenerate the notebook and
upload the matching ZIP/notebook pair. The notebook verifies the ZIP checksum,
then every image and label checksum, readability, label coordinates, class IDs,
unique memberships, source counts, shared holdouts, and known scene groups.
