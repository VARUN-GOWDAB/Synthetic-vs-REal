# Train the 300-image comparison on Google Colab

Use `Train_SynthReal_on_Colab.ipynb` with `synthreal_300.zip`. The ZIP contains
929 unique images and matching labels: 300 from each source, plus 7 real
validation images and 22 real test images. Files are reused across experiments;
there are **300 distinct training images in each experiment**, with no duplicate
image padding or validation/test images counted as training.

## Start training

1. Create a `SynthReal` folder in Google Drive's **My Drive**.
2. Upload the locally generated `colab/synthreal_300.zip` into that folder.
3. Open [Google Colab](https://colab.research.google.com/), select **Upload
   notebook**, and choose `colab/Train_SynthReal_on_Colab.ipynb` from this computer.
4. Select **Runtime → Change runtime type → GPU**, then save the setting.
5. Run the cells from top to bottom and authorize the Drive connection.

No local model training is required. The notebook checks for a GPU and stops if
one is unavailable. Colab GPU availability and session limits vary; see the
[Colab FAQ](https://research.google.com/colaboratory/faq.html).

The defaults are YOLOv8n, 50 epochs, image size 640, batch 8, and seed 42.
All five experiments start from the same pretrained checkpoint and use the same
settings and real holdouts. The mixtures contain 150/75/75 images; see the root
README for the source breakdown.

## Results and resuming

Checkpoints, plots, metrics, and the dataset manifest are stored under:

```text
MyDrive/SynthReal/runs/existing_labels_300_seed42_<archive hash>/
```

After a disconnect, reconnect to a GPU and rerun the cells. Completed experiments
are skipped; unfinished experiments resume from their saved `weights/last.pt`.
At least one completed epoch must have saved a checkpoint. If a session stops
before that first checkpoint, use a new run name. Periodic epoch checkpoints are
also retained. Training follows the [Ultralytics resume workflow](https://docs.ultralytics.com/modes/train/).

Do not change the dataset, epochs, batch size, or other settings while resuming.
To run different settings, set a new `RUN_NAME`. If batch 8 exceeds GPU memory,
use batch 4 in a new run folder for all five experiments.

The final cells display `comparison_metrics.csv` and create a ZIP of the five
best models and metrics in Drive. Test metrics are computed after training;
validation is used for model selection. Training has not been executed as part
of preparing this notebook.

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

The ZIP is generated and ignored by Git; it must be rebuilt after cloning. In a
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
