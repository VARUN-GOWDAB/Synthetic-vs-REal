# Blender-only run diagnosis

This report examines the Blender-only model from the original five-model Colab comparison, preserved in the [expanded run](../../results/colab_run/README.md). That same checkpoint and its recorded test scores are included in the [completed eight-model comparison](../../results/README.md).

## Verification

All five checkpoint SHA-256 values match their exported metrics. The exported
manifest matches the run specification hash. Its parsed contents match the original five-model comparison manifest used during verification (JSON formatting differs). This is the original baseline manifest, rather than the expanded eight-model manifest. Status records five completed runs
on a Tesla T4 with Ultralytics 8.4.173. Blender checkpoint class names are
0 worker, 1 helmet, 2 vest, matching the dataset.

## Observed training behavior

The exported checkpoint contains 50 epochs of scalar training history. These
values were extracted using pickle opcode inspection without loading or executing
the serialized model. The recovered history is in [rendered_training_history.csv](rendered_training_history.csv). Epoch numbers below use the 1-based convention.

- Training box loss: 1.31711 → 0.50326.
- Training classification loss: 2.21619 → 0.40760.
- Real validation box loss: 3.72311 → 3.80739.
- Real validation classification loss: 4.16963 → 4.37014.
- Best validation mAP50–95: 2.727%, at epoch 4.
- Final validation mAP50–95: 0.045%.
- Exported best-checkpoint test mAP50–95: 0.6447%.

The training labels became easier for the model to fit, but that learning did not
transfer well to the real validation/test images. This is consistent with a
render-to-real distribution mismatch and/or differences or errors in labels.
The metrics do not identify the relative contribution of those causes. Training
loss is measured under augmentation, so it is not itself a rendered holdout score.

## Test performance by class

| Class | AP50–95 |
| --- | ---: |
| Worker | 1.8709% |
| Helmet | 0.0121% |
| Vest | 0.0511% |

The failure affects all three classes, with particularly poor PPE detection.

## Limits and next diagnostic

There are only 7 real validation and 22 real test images, with correlated frames.
Blender-only confusion matrices and prediction panels are absent from the original compact export and the imported expanded run. If the original Blender run folder is still available on Drive, check it for those artifacts.

If recovered from Drive, inspect the Blender validation prediction panels and confusion matrix before changing training settings. For a new diagnostic experiment, reserve genuinely
unseen rendered scenes as a rendered holdout to measure whether failure also
occurs within the rendered domain. Do not use this run's training images as an
unseen holdout or tune against the existing real test set. More epochs alone
are not supported as a remedy by the recovered curves. No annotation review,
retraining, or deployment changes were made during this analysis.

## Related files

The [original baseline metrics](../../results/colab_run/rendered_only_300_existing_labels/test_metrics.json) contain the test values, while the recovered CSV contains training/validation history. The [results index](../../results/README.md) contains the comparison tables and explains the raw run artifacts. This diagnosis does not require dataset preparation or another training run.
