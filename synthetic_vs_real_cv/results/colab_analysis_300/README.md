# Blender-only run diagnosis

Source: `/home/puneeth/Downloads/existing_labels_300_seed42_5ea8644b_models_and_metrics/`.

## Verification

All five checkpoint SHA-256 values match their exported metrics. The exported
manifest matches the run specification hash. Its parsed contents equal the local
300-image manifest (JSON formatting differs). Status records five completed runs
on a Tesla T4 with Ultralytics 8.4.173. Blender checkpoint class names are
0 worker, 1 helmet, 2 vest, matching the dataset.

## Observed training behavior

The exported checkpoint contains 50 epochs of scalar training history. These
values were extracted using pickle opcode inspection without loading or executing
the serialized model. The recovered history is in `rendered_training_history.csv`.

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
No confusion matrix or prediction images were included in this compact export.
The original full run folders on Drive should retain those outputs.

Inspect the existing validation prediction panels and confusion matrix before
changing training settings. For a new diagnostic experiment, reserve genuinely
unseen rendered scenes as a rendered holdout to measure whether failure also
occurs within the rendered domain. Do not use this run's training images as an
unseen holdout or tune against the existing real test set. More epochs alone
are not supported as a remedy by the recovered curves. No annotation review,
retraining, or deployment changes were made during this analysis.
