# Eight-model comparison — 2026-10-08

All eight exported checkpoint hashes match their metrics. The frozen dataset manifest matches the completed run specification and the active deployment metadata. All models use 300 unique training images, identical 7-image real validation and 22-image real test sets, and matching settings and initialization. The five reused models retain their original scores and checkpoint hashes. The exported CSV agrees with all per-model metric files.

## How to read the scores

All table values are percentages, with higher values indicating stronger performance on this holdout. Precision measures how often detections are correct; recall measures how many labeled objects are detected. mAP50 averages class detection performance at an overlap threshold of 0.50; mAP50–95 averages across stricter thresholds from 0.50 to 0.95. The tables are ordered by overall test mAP50–95. Class AP uses the same threshold range for each individual class.

These are saved Colab test scores, not new evaluations performed while writing this report.

## Overall test metrics (%)

| Model | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| 50% real / 25% AI / 25% Blender | 95.63 | 84.55 | 92.40 | 56.34 |
| 75% real / 25% AI | 96.70 | 82.38 | 94.28 | 53.75 |
| 100% real | 91.45 | 87.86 | 93.75 | 52.44 |
| 50% AI / 50% real | 85.46 | 82.47 | 89.67 | 51.77 |
| 50% AI / 25% Blender / 25% real | 94.44 | 80.67 | 92.56 | 49.12 |
| 75% AI / 25% real | 89.38 | 78.91 | 88.39 | 48.81 |
| 100% AI | 77.13 | 62.05 | 67.88 | 29.17 |
| 100% Blender | 5.78 | 10.34 | 3.72 | 0.64 |

## Per-class test AP50–95 (%)

| Model | Worker | Helmet | Vest |
| --- | ---: | ---: | ---: |
| 50% real / 25% AI / 25% Blender | 56.33 | 52.97 | 59.74 |
| 75% real / 25% AI | 52.64 | 56.27 | 52.33 |
| 100% real | 53.32 | 50.65 | 53.35 |
| 50% AI / 50% real | 57.73 | 51.32 | 46.25 |
| 50% AI / 25% Blender / 25% real | 48.48 | 48.99 | 49.90 |
| 75% AI / 25% real | 42.40 | 50.52 | 53.52 |
| 100% AI | 17.35 | 28.07 | 42.10 |
| 100% Blender | 1.87 | 0.01 | 0.05 |

## Findings

- The original 50% real / 25% AI / 25% Blender model retains the highest overall mAP50–95 (56.34%) and vest AP (59.74%). None of the new mixtures surpasses its overall score.
- The new 75% real / 25% AI model has the highest precision (96.70%), mAP50 (94.28%), and helmet AP (56.27%). Its overall mAP50–95 is 53.75%, 1.30 percentage points above real-only and 2.60 below the leading three-source mixture.
- Real-only retains the highest recall (87.86%).
- The 50/50 AI-real model has the highest worker AP (57.73%), but its overall score (51.77%) is below real-only.
- Blender-only remains weak, but the strongest overall mixed model contains Blender. This does not isolate or prove a benefit from Blender: mixture proportions and image membership also differ.

## Interpretation and next step

Treat these as rankings on this specific small holdout, not established deployment performance. The single seed, unreviewed training labels, repeated frames and heuristic scene grouping limit confidence. The new experiments were chosen after observing previous test results, so this test set also no longer serves as a wholly untouched final evaluation.

For the next independent evaluation, compare the leading three-source mixture, the 75% real / 25% AI model, and the real-only baseline on a larger, scene-independent real test set. Choose the final model based on the required balance of missed detections and false detections. Avoid repeatedly tuning against these same 22 test images. This report analyzes the saved results; it does not retrain models or rerun evaluation. All eight completed models are available in the active deployment.

## Files and related guides

- [comparison_metrics.csv](comparison_metrics.csv): eight-model score summary retained with this analysis.
- [verified_results.json](verified_results.json): per-model measurements and checkpoint hashes.
- [Original imported run](../colab_run/README.md): authoritative run metadata, checkpoints and test metric files.
- [Original five-model export](../colab_baselines/README.md): provenance for the five reused results.
- [Deployment guide](../../deployment/README.md): model files used by the dashboard.

For machine processing, prefer the original run's [comparison CSV](../colab_run/comparison_metrics.csv) or per-model `test_metrics.json` files. Historical absolute paths in analysis provenance describe where files were originally verified; they are not required locations for another checkout.
