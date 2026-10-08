# Imported eight-model Colab run

This is the completed run **`expanded_8_models_300_seed42_bcf65166`**, imported from the supplied Google Drive download. All **132 original files** were copied byte-for-byte; this README is additional documentation. The original download was left unchanged.

The [status record](status.json) marks all eight experiments complete. The manifest, settings, per-model test metrics and best-checkpoint hashes match the [active deployment](../../deployment/README.md).

## Available files

| Location | Contents |
| --- | --- |
| [dataset_manifest.json](dataset_manifest.json) | Frozen memberships, class mapping and image/label hashes |
| [run_specification.json](run_specification.json) | Training settings and manifest checksum |
| [comparison_metrics.csv](comparison_metrics.csv) | Saved comparison scores for all eight models |
| [reused_baselines.json](reused_baselines.json) | Provenance for the five reused completed models |
| Root environment, initialization and status files | Execution and completion metadata |
| `<experiment_id>/weights/best.pt` | Best checkpoint for each of the eight models |
| `<experiment_id>/test_metrics.json` | Its recorded real-test evaluation |
| Three additional AI/real experiment folders | Training CSV, arguments, curves, confusion matrices, previews and periodic/last checkpoints |
| Three corresponding `<experiment_id>_test/` folders | Test curves, confusion matrices and labeled/predicted batches |
| `initial_weights/` | Original YOLOv8n initialization checkpoint |

The detailed training and test plots are available for `mixed_ai50_real50_300_existing_labels`, `mixed_real75_ai25_300_existing_labels` and `mixed_ai75_real25_300_existing_labels`. The five reused models have their best checkpoints and test metric records here; their full earlier histories and plots are not in this download. Their reuse hashes and original settings are retained in [reused_baselines.json](reused_baselines.json); the original compact export remains in Drive.

## Understand the run

Each experiment used 300 distinct training images, the same 7 real validation images and 22 real test images, YOLOv8n, 50 epochs, image size 640, batch 8 and seed 42. Validation selected `best.pt`; the saved test scores refer to that checkpoint. Training-folder plots should not be confused with plots in the separate `_test/` folders.

Original `/content/` paths in arguments describe the Colab environment. They are historical evidence, not paths to fix for this local checkout. No retraining or reevaluation was performed during import. See the [eight-model report](../README.md) for scores and limits, including incomplete label review and the small correlated holdouts.

Best checkpoints and plots are allowed by Git. Periodic/last checkpoints and the pretrained initialization are preserved locally but ignored. Keep the complete downloaded archive separately if you need every resume artifact on another computer.
