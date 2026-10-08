# Results and annotation audits

- `curation_600/`: real-data selection, repair, and validation records.
- `synthetic_label_correction/`: final synthetic audit, image/label hashes, corrected overview sheets, original-label backup, and historical correction scripts.
- `runs/`: reserved for actual YOLO training/evaluation outputs.

Old prototype metrics, sample review images, and intermediate correction artifacts were moved into `../archive/cleanup_2026-10-06.zip`. No new model training or measured research results were produced during cleanup.

The previous v2 model scores and dashboard validation reports are now in
[the model archive](../archive/models_v2_2026-10-07/README.md). Annotation audits
and `colab_preparation_300/` stay here as inputs to the current Colab workflow.
