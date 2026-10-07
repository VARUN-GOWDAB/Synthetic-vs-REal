# Datasets

- `real_safety_500/`: 500 curated real image-label pairs and fixed train/val/test lists.
- `ai_generated/`: 680 image-label pairs; `annotation_manifest.json` selects 656 unique images.
- `3d_rendered/`: 400 image-label pairs; all selected by its manifest.
- `incoming/`: place new, unreviewed deliveries here in separate source folders (created when needed).
- `processed/`: reserved for newly generated experiment splits (created by preparation scripts).

Every active source uses `images/` and `labels/`, with matching stems and YOLO normalized boxes. Class IDs are 0 worker, 1 helmet, 2 vest. Honor each annotation manifest when preparing training inputs. Keep the real test set out of training; create source-aware research splits before mixed-data experiments.

Old sample/demo processed outputs were archived. Existing legacy preparation scripts still target the older data/real and data/synthetic layout; they require adaptation before use with these sources. Do not run them expecting the current datasets to be selected automatically.

Annotation audit locations: `../results/real_ppe_v2_import/` and `../results/synthetic_label_correction/`.

Active real split: 400 train, 50 validation, 50 test. Source labels: Person 6 -> worker 0; Helmet 0 -> helmet 1; Vest 2 -> vest 2. Other labels are discarded. See ../reports/real_dataset_replacement.md.
