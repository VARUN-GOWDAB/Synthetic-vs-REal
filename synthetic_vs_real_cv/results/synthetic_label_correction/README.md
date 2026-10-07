# Synthetic annotation correction

Classes: 0 worker/person, 1 hard hat/safety helmet, 2 safety/high-visibility vest.

Original annotations: original_labels.zip. All source images preserved; SHA-256 hashes checked. Exact duplicate images are excluded by the source annotation manifests and their labels synchronized.

AI person boxes retain original annotations with model additions. Vest candidates were classified and all 1,674 crops visually screened; ordinary clothing, harnesses and protective suits were removed. Flagged helmet crops were screened for welding masks and hoods. 3D worker boxes were regenerated; original equipment annotations were retained/merged.

These are model-assisted corrections with targeted visual review, not exhaustive manually verified ground truth. Sampled whole-image checks may leave missed or imperfect boxes. Training has not been started.

## ai_generated
{
  "images": 680,
  "training_unique_images": 656,
  "changed_label_files": 680,
  "before_objects": {
    "0": 4592,
    "1": 2015,
    "2": 1595
  },
  "after_objects": {
    "0": 4558,
    "1": 1453,
    "2": 1002
  },
  "empty_labels": []
}

## 3d_rendered
{
  "images": 400,
  "training_unique_images": 400,
  "changed_label_files": 400,
  "before_objects": {
    "0": 2211,
    "1": 448,
    "2": 406
  },
  "after_objects": {
    "0": 489,
    "1": 468,
    "2": 420
  },
  "empty_labels": []
}
