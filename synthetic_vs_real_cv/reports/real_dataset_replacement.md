# Replacement real dataset — October 6, 2026

Selected 500 images from the 1,416 supplied images in C:/Users/SIC/Downloads/archive/data. The original download is unchanged. Active dataset: data/real_safety_500.

## Labels

| Supplied ID | Supplied class | Project ID | Project class |
| --- | --- | --- | --- |
| 6 | Person | 0 | worker |
| 0 | Helmet | 1 | helmet |
| 2 | Vest | 2 | vest |

All other annotation classes are discarded; their objects may remain visible as background. Ten orphan source label files have no matching image and were not used.

## Selection and splits

Selection ranks exposure, sharpness and noise, requires a person plus helmet or vest, excludes crowded images and the unrelated examples appended at source image1120 and above, and distributes training examples across similarity groups. CLIP similarity and perceptual hashes group similar examples before splitting. This is a heuristic quality selection, not a claim that every annotation is correct. Rotated/augmented source images remain. Not every selected image contains all three classes.

400 training, 50 validation, 50 test. Every experiment uses the same frozen validation/test lists. Filenames retain their ORIGINAL split prefix; the train.txt/val.txt/test.txt lists define current membership. No exact image duplicates or perceptual-hash cross-split candidates (Hamming <=6) remain. This screen does not prove scene/camera independence. Test annotations still need exhaustive manual review, and quality-selected holdouts do not represent all factory conditions.

Object counts (worker=0, helmet=1, vest=2):

{
  "train": {
    "1": 738,
    "0": 811,
    "2": 663
  },
  "val": {
    "1": 107,
    "2": 106,
    "0": 105
  },
  "test": {
    "2": 93,
    "0": 88,
    "1": 97
  }
}

## Replacement and training

Removed old real_safety_600, old comparison_v1 real baseline/real-only runs and checkpoints, their deployment copies, and cached model copies. Removed the intermediate full import; only the chosen 500 remain in the active real dataset. AI-generated and Blender datasets are retained. The unfinished old AI run is retained under results/retired/comparison_v1 but is not active or used for comparison.

configs/active_dataset.json selects comparison_v2. Five fresh YOLOv8n experiments use 400 training images each, seed 42, 50 epochs, 640 pixels, batch 8 and identical pretrained weights. Order: real only, AI only, 50/25/25 mixture, Blender only, 25/25/50 mixture (ratios AI/Blender/Real). Results from the old test set must not be compared directly with these new results.

From the repository root, validation command:

```powershell
.\.venv\Scripts\python.exe synthetic_vs_real_cv/src/training/run_comparison.py --check-only
```

Queue status: results/runs/comparison_v2/queue_status.json. Training log: results/comparison_preparation_v2/training_stdout.log. Dashboard: http://127.0.0.1:8765. Completed test metrics appear after each 50-epoch run and its evaluation. The new training has not established accuracy yet.

Old deployment models are removed. Once replacement models finish, run `.\.venv\Scripts\python.exe scripts/export_deployment.py` at repository root and push deployment/ with the code. Other systems can then run inference without datasets or training. Until export, the portable bundle contains no models.
