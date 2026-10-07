> Current dataset replacement: use `configs/comparison_v2.json` and `data/real_safety_500` (400/50/50). Older counts, checkpoints and commands below describe historical runs. See [replacement report](real_dataset_replacement.md).

# Sequential comparison training

Preparation completed on 2026-10-06. All 1,656 selected images decoded and their YOLO annotations passed bounds/class checks. Exact hashes and a perceptual-hash screen (Hamming distance <= 6) found no real cross-split similarity candidates. This screen does not establish that every camera or scene is independent.

## Fixed experiments

| Order | Experiment | Real | AI | 3D | Validation | Test |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Real baseline | 420 | 0 | 0 | 90 real | 90 real |
| 2 | Matched real only | 400 | 0 | 0 | same | same |
| 3 | Matched AI only | 0 | 400 | 0 | same | same |
| 4 | Matched 3D only | 0 | 0 | 400 | same | same |
| 5 | Matched mixture | 200 | 100 | 100 | same | same |

All start independently from the same YOLOv8n pretrained checkpoint. Settings: 50 epochs, 640-pixel input, batch 8, AdamW, initial learning rate 0.001, seed 42, CPU with four computation threads. Early stopping is disabled for matched epoch budgets. This is a preliminary single-seed comparison. The 420-image baseline is reported separately from equal-size comparisons.

Synthetic sources are used exclusively for training. They are never randomly distributed into validation/test sets, so similar synthetic scenes cannot cross those boundaries. AI exact duplicates are excluded through the existing source manifest. Per-source subsets are deterministic, nested where applicable, and contain every class.

Annotations remain model-assisted with targeted visual review, not exhaustive manual ground truth. Report accuracy as agreement with these labels, pending a fully manual holdout audit. Test metrics are computed only after each run, not used for checkpoint selection; validation chooses best.pt.

## Files and monitoring

- `configs/comparison_v1.json`: frozen experiment settings.
- `data/processed/comparison_v1/`: per-experiment lists, YAML files, and hash manifests.
- `results/comparison_preparation/`: inventory, similarity candidates, audit counts, process ID, and logs.
- `results/runs/comparison_v1/queue_status.json`: current run, completed epochs, errors, and pending runs.
- `results/runs/comparison_v1/<experiment>/weights/best.pt`: best validation checkpoint once available.
- `results/runs/comparison_v1/<experiment>/test_metrics.json`: actual test metrics after completion.
- `results/runs/comparison_v1/comparison_metrics.csv`: completed-run comparison, updated sequentially.

The background Python process runs the five jobs one after another and stops on failure. Keep this computer awake. Interruptions require inspecting the last checkpoint and status before resuming; the runner refuses to overwrite an existing queue. No results are populated in advance.

Validate frozen inputs without training (from workspace root):

```powershell
.\.venv\Scripts\python.exe synthetic_vs_real_cv/src/training/run_comparison.py --check-only
```
