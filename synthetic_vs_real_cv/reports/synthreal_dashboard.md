> Current dataset replacement: use `configs/comparison_v2.json` and `data/real_safety_500` (400/50/50). Older counts, checkpoints and commands below describe historical runs. See [replacement report](real_dataset_replacement.md).

# SynthReal PPE Detection

## Open the dashboard

Visit **http://127.0.0.1:8765** on this computer. The local server is running separately from training.
To start it again from the workspace root, run `./start_dashboard.ps1` in PowerShell.
Alternatively, from `synthetic_vs_real_cv`, run `../.venv/Scripts/python.exe -m src.dashboard.server`.
The launcher does not start or duplicate training. Closing a browser tab does not stop training.

## Features

- **Overview:** actual queue progress, checkpoint availability, source inventory, and human test-review progress.
- **Experiments:** A/B/C/D plus the 420-image baseline and Blender-only comparison. Tables show source ratios, precision, recall, mAP50, and mAP50–95 from completed test evaluations only. A chart compares test mAP50–95; details include settings, class AP, paths, and annotation status.
- **Image comparison:** upload JPG, PNG, or WebP, select available checkpoints, and compare annotated results and pixel boxes side by side. Models run sequentially to limit RAM use. No accuracy is inferred from confidence.
- **Live detection:** choose one checkpoint and enable a webcam or a local video file. Change model, confidence, NMS threshold, and PPE requirements. Stop releases the camera. Frames are processed locally and are not saved by the dashboard. CPU inference can skip video frames.
- **PPE alerts:** greedy IoU person tracking, exclusive spatial helmet/vest association, minimum three consecutive observations, and configurable elapsed persistence. Missing person observations reset the streak; tiny/edge-clipped people suppress alerts. The tracker expires after five seconds without observations. Occlusion, crowded scenes, and identity changes still need validation. Alerts mean *possible missing PPE*, not verified noncompliance. Sound is opt-in; events remain in the browser session.
- **Test-set review:** inspect all 90 test images with existing boxes, mark reviewed after explicit confirmation, or flag corrections. Decisions are hash-bound and stored separately. The tool does not silently edit frozen labels or assert that model-assisted annotations are manual ground truth.

## Experiment alignment

| Experiment | AI | Blender | Real | Training images |
| --- | ---: | ---: | ---: | ---: |
| A | 100% | 0% | 0% | 400 |
| B | 50% | 25% | 25% | 400 |
| C | 25% | 25% | 50% | 400 |
| D | 0% | 0% | 100% | 400 |
| Blender reference | 0% | 100% | 0% | 400 |
| Real baseline | 0% | 0% | 100% | 420 |

Architecture remains YOLOv8n throughout. All use the same 90 real validation and 90 real test images. Experiment B uses deterministic subsets of the existing source pools and is prepared in `configs/comparison_b.json`. A separate continuation process waits for the original five-run queue to finish before starting B. It does not compete with that queue for training resources. Its waiting state is `results/comparison_preparation/experiment_b_queue.json`.

## Registry and files

- `src/dashboard/`: local HTTP server, registry, PPE rules, and browser assets.
- `results/model_registry.json`: refreshed registry derived from actual run artifacts; includes training settings, source ratios, checkpoint paths, and available test metrics.
- `results/runs/comparison_v1/`: original five experiment outputs.
- `results/runs/comparison_b/`: experiment B output once its turn starts.
- `results/test_review/decisions.json`: user review decisions, created on the first saved review.
- `results/dashboard_cache/selected_model.pt`: one local checkpoint snapshot used for inference.
- `results/comparison_preparation/dashboard_*.log`: background server logs.
- `results/dashboard_tests/`: local smoke-test artifacts and a dashboard screenshot.

Available training checkpoints are explicitly marked as previews. The selected snapshot remains loaded until switching models or using the dashboard's reload control; it is not silently swapped mid-feed. Completed comparison metrics remain empty until evaluation produces them.

## Verification and remaining work

Eight rule tests cover PPE association, persistence, resets, clipping, requirement changes, and numeric validation. API smoke checks cover six registry entries, review-image loading, confirmation enforcement, and invalid requests. Uploaded-image inference and local-video inference were exercised through the browser with a real checkpoint. No browser console errors were observed during those checks. Physical webcam permission and hardware operation have not been exercised; enable them from Live detection when ready.

Training remains in progress. Full manually reviewed test ground truth, completed-model comparisons, crowded-scene tracking validation, and operational alarm validation remain necessary. Export is optional; the dashboard loads `.pt` checkpoints directly. Keep the computer awake for the current training and experiment-B continuation processes. Restarting the machine will stop those processes; inspect saved queue status and checkpoints before restarting training.
