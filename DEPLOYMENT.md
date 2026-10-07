# Run SynthReal without training

Clone the repository and enter its directory. Install Python 3.11 or 3.12, then run:

```powershell
python run_dashboard.py --setup
```

On Linux/macOS use `python3` instead. Open **http://127.0.0.1:8765**.

The first launch creates `.venv-inference`, installs dependencies (internet required), verifies bundled model checksums, and starts inference. No training or model downloads run. For later launches omit `--setup`. Stop with Ctrl+C. Use `--port 8766` if the default port is occupied.

To verify model loading and inference without starting the server:

```powershell
python run_dashboard.py --check
```

## What to push

Include `run_dashboard.py`, `start_dashboard.ps1`, `requirements-inference.txt`, `deployment/` including its `.pt` files, and `synthetic_vs_real_cv/src/dashboard/`. Other research code can remain. Model files in `deployment/models/` are explicitly allowed by `.gitignore`; no Git LFS or separate model download is needed for these small models.

The older real dataset and its models have been retired. The bundle is currently empty until replacement training and evaluation finish. The dashboard can open, but detection requires exporting at least one completed replacement model. Refresh the bundle on the training computer:

```powershell
.\.venv\Scripts\python.exe scripts/export_deployment.py
```

Commit and push the updated deployment directory; the other computer can pull and restart. Exporting does not stop, start, or resume training. Only completed, evaluated models are exported.

## Portable behavior

- Uses relative model paths and checksum verification.
- Shows bundled models and saved metrics, not the original machine's training queue.
- Supports image comparisons, webcam/local-video inference, and prototype PPE alerts.
- Hides annotation review because datasets are not distributed.
- Does not need training datasets, Blender assets, training scripts, or run folders.
- Runs on CPU; camera use needs browser permission and compatible hardware.
- Writes only local runtime configuration/cache files as needed.

Launch paths support Windows, Linux, and macOS. Execution has been tested on Windows, including a relocated copy without datasets or training outputs; other operating systems have not been tested here. Dependencies must be available for the target Python/OS. Saved metrics use model-assisted annotations, not exhaustive manually verified ground truth.

The original research server remains available via `python -m src.dashboard.server` from `synthetic_vs_real_cv`. The portable launcher always selects inference-only mode.
