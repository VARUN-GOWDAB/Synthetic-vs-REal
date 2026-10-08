# Run SynthReal without training

**Current status:** all eight completed 300-image Colab models are installed in
`deployment/models/`, with metrics and checksums in `deployment/registry.json`.
The full matching run is in [results/colab_run](../results/colab_run/README.md).

On this computer, use `python run_dashboard.py`. The launcher automatically
uses `.venv-inference/` if present, otherwise the installed `.venv/`.
Explicit `--setup` installs into `.venv-inference/`; `--current-env` selects the
Python interpreter running the launcher.

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

Include `run_dashboard.py`, `start_dashboard.ps1`, `requirements-inference.txt`, `deployment/` including its `.pt` files, and `src/dashboard/`. Other research code can remain. Model files in `deployment/models/` are explicitly allowed by `.gitignore`; no Git LFS or separate model download is needed for these small models.

Retired non-Colab model archives are removed. Import completed Colab exports using:

```bash
python3 scripts/import_colab_deployment.py /path/to/models_and_metrics
```

Commit and push the updated deployment directory; another computer can pull and restart. The legacy local-training runner and exporter have been removed.

## Portable behavior

- Uses relative model paths and checksum verification.
- Shows bundled models and saved metrics, not the original machine's training queue.
- Supports image comparisons, webcam/local-video inference, and prototype PPE alerts.
- Uses completed Colab models; local training queues and annotation-review editing are retired.
- Does not need training datasets, Blender assets, training scripts, or run folders.
- Runs on CPU; camera use needs browser permission and compatible hardware.
- Writes only local runtime configuration/cache files as needed.

Launch paths support Windows, Linux, and macOS. Execution has been tested on Windows, including a relocated copy without datasets or training outputs; Linux validation is recorded in [the cleanup report](cleanup_report.md). Dependencies must be available for the target Python/OS. Saved metrics use model-assisted annotations, not exhaustive manually verified ground truth.

The server is also available via `python -m src.dashboard.server` from the repository root. The portable launcher always selects inference-only mode.
