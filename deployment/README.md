# Portable model bundle

Commit this directory, including `models/*.pt` and `registry.json`. Checkpoints are small enough for normal Git. The registry uses relative paths, saved evaluation metrics, and SHA-256 checksums.

Only completed/evaluated runs are included. Refresh after more training runs finish with `python scripts/export_deployment.py` using the training environment, then commit and push these files. Incomplete checkpoints are deliberately excluded.

See ../DEPLOYMENT.md for setup and launch instructions. The older real models were removed when the dataset was replaced. Export completed comparison_v2 models before using detection on another system. The registry is empty until then.
