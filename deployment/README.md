# Portable model bundle

Commit this directory, including `models/*.pt` and `registry.json`. Checkpoints are small enough for normal Git. The registry uses relative paths, saved evaluation metrics, and SHA-256 checksums.

Only completed/evaluated runs are included. Refresh after more training runs finish with `python scripts/export_deployment.py` using the training environment, then commit and push these files. Incomplete checkpoints are deliberately excluded.

See [deployment instructions](../DEPLOYMENT.md) for setup and launch commands.
The current registry includes `real_only_400` and `ai_only_400`, both evaluated
with dataset version 2. The active v4 annotation review is incomplete; these
checkpoints are earlier results and are available for inference now.
