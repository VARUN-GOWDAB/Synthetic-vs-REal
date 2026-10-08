# Archived v2 models and results — 2026-10-07

Previous 400-image real-only and AI-only models were retired before the new
300-image Colab comparison. This archive preserves both checkpoints, the original
registry with measured metrics and settings, and two historical dashboard
validation reports. No original training-run directories were present locally.

- `deployment/models/`: original `real_only_400.pt` and `ai_only_400.pt`.
- `deployment/registry.json`: original model hashes, evaluation scores, and settings.
- `comparison_metrics.csv`: convenient table derived from that registry.
- `synthetic_vs_real_cv/results/dashboard_tests/`: prior dashboard validation results.
- `manifest.json`: original paths, sizes, and SHA-256 checksums for all five archived files.

All copies were verified before the originals were removed from active locations.
The active deployment registry now has no models. Dataset preparation and
annotation audit records remain in place because the Colab builder depends on them.

## Restore the previous deployment

From the repository root, before adding replacement models, run:

```python
from pathlib import Path
import hashlib, json, shutil
archive = Path('synthetic_vs_real_cv/archive/models_v2_2026-10-07')
assert not json.loads(Path('deployment/registry.json').read_text())['models'], 'Active models already exist'
manifest = json.loads((archive / 'manifest.json').read_text())
files = [r for r in manifest['files'] if r['original_path'].startswith('deployment/')]
for row in files:
    source = archive / row['archive_path']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row['sha256']
for row in files:
    target = Path(row['original_path'])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive / row['archive_path'], target)
```

This restores only the historical inference deployment. The archived scores
remain v2 results and must not be presented as results of the 300-image experiments.
