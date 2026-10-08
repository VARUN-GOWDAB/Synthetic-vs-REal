# Active model deployment

Eight completed 300-image Colab models are installed in `models/` and registered
in `registry.json`, with readable names, source ratios, measured metrics, and
SHA-256 checksums. Both Blender-containing mixtures remain included.

Original evaluation metadata and the frozen dataset manifest are preserved in
`evaluation/`. All models share 7 real validation and 22 real test images; these
small holdouts and existing unreviewed labels limit interpretation of the scores.

The previous 400-image v2 models remain in the
[archive](../synthetic_vs_real_cv/archive/models_v2_2026-10-07/README.md).

From the repository root, launch with automatic environment selection:

```bash
python run_dashboard.py
```

Open http://127.0.0.1:8765. To verify all models without starting a server, add
`--check`. For setup on another computer, see [deployment instructions](../DEPLOYMENT.md).

To import a future completed Colab export:

```bash
python3 scripts/import_colab_deployment.py /absolute/path/to/models_and_metrics
```

This validates every checkpoint before updating the active registry. The
legacy `export_deployment.py` remains for local research-run outputs.
