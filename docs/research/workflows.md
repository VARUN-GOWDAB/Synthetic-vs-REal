# Research workflow notes

The current workflow is the completed eight-model, 300-image Colab comparison. Start with the [root README](../../README.md), [Colab guide](../../colab/README.md) and [comparison report](../../results/colab_analysis_8_models/README.md).

## Current workflow

Training runs in Colab using the notebooks and helpers in `colab/` and `scripts/`. Completed settings, membership, checkpoint hashes and metrics are frozen in `deployment/evaluation/`. Both `python run_dashboard.py` and `python -m src.dashboard.server` use `deployment/registry.json`.

The local training/configuration workflow and local dataset selection, audit and correction tools have been removed. Existing datasets and ready-to-upload Colab bundles are retained. Source images, labels and split membership were preserved; metadata fields pointing to deleted audit logs were removed without changing annotation content.

`results/` contains only Colab outputs: the full eight-model run, its analyses and original five-model baseline export. All 132 downloaded run files were byte-verified on import; active checkpoint and metric records match. The [source selection report](real_dataset_replacement.md) describes historical label remapping rather than an executable preparation workflow.

## Methodological follow-up

For future independent studies, use scene/camera-aware splits, consistent class/annotation conventions and augmentation, matched total training budgets, repeated seeds with variability reporting, class-balance checks, and documented dataset licensing. Test synthetic lighting, clutter, viewpoint and occlusion diversity. These were recommendations in earlier planning reviews; they are not claims that the completed single-seed comparison performed those additional checks. The current [comparison report](../../results/colab_analysis_8_models/README.md) explains the measured results and remaining limitations.
