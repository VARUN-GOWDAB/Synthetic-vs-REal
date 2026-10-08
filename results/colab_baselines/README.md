# Original five-model Colab export

This folder preserves the original completed **five-model, 300-image Colab comparison**. These five checkpoints and their original test scores were reused in the expanded eight-model run after membership, data hashes, settings, initialization and checkpoint checks.

| Original experiment | Real images | AI images | Blender images |
| --- | ---: | ---: | ---: |
| Real only | 300 | 0 | 0 |
| AI only | 0 | 300 | 0 |
| Blender only | 0 | 0 | 300 |
| AI 50% / Blender 25% / real 25% | 75 | 150 | 75 |
| Real 50% / AI 25% / Blender 25% | 150 | 75 | 75 |

All five used the same 7 real validation images and 22 real test images. Each experiment folder contains `weights/best.pt` and `test_metrics.json`. Root files include the [five-model comparison CSV](comparison_metrics.csv), [original dataset manifest](dataset_manifest.json), run specification, environment, initialization record and completion status. This is a compact export; full training curves and prediction panels are not included.

## How this relates to the current results

The expanded comparison adds three completed AI/real mixtures while retaining these five original checkpoints and scores. Reuse is recorded in [the expanded run's provenance](../colab_run/reused_baselines.json); it does not represent fresh evaluation of these five models.

Read the [eight-model comparison](../colab_analysis_8_models/README.md) for the current findings, or [colab_run](../colab_run/README.md) for the expanded run's artifacts. The dashboard uses the eight models in [deployment](../../deployment/README.md), rather than loading this provenance folder directly. Preserve the exported metadata and checkpoint bytes so future reuse checks remain meaningful.
