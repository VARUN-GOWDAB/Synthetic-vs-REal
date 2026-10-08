# Completed Google Colab results

This folder contains **Colab outputs and analyses of those outputs only**. The current comparison has eight completed models; no local preparation results or unrelated local training archives remain here.

## Where to start

| Folder | What it contains | When to use it |
| --- | --- | --- |
| [colab_analysis_8_models](colab_analysis_8_models/README.md) | Eight-model score tables, findings and limitations | Understand the research comparison |
| [colab_run](colab_run/README.md) | Imported run metadata, eight best checkpoints/metrics, and detailed artifacts for the three additional mixtures | Inspect original Colab evidence |
| [colab_baselines](colab_baselines/README.md) | Original five-model Colab export | Trace the five reused models and their original results |
| [colab_analysis_300](colab_analysis_300/README.md) | Blender-only training-history diagnosis | Understand its poor transfer to real images |

The imported run's manifest, settings, test metrics and eight best-checkpoint hashes match the [active deployment](../deployment/README.md). Five models reuse their original completed Colab checkpoints and scores; three additional mixtures were trained for the expanded comparison. Their histories and plots are included, but the five reused models' full earlier training plots are not included in this download.

## Read the artifacts correctly

- `test_metrics.json` and `comparison_metrics.csv` contain saved real-test evaluation scores.
- `results.csv`, training curves and plots inside experiment folders describe training and validation.
- The three `_test/` folders contain real-test evaluation plots and prediction previews.
- `weights/best.pt` is the validation-selected checkpoint used for the recorded test scores. `last.pt` and periodic checkpoints support training recovery; they are not interchangeable with the evaluated best checkpoint.
- `initial_weights/` contains the pretrained initialization, rather than a completed experiment model.

See the individual folder READMEs for available files and research limitations. These are preserved Colab results; importing or documenting them did not perform new evaluations.

## Local files versus GitHub files

Best checkpoints, metrics and plots are allowed by `.gitignore`. Intermediate/last and initial pretrained checkpoints are retained locally but ignored. A normal Git checkout therefore may not include every file in the original downloaded archive. The dashboard reads `deployment/registry.json` and stores its runtime cache under `.runtime/`, outside this results folder.
