# Existing Google Colab workflow

**All eight models are already trained and installed locally.** Use these notebooks only to reproduce the comparison or resume a Colab run. Your datasets and upload bundles are ready; no local preparation or bundle generation is needed.

## Choose the matching notebook and bundle

| Situation | Notebook | ZIP to upload into `MyDrive/SynthReal/` |
| --- | --- | --- |
| Use the self-contained eight-model bundle | [Train_SynthReal_on_Colab.ipynb](Train_SynthReal_on_Colab.ipynb) | `synthreal_300_ai_real.zip` |
| Reuse the original dataset and five completed models already in Drive | [Update_Existing_Dataset_on_Colab.ipynb](Update_Existing_Dataset_on_Colab.ipynb) | `synthreal_experiment_update.zip` |

Keep the notebook and its matching ZIP together. Dataset folders and ZIP files are ignored by Git because the datasets are stored in Drive. A clone includes the notebooks and documentation; use the matching ZIP already in Drive, or download it if a local copy is needed. Adjacent `.sha256` files record bundle checksums, and the notebooks verify their expected bundle hash.

The self-contained bundle includes 929 unique images with labels: 300 real training images, 300 AI images, 300 rendered images, 7 real validation images and 22 real test images. Images are shared across experiments as defined by the frozen manifest. Each experiment has 300 distinct training images; holdouts are not counted as training images.

## Use the self-contained bundle

1. Create `SynthReal` in Google Drive's **My Drive** and upload `synthreal_300_ai_real.zip` there.
2. Open [Google Colab](https://colab.research.google.com/) and upload `Train_SynthReal_on_Colab.ipynb`.
3. Select a GPU runtime.
4. Run the notebook cells from top to bottom and authorize its Drive connection.
5. Review the printed run folder and completion/metric output.

The notebook checks for a GPU before training. It validates image/label hashes, class IDs, YOLO coordinates, unique training memberships and shared holdouts at runtime. It installs the Colab dependencies and saves outputs to Drive.

## Reuse an existing original upload

Upload the small `synthreal_experiment_update.zip` and run `Update_Existing_Dataset_on_Colab.ipynb` with a GPU runtime. The defaults expect:

```text
MyDrive/SynthReal/synthreal_300.zip
MyDrive/SynthReal/runs/existing_labels_300_seed42_5ea8644b/
```

The update notebook reuses `/content/synthreal_5ea8644bd890` when present. After a runtime reset, it reads the original ZIP from Drive. If you moved the original upload or completed run, edit `OLD_DATA_ZIP` or `PREVIOUS_RUN` in the notebook. Keep those original files intact; the expanded run writes to its own folder.

## Training settings and baseline reuse

The defaults are YOLOv8n, 50 epochs, image size 640, batch 8 and seed 42, with the same 7 real validation and 22 real test images. The [main README](../README.md#understand-the-completed-study) lists all eight source ratios.

With `REUSE_BASELINES = True`, five original completed models are verified and reused: real-only, AI-only, Blender-only and both three-source mixtures. In a fresh expanded run, only the three additional AI/real mixtures train. Their completed local results are already available; “additional” describes their place in the study, not unfinished work.

The helper checks baseline membership, image/label hashes, training settings, initialization and checkpoint hashes before reuse. Original scores remain original scores, and reuse is recorded in `reused_baselines.json`.

## Output folders and resuming

Both notebooks save runs under `MyDrive/SynthReal/runs/`, but their default names differ:

| Notebook | Default `RUN_NAME` |
| --- | --- |
| Self-contained notebook | `ai_real_mixtures_300_seed42_<bundle hash prefix>` |
| Existing-data update notebook | `expanded_8_models_300_seed42_<update hash prefix>` |

The imported completed run is **`expanded_8_models_300_seed42_bcf65166`**, documented in [results/colab_run](../results/colab_run/README.md). To resume any existing expanded run, use its matching notebook/bundle, unchanged settings and exact `RUN_NAME`. Do not point the expanded workflow at the original five-model run folder.

After a disconnect, reconnect to a GPU and rerun the cells. Completed experiments are skipped; unfinished experiments resume from `weights/last.pt` when available. A saved checkpoint is needed for resume. If interruption occurred before the first checkpoint and the helper cannot resume that folder, use a new run name. Periodic checkpoints are also retained.

For deliberately different settings, use a **new `RUN_NAME`** and set `REUSE_BASELINES = False` so all eight models use the new settings. This includes reducing batch size for GPU memory. Leave reuse enabled for the supplied unchanged defaults.

The final cells display `comparison_metrics.csv` and write a compact ZIP containing the eight best checkpoints and metrics. The full run folder contains more artifacts than this compact ZIP, including histories and resume checkpoints for freshly trained models. Validation selects the best checkpoint; test evaluation follows training.

## Existing local results and limits

- [Full imported run](../results/colab_run/README.md): eight best checkpoints/metrics and detailed training/test artifacts for the three additional mixtures.
- [Original five-model export](../results/colab_baselines/README.md): frozen reused baseline evidence.
- [Eight-model analysis](../results/colab_analysis_8_models/README.md): scores and interpretation.
- [Active deployment](../deployment/README.md): the models used by the local dashboard.

Labels were used without exhaustive manual review. The small real holdouts contain correlated scenes, and the additional mixtures were selected after earlier test results were observed. Treat the scores as preliminary comparisons rather than established performance on an independent final test set. Runtime integrity checks remain useful, but do not establish label accuracy.
