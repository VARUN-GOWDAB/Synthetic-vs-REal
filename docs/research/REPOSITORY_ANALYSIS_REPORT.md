# Repository Analysis Report

## Repository identity

- **Repository:** `VARUN-GOWDAB/Synthetic-vs-REal`
- **Default branch reviewed:** `main`
- **Repository ID:** `1379589617`
- **Visibility:** Private
- **Primary language:** Python (100%)
- **Latest reviewed commit:** `865893fd1a160352f1d2ac2ba5a4e2798f399e67`
- **Latest commit message:** `Add synthetic versus real detection research project`
- **Review scope:** Every tracked repository entry visible in the recursive tree, all readable text/configuration/source files, the binary proposal file as a binary artifact, and the available commit history/change metadata.

## Executive summary

This repository is a research workspace and implementation scaffold for evaluating whether synthetic training data can reduce the amount of labeled real-world data required for industrial safety object detection. The intended task focuses on two classes—`worker/person` and `helmet`—and compares five training compositions: 100% real, 75% real + 25% synthetic, 50% real + 50% synthetic, 25% real + 75% synthetic, and 100% synthetic.

The repository has a thoughtful scientific structure: it documents the research question, dataset-selection criteria, leakage controls, domain-gap risks, experiment matrix, reproducibility requirements, and a lightweight prototype. However, it is not yet a completed end-to-end object-detection study. The real dataset is not committed, Blender output is not committed, the YOLO training function is explicitly a placeholder, and the final research report still contains `TBD`/placeholder sections. The only checked-in numeric result is a deterministic prototype evaluation, not a trained YOLO result.

## Repository inventory

### Top-level files

| Path | Analysis |
|---|---|
| `.gitignore` | Excludes virtual environments, Python caches, local datasets, processed data, logs, model checkpoints, and editor metadata. The ignore rules correctly avoid committing large/private artifacts, but they also mean the actual datasets and trained models cannot be audited from the repository alone. |
| `README.md` | Provides the workspace purpose, a Windows-specific prototype command, and an explicit warning that final real-world results have not been produced. |
| `Research Framework.txt` | Ten-item outline for the research plan. It contains a typo, `Avlation Study`, which should be `Ablation Study`. |
| `Synthetic_Data_Research_Proposal.md` | Full research proposal covering the problem, question, gap, hypothesis, methodology, baselines, dataset, metrics, contribution, and planned ablations. It emphasizes that conclusions must come from measured results. The proposal names the team as "Neural Nomads" and lists four team members with student IDs. |
| `Synthetic_Data_Research_Proposal.docx` | Binary Word version of the proposal. It is a valid Office Open XML artifact, but it is not directly diff-friendly or reviewable as source. The Markdown proposal is the maintainable version. |

### `synthetic_vs_real_cv/` (project root)

| Path | Analysis |
|---|---|
| `README.md` | Detailed project-level documentation. Defines the controlled ratio experiment, SHWD as a candidate real dataset, Blender as the synthetic generator, two initial classes, the critical-review loop, and the intended workflow. |
| `requirements.txt` | Runtime dependencies: NumPy, pandas, PyYAML, matplotlib, OpenCV headless, scikit-learn, and Ultralytics. No Python version or lock file is provided. |
| `configs/experiment_config.yaml` | Research configuration and design contract. Specifies candidate data, 70/15/15 split policy, five experiments, proposed YOLO settings, metrics, error analysis, and reproducibility fields. Several values remain deliberately unresolved placeholders. |
| `configs/yolo_training_config.json` | Concrete pilot configuration: YOLOv8n, 640px input, 50 epochs, batch size 16, learning rate 0.01, AdamW, seed 42, and precision/recall/mAP metrics. Its generic `data/processed/train`, `val`, and `test` paths do not match the per-experiment paths generated elsewhere. |
| `experiments/` | Per-ratio documentation and machine-readable manifests. The five committed manifests are marked `prepared`, not trained or evaluated. |
| `notebooks/` | Empty analysis area with a README recommending dataset audits, synthetic-vs-real comparisons, metric analysis, error inspection, and figure generation. |
| `reports/` | Research plan, dataset plan, critical review, revised methodology, and a results template. |
| `results/` | Results contract and one prototype metric artifact. Full experiment outputs, curves, confusion summaries, qualitative examples, and per-class results are absent. |
| `src/` | Python package containing data preparation/conversion, synthetic generation, experiment bookkeeping, training scaffolding, evaluation, plotting, and the prototype pipeline. |

## Source-code analysis

### Package and analysis modules

- `src/__init__.py` identifies the package as synthetic-versus-real object-detection research code.
- `src/analysis/__init__.py` documents analysis utilities.
- `src/analysis/plots.py` exposes `save_metric_curve()`, which creates a static PNG using matplotlib's `Agg` backend. It creates the output directory and closes the figure correctly. The unused `Iterable` import and the docstring's reference to a "text-based metric plot" are minor cleanliness issues.
- `src/data/__init__.py`, `src/evaluation/__init__.py`, `src/prototype/__init__.py`, `src/synthetic/__init__.py`, and `src/training/__init__.py` are documentation-only package initializers.

### Data preparation and conversion

- `src/data/build_split_manifest.py` creates JSON manifests describing dataset source, classes, split counts, purposes, and intended leakage controls. Its executable example uses illustrative counts of 700/150/150 real images and 800/100/100 synthetic images; those are planning values, not repository-measured counts.
- `src/data/dataset_split.py` loads a manifest and performs a deterministic positional 70/15/15-style split. Despite the name `split_by_group`, it does not actually group by video, camera, scene, or near-duplicate fingerprint; it only slices the input list. This is a significant gap between the documented methodology and the implementation.
- `src/data/prepare_datasets.py` copies real and synthetic images and labels into processed directories, creates the five experiment training sets, and writes `experiment_manifest_summary.json`. It uses a seeded random split for each dataset, but the experiment selection uses the first `N` sorted training images rather than a randomized or stratified sample. It also calculates real and synthetic counts independently from their respective pool sizes, so the five ratios are not necessarily equal-total-image experiments.
- `src/data/convert_voc_to_yolo.py` converts Pascal VOC XML annotations into YOLO text labels. It clamps boxes, rejects invalid dimensions, skips unsupported classes, copies images, and writes `voc_conversion_report.json`. The input class map is `person -> 0` and `hat -> 1`, while the output report calls those classes `worker` and `helmet`; this semantic mapping should be explicitly validated because `hat` and `helmet` may not be interchangeable.

### Synthetic-data generation

- `src/synthetic/generate_synthetic.py` builds a deterministic placeholder manifest rather than rendering images. It creates 50 scene records by default, cycles through lighting/background categories, and emits worker boxes with an `occluded` flag. It does not generate helmet annotations and does not create image files.
- `src/synthetic/blender_generation.py` defines `BlenderScenePlan`, writes a scene manifest, and generates a Blender Python script. The planned script targets 1,000 640×640 Cycles renders across four scene types and four lighting modes. It creates simple primitive people, helmets, floors, barriers, machines, and cameras. The annotations currently use hard-coded normalized boxes rather than projecting actual object geometry into the camera, so the exported labels are not yet reliable ground truth.
- `src/synthetic/blender_scene_template.py` is the generated/static Blender script. It repeats the simple-scene logic and renders 1,000 PNG files with JSON annotations. It cycles scene and lighting modes instead of independently randomizing all requested factors, and its object annotations again use fixed boxes unrelated to each worker's actual position, scale, camera, or visibility. This is suitable as a scaffolding template, not as a validated annotation generator.

### Prototype and evaluation

- `src/prototype/prototype_pipeline.py` creates eight in-memory demo records, each containing `worker`, `helmet`, and `vest` annotations. `SimpleDetector` simply memorizes the annotations and returns slightly shifted copies as predictions. The pipeline evaluates only the first four records and writes `results/prototype_metrics.json`.
- `src/evaluation/evaluate.py` implements IoU for `[x, y, width, height]` boxes and greedy per-class matching at a configurable IoU threshold. It returns precision, recall, F1, TP, FP, and FN. It does not implement mAP, confidence ranking, precision-recall curves, class AP, or the full Ultralytics evaluation protocol described in the research documents.
- The committed prototype result is precision `1.0`, recall `1.0`, F1 `1.0`, TP `12`, FP `0`, FN `0`. This is internally consistent with the demo's self-generated predictions but must not be interpreted as model performance or evidence about synthetic data.

### Training and experiment orchestration

- `src/training/experiment_runner.py` creates the five-ratio experiment matrix and can save it as JSON.
- `src/training/yolo_experiment_runner.py` creates concrete run records with counts and train/validation/test paths. Its mixed paths (`data/processed/mixed_75_25/train`, etc.) do not match the paths produced by `prepare_datasets.py`, which uses `data/processed/<experiment>/train`.
- `src/training/connect_experiments.py` connects real and synthetic split manifests, calculates ratio-based counts, and writes detailed run connection manifests. Defaults are 700 real and 800 synthetic training images, again representing planning defaults.
- `src/training/experiment_scheduler.py` reads run JSON and writes per-run manifests with status `prepared`.
- `src/training/generate_data_yaml.py` writes one Ultralytics-style YAML per experiment. It correctly keeps validation and test data on real splits, but it points training to the experiment-specific directory and therefore needs to be reconciled with `yolo_experiment_runner.py`'s mixed-path convention.
- `src/training/train_yolo_config.py` builds and saves the same concrete YOLOv8n pilot settings found in the JSON config.
- `src/training/train_yolo.py` is explicitly a placeholder. `train_experiment()` records configuration and writes `training_summary.json`, but never imports or invokes Ultralytics and never trains a model.

## Experiment configuration and manifests

The five committed run manifests are:

| Experiment | Real ratio | Synthetic ratio | Training count in manifest | Validation | Test | Status |
|---|---:|---:|---:|---|---|---|
| `100_real` | 100% | 0% | 100 | real | real | prepared |
| `75_real_25_synthetic` | 75% | 25% | 100 | real | real | prepared |
| `50_real_50_synthetic` | 50% | 50% | 100 | real | real | prepared |
| `25_real_75_synthetic` | 25% | 75% | 100 | real | real | prepared |
| `100_synthetic` | 0% | 100% | 100 | synthetic | real | prepared |

The manifest counts are illustrative and conflict with the separate 700/800 planning defaults. The repository should establish one authoritative manifest-generation path before any scientific result is reported.

## Research-document analysis

- `reports/dataset_plan.md` narrows the first defensible taxonomy to worker/person and helmet, recommends SHWD screening, specifies Blender variation factors, and requires grouped leakage-aware splits. It correctly identifies the synthetic-to-real domain gap as the main risk.
- `reports/critical_review.md` identifies data leakage, unrealistic synthetic images, inadequate test diversity, changing total data quantity, annotation mismatch, single-run instability, and licensing as major threats. Its redesign recommendation—matched data budgets, multiple diversity settings, repeated seeds, and a fixed untouched test set—is stronger than the currently implemented code.
- `reports/revised_methodology.md` maps criticisms to concrete design decisions and requires fixed training variables, matched controls, repeated seeds, and average/variability reporting.
- `reports/research_report.md` is a report template. Its abstract, literature references, results, discussion, and conclusion are intentionally incomplete until real experiments run.
- `Synthetic_Data_Research_Proposal.md` is the broad proposal and adds planned ablations for ratio granularity, synthetic diversity, matched dataset size, per-class effects, and test-set sensitivity.
- `Research Framework.txt` is only an outline and should be synchronized with the more complete Markdown documents.

## Dependency and execution assessment

### Declared stack

- Python (version unspecified)
- NumPy, pandas, PyYAML
- matplotlib
- OpenCV headless
- scikit-learn
- Ultralytics YOLO
- Blender (for intended synthetic-rendering stage, not in requirements.txt)

### Documented prototype command

```powershell
cd "c:\Users\SIC\Desktop\Synthetic-vs-REal\synthetic_vs_real_cv"
& "c:/Users/SIC/Desktop/Synthetic-vs-REal/.venv/Scripts/python.exe" -m src.prototype.prototype_pipeline
```

The command is machine-specific and should be replaced with a portable virtual-environment workflow in the project documentation. From the `synthetic_vs_real_cv` directory, a portable equivalent is expected to be:

```bash
python -m venv .venv
# activate .venv using the platform-specific command
pip install -r requirements.txt
python -m src.prototype.prototype_pipeline
```

The prototype should run without the real dataset or Blender. Full dataset preparation, Blender rendering, and YOLO training require additional inputs and software that are not checked into the repository.

## Change-history analysis

The available history shows a short, additive project build-out:

1. `48e12df...` — `Created Research Framework`; initial framework commit.
2. `bf173c5...`, `08cef0d...`, and `a595118...` — successive `Add files via upload` commits.
3. `865893fd...` — `Add synthetic versus real detection research project`, adding the current research workspace and implementation scaffold.

The latest commit added 2,312 lines and deleted none. Its changes introduced the root documentation and ignore rules, research proposal/report material, experiment configuration and manifests, dependency declarations, prototype results, data conversion/splitting utilities, synthetic-generation templates, plotting, evaluation, and training orchestration placeholders. No evidence was found in the available latest-commit patch of completed model training, real data ingestion, or final benchmark results.

## Strengths

1. The repository separates scientific design from implementation concerns.
2. The five-condition ratio sweep is clear and easy to extend.
3. The fixed real-world test-set principle is repeatedly documented.
4. The project explicitly calls out leakage, class imbalance, annotation mismatch, domain gap, licensing, and statistical variation.
5. Configuration, manifests, and generated metadata are treated as first-class research artifacts.
6. The lightweight prototype gives a deterministic smoke test for the evaluation path.
7. The documentation avoids presenting placeholder metrics as final research evidence.

## Priority issues and recommended actions

### Critical

1. **Implement real YOLO training.** Replace `src/training/train_yolo.py`'s placeholder with an actual Ultralytics invocation or clearly separate the orchestration layer from a tested training adapter.
2. **Fix data-path inconsistencies.** Reconcile `prepare_datasets.py`, `yolo_experiment_runner.py`, `generate_data_yaml.py`, `yolo_training_config.json`, and the committed manifests.
3. **Implement grouped splitting.** Make `dataset_split.py` genuinely group by source sequence/camera/location or near-duplicate fingerprint before splitting.
4. **Generate valid annotations.** Replace fixed synthetic boxes with camera-projected bounding boxes derived from actual Blender object geometry and visibility.
5. **Define one authoritative dataset manifest.** Remove conflicting illustrative counts and record actual image/class counts for every split.
6. **Add tests.** Test VOC conversion, box clamping, IoU/matching, split counts, label copying, ratio construction, path generation, and manifest consistency.

### High

1. Add real mAP and per-class AP evaluation, preferably by using the same Ultralytics validation protocol for every experiment.
2. Add image existence and label existence validation before training.
3. Prevent filename collisions when real and synthetic images share stems.
4. Randomize or stratify experiment sampling instead of selecting the first sorted files.
5. Add seed, dataset version, class distribution, hardware, software, checkpoint, and total-image metadata to each completed run.
6. Add matched-total-image controls so composition is not confounded with training-set size.
7. Resolve the `person/hat` versus `worker/helmet` naming convention and verify class semantics against the real dataset.
8. Add a license record for SHWD and every synthetic asset/model used.

### Medium

1. Replace the absolute Windows quick-start paths with platform-neutral instructions.
2. Add a `pyproject.toml` or lock file and document the supported Python version.
3. Synchronize the proposal, framework outline, configuration, and report template.
4. Remove unused imports and correct documentation inconsistencies such as the plot function's "text-based" description.
5. Convert the binary proposal into a reproducible export process from the Markdown source, or document which file is authoritative.

## Overall assessment

**Status: research scaffold / prototype, not a completed experiment.** The repository's strongest contribution is its explicit research-rigor framing and critical self-review. The current implementation demonstrates data-format conversion, manifest creation, deterministic toy evaluation, synthetic-scene scripting, and experiment bookkeeping, but it does not yet support defensible conclusions about synthetic data effectiveness. The next milestone should be a reproducible pilot that uses one verified dataset, one consistent path/manifest contract, valid annotations, actual YOLO training, real mAP evaluation, and at least one leakage-aware held-out test split.
