# Repository cleanup — 2026-10-08

The completed eight-model Colab comparison and dashboard are preserved. The redundant research wrapper is removed. All commands now run from the repository root.

## Original structure

```text
.
├── assets/blender/
├── colab/
├── deployment/{models,evaluation}/
├── docs/research/
├── scripts/
├── tests/
├── run_dashboard.py
├── DEPLOYMENT.md
├── requirements-{inference,colab}.txt
└── synthetic_vs_real_cv/
    ├── src/{dashboard,data,training,evaluation,synthetic,analysis,prototype}/
    ├── data/
    ├── configs/
    ├── results/
    ├── reports/
    ├── tests/
    ├── experiments/ (six README-only planning directories)
    ├── notebooks/ (README-only placeholder)
    ├── archive/
    ├── requirements.txt
    └── README.md
```

## Final structure

```text
.
├── assets/blender/
├── colab/                # Existing matching notebooks and upload bundles
├── data/                 # Existing images, labels, splits and source manifests
├── deployment/{models,evaluation}/
├── docs/
│   ├── deployment.md
│   ├── cleanup_report.md
│   └── research/
├── results/              # Colab outputs only
│   ├── colab_run/        # Imported full completed eight-model run
│   ├── colab_baselines/  # Original five-model Colab export
│   ├── colab_analysis_8_models/
│   └── colab_analysis_300/
├── scripts/              # Colab training, existing-data reuse and export import
├── src/{dashboard,evaluation,synthetic,analysis}/
│   └── dashboard/static/
├── tests/
├── run_dashboard.py
├── start_dashboard.ps1
├── requirements.txt
├── requirements-inference.txt
├── requirements-colab.txt
├── README.md
└── .gitignore
```

Earlier sections record intermediate cleanup decisions. The latest scope removes local preparation and restricts results to Colab outputs. Environments and runtime files are omitted from the tree; inference snapshots now use `.runtime/dashboard_cache/`.

## Analysis and moves

Git was clean before editing; no project AGENTS.md was found. The inventory included ignored datasets, checkpoints, archives and upload ZIPs. The old-to-new mapping and content hashes were recorded before changes. No agents, training, annotation regeneration or new research evaluation were used.

| Old path | New path |
| --- | --- |
| `synthetic_vs_real_cv/data` | `data` |
| `synthetic_vs_real_cv/configs` | `configs` |
| `synthetic_vs_real_cv/src` | `src` |
| `synthetic_vs_real_cv/results` | `results` |
| `synthetic_vs_real_cv/requirements.txt` | `requirements.txt` |
| `synthetic_vs_real_cv/archive` | `results/history` |
| `synthetic_vs_real_cv/reports/research_report.md` | `docs/research/research_report.md` |
| `synthetic_vs_real_cv/reports/critical_review.md` | `docs/research/critical_review.md` |
| `synthetic_vs_real_cv/reports/comparison_training.md` | `docs/research/comparison_training.md` |
| `synthetic_vs_real_cv/reports/folder_organization.md` | `docs/research/folder_organization.md` |
| `synthetic_vs_real_cv/reports/dataset_plan.md` | `docs/research/dataset_plan.md` |
| `synthetic_vs_real_cv/reports/real_dataset_replacement.md` | `docs/research/real_dataset_replacement.md` |
| `synthetic_vs_real_cv/reports/annotation_corrections_v3.md` | `docs/research/annotation_corrections_v3.md` |
| `synthetic_vs_real_cv/reports/full_visual_review_v4.md` | `docs/research/full_visual_review_v4.md` |
| `synthetic_vs_real_cv/reports/synthreal_dashboard.md` | `docs/research/synthreal_dashboard.md` |
| `synthetic_vs_real_cv/reports/real_3class_training.md` | `docs/research/real_3class_training.md` |
| `synthetic_vs_real_cv/reports/revised_methodology.md` | `docs/research/revised_methodology.md` |
| `synthetic_vs_real_cv/tests/test_dashboard.py` | `tests/test_dashboard.py` |
| `synthetic_vs_real_cv/README.md` | `docs/research/workflows.md` |
| `results/synthetic_label_correction/scripts` | `src/data/corrections` |
| `DEPLOYMENT.md` | `docs/deployment.md` |

The six original experiment planning READMEs are consolidated in `docs/research/original_experiment_design.md`, retaining their text. The root README is rewritten around the completed comparison. Historical reports remain identifiable as records of earlier work.

## Deletions and reasons

- `synthetic_vs_real_cv/src/training/experiment_scheduler.py`: Unused metadata-only demo scaffold; superseded by completed Colab workflow.
- `synthetic_vs_real_cv/src/training/experiment_runner.py`: Unused metadata-only demo scaffold; superseded by completed Colab workflow.
- `synthetic_vs_real_cv/src/training/yolo_experiment_runner.py`: Unused metadata-only demo scaffold; superseded by completed Colab workflow.
- `synthetic_vs_real_cv/src/training/train_yolo.py`: Unused metadata-only demo scaffold; superseded by completed Colab workflow.
- `synthetic_vs_real_cv/src/prototype`: Unused memorizing demo detector; not a trained model or dashboard PPE code.
- `synthetic_vs_real_cv/notebooks`: README-only reservation; useful notebooks are in colab/.
- `synthetic_vs_real_cv/experiments`: Six README-only planning folders consolidated without losing their text in docs/research/original_experiment_design.md.
- `synthetic_vs_real_cv/tests/__pycache__`: Generated Python bytecode.
- `src/__pycache__`: Generated Python bytecode.
- `src/dashboard/__pycache__`: Generated Python bytecode.
- `scripts/__pycache__`: Generated Python bytecode.
- `tests/__pycache__`: Generated Python bytecode.
- `__pycache__`: Generated Python bytecode.
- `results/dashboard_cache`: Disposable model snapshot; source checkpoints remain in deployment/models/.

The removed prototype memorized demo annotations; it was not a trained detector or the dashboard's PPE code. The four removed training scaffolds had no executable consumers in the retained repository and only wrote placeholder metadata. They were discussed in an old repository analysis report; that historical report is retained. Their behavior is superseded by the actual Colab runner and frozen experiment manifest.

Empty wrapper/report/test directories were removed after moving their contents. No dataset, trained model, saved research score or virtual environment was deleted. No Colab ZIP was rebuilt. Runtime caches regenerated by verification were removed again afterward.

## Import and path changes

- `run_dashboard.py`: subprocess working directory is the repository root.
- `scripts/prepare_colab.py`: project root is the repository root; baseline export is under `results/history/`.
- `scripts/export_deployment.py`: import search path and research project root follow the new layout. This legacy exporter was not executed against the current deployment.
- `scripts/import_colab_deployment.py` and the live deployment registry: previous-model archive locator points to `results/history/`.
- `scripts/watch_training.ps1`: uses the repository root.
- Dashboard registry and server: deployment and Ultralytics configuration now resolve directly under the repository root.
- Retained annotation, legacy training and Blender-generation tools: weights, outputs and working-directory paths follow the new root. Historical correction scripts resolve the root from their source-file location.
- Dashboard test imports remain `src.dashboard.*`; tests are consolidated into root `tests/`.
- Local Markdown links and current documentation commands follow their moved targets; deployment instructions are under `docs/deployment.md`.
- `.gitignore`: data/result paths and checkpoint exceptions follow moves; environment, temporary-file and tool-cache patterns are extended. All active and archived evaluated checkpoint exceptions were verified.

## Evidence preserved and uncertain items kept

- Older annotation/review inventories: Colab preparation directly reads `annotation_review_v5`, `full_visual_review_v4`, `comparison_preparation_v4` and `real_ppe_v2_import`.
- Other audits, curation records, original-label backups, historical model scores and reports: unique provenance; no proof that they are disposable duplicates.
- Original five-model checkpoint copies: still consumed by baseline-reuse integrity checks and bundle construction. Their archive is independently reproducible; removing copies would require changing its integrity contract. Active inference uses the deployment copies.
- Older v2 checkpoints and validation evidence: a different historical experiment, not duplicates of the eight current models.
- Local full/update Colab ZIPs: existing upload and resume workflows reference them; all ZIP bytes and checksums are unchanged.
- Legacy executable research tools and configs: the dashboard retains a research mode, and independent historical reproduction may need these files. Unused demo scaffolding was removed, but uncertain research logic was kept rather than guessed disposable.
- Blender assets, environments and runtime configuration: preserved.

Frozen dataset, evaluation and archive JSON files, raw splits and historical absolute paths were kept byte-identical. These records may mention the former wrapper or Windows machine; they are historical evidence, not accidentally missed live references. The only evaluation-adjacent JSON changed is the live registry's archive locator, not its models, scores, settings or checksums.

## Validation

- Before restructuring: 17 root tests and 8 dashboard tests passed.
- After restructuring: all 25 tests passed under one root discovery command.
- All retained Python files passed syntax parsing; dashboard imports and active configuration resolution passed.
- `python run_dashboard.py --check` passed: all eight checkpoints verify, load and perform a small CPU inference check.
- Temporary launcher/server startup passed; health, eight-model deployment registry, HTML, JavaScript and CSS endpoints returned HTTP 200. The temporary server was stopped.
- 3,333 protected files are byte-identical after relocation: all non-code/non-documentation data, labels, splits, model weights, frozen metadata, assets and Colab artifacts in the original inventory. Disposable cache files and the live registry locator are excluded.
- Regenerated Colab manifest in memory matches the saved preparation manifest exactly, including all records, memberships, source ratios and hashes. No bundle was written.
- Frozen deployment manifest hash matches its run specification.
- No broken local Markdown links were found.
- Git ignore rules allow active/historical checkpoints and ignore generated dashboard snapshots.
- Full diff reviewed with an isolated temporary Git index: 2,558 affected paths, predominantly renames; 471 added and 664 removed text lines. The complete diff whitespace check passed. The user index remains untouched.
- Live Python, launcher, PowerShell and ignore paths no longer reference the removed wrapper. Historical JSON/split records intentionally retain their original paths.

## Remaining limitations and manual checks

- Legacy local 400-image inputs still contain original Windows paths, incomplete review, absent prepared splits/weights and six missing rendered images. They were already nonportable; no legacy retraining or rebaselining was attempted.
- Historical correction programs need staged intermediate inputs. The archive README references `cleanup_2026-10-06.zip`, which is absent from this checkout; preserve its manifests and obtain the archive before rerunning corrections.
- Browser webcam permission, interactive video playback, Windows PowerShell execution and Blender rendering were not exercised. Static dashboard serving and existing PPE tests passed.
- Full Colab GPU execution and importer/exporter writes were not repeated, to avoid retraining or replacing completed results. Frozen selection/baseline tests and portable notebook/bundle bytes were checked.
- Current scores retain the existing single-seed, annotation and small-holdout limitations.
- Changes are left uncommitted and unstaged. Git initially displays moved tracked files as deletions plus new paths; normal staging will recognize unchanged files as renames.


## Follow-up Markdown cleanup

At the user's request, removed 11 obsolete or redundant Markdown files after reading their contents and searching references. The earlier move table above is a historical record; documents listed below were subsequently removed. Unique retired training provenance and future-methodology recommendations are summarized in `docs/research/workflows.md`.

| Removed file | Reason |
| --- | --- |
| `docs/research/REPOSITORY_ANALYSIS_REPORT.md` | Pre-training scaffold analysis; its repository state and recommendations are superseded. |
| `docs/research/dataset_plan.md` | Superseded two-class dataset selection plan; current data guide and manifests define the sources. |
| `docs/research/folder_organization.md` | Older layout report superseded by docs/cleanup_report.md. |
| `docs/research/original_experiment_design.md` | Obsolete planning templates; current eight-model definitions are frozen in deployment/evaluation. |
| `docs/research/research_report.md` | Unfinished report template with TBD results; the actual comparison report contains measured outcomes. |
| `docs/research/synthreal_dashboard.md` | Outdated local-training dashboard guide; current README and deployment guide provide working commands. |
| `docs/research/critical_review.md` | Planning review; relevant methodological cautions consolidated into research workflow notes. |
| `docs/research/revised_methodology.md` | Proposed redesign, not the executed study; useful future-work points consolidated into workflow notes. |
| `docs/research/real_3class_training.md` | Retired 600-image workflow; unique provenance summarized in workflow notes and curation records retained. |
| `docs/research/comparison_training.md` | Superseded local queue instructions; original design/settings summarized in workflow notes. |
| `results/curation_600/original_README.md` | Generic dataset reservation placeholder, not a curation record. |

Current setup/Colab/deployment guides, the research proposal, measured five/eight-model analyses, dataset/annotation reports and archive restoration documentation remain. No source code, datasets, checkpoints, manifests, scores or notebooks were changed by this follow-up. References to removed documents remain only in this historical cleanup ledger. Local Markdown links were checked after deletion.


## Follow-up JSON cleanup

At the user's request, parsed all project JSON files, searched code/documentation references and checked active workflow dependencies. Removed four verified obsolete JSON files:

| Removed file | Reason |
| --- | --- |
| `configs/yolo_training_config.json` | Unused generated two-class placeholder configuration; reproduced exactly by src/training/train_yolo_config.py. No retained code reads it. |
| `configs/yolo_real_3class.json` | Unused legacy standalone config pointing to retired real_safety_500 inputs; the retained three-class runner uses its own arguments and defaults. No readers found. |
| `results/comparison_preparation/experiment_b_queue.json` | Stale waiting-state file for an absent historical local queue; only the legacy producer writes it, and no retained code reads it. |
| `results/history/cleanup_validation.json` | Obsolete housekeeping checklist; verification facts preserved below, distinct from dataset manifests and measured research results. |

The removed historical checklist recorded 3,360 unchanged dataset files, passed Python syntax checks, a verified cleanup archive, 1,080 verified original annotation-backup entries, and no training started during that earlier cleanup. These are historical checklist claims, not new measurements of the current checkout.

The two referenced deletion paths in legacy source are output filenames, not missing input dependencies: `train_yolo_config.py` can recreate its placeholder config, and `continue_experiments.py` writes its own queue-status file. Those producer behaviors were retained.

Kept current configuration, dataset manifests, selection/exclusion reports, human annotation decisions, hash inventories, measured metrics, initialization/run specifications and baseline-reuse provenance. Byte-identical historical metric copies and initialization metadata remain because the original baseline export has independent validation consumers; file age or matching contents alone would not justify deleting them. Frozen archive manifests and the files listed by their integrity contracts remain intact.

All 85 remaining project JSON files are byte-identical to this follow-up’s starting state. All eight deployment checkpoint hashes still match the live registry.

After JSON deletion, all 25 Colab/dashboard/launcher tests passed. Remaining documentation links and Git diff whitespace checks passed. Test-generated Python caches were removed afterward.


## Retirement of the local-training workflow

The user explicitly authorized removing the older local-training workflow and its dependent scripts so that the configuration folder can be eliminated. This supersedes earlier decisions in this ledger to retain an optional research mode.

Removed `configs/` (six JSON configurations and the unused two-class YAML planning template), the complete legacy `src/training/` package, local queue monitoring/export tools, and split builders specific to those retired workflows:

- `configs/comparison_b.json`
- `configs/comparison_v4.json`
- `configs/active_dataset.json`
- `configs/comparison_v3.json`
- `configs/comparison_v1.json`
- `configs/comparison_v2.json`
- `configs/experiment_config.yaml`
- `src/training/__init__.py`
- `src/training/run_comparison.py`
- `src/training/continue_experiments.py`
- `src/training/train_yolo_config.py`
- `src/training/connect_experiments.py`
- `src/training/generate_data_yaml.py`
- `src/training/train_real_3class.py`
- `scripts/export_deployment.py`
- `scripts/watch_training.ps1`
- `src/data/prepare_comparison.py`
- `src/data/add_experiment_b.py`
- `src/data/build_corrected_comparison.py`
- `src/data/prepare_datasets.py`
- `src/data/build_split_manifest.py`

The dashboard now reads `deployment/registry.json` for both the root launcher and direct server startup. Its legacy queue/config selector and annotation-review writer are removed. Review controls and local queue rendering are removed from the frontend; old review API requests return the existing inference-only error. Image comparisons, CPU inference, webcam/video handling and PPE tracking remain. The root launcher no longer requires the legacy deployment-mode environment switch.

Colab training, baseline reuse, frozen dataset preparation, bundle/notebook generation and completed-export import remain in `scripts/` and `colab/`. Current training settings are in `deployment/evaluation/run_specification.json` and portable run manifests. Research reports retain source/label provenance while obsolete commands are removed. README, deployment/data guides and workflow notes reflect the new layout.

Validation: all 29 unit tests passed, including new tests for deployment-only startup without the old environment switch, checkpoint path confinement, missing-registry guidance and rejected review reads/writes. All eight models passed checksum, loading and small CPU inference checks. JavaScript syntax passed. Browser overview showed eight completed models and the correct 300/7/22 split counts; the experiment view rendered all eight scores and model details with no browser console errors. Python syntax and local documentation links passed. No live source references to removed configs, training package, exporter or selector remain.

All 3,319 protected non-documentation files are byte-identical to this phase’s starting state, including datasets, labels, checkpoints, metrics, frozen metadata, Colab ZIP/notebook pairs and source assets. No training or new research evaluation was performed. Interactive webcam/video capture and Blender execution were not repeated.


## Follow-up results cleanup

After retirement of local training, the user requested cleanup inside `results/`. Removed three superseded local preparation directories and their six JSON outputs. No remaining executable code reads these directories; the current Colab preparation uses the distinct frozen V4 inventory instead. These were local split-preparation inventories, derived similarity candidates and a count summary, not model weights or measured evaluation scores. Earlier local design/count summaries remain in research workflow notes; V3 human correction decisions and applied-change records remain in their audit directory.

Deleted files:

- `results/comparison_preparation/similarity_candidates.json`
- `results/comparison_preparation/inventory.json`
- `results/comparison_preparation/audit_summary.json`
- `results/comparison_preparation_v2/similarity_candidates.json`
- `results/comparison_preparation_v2/inventory.json`
- `results/comparison_preparation_v3/inventory.json`

Removed 2,356,442 bytes (about 2.25 MiB). Current Colab preparation/analysis, V4 hash audits, human review ledgers, correction history, source import/curation provenance and original baseline/model exports remain. The unique Blender training-history diagnosis and eight-model comparison are retained. Earlier sections of this ledger describe the state before these deletions.

All 138 retained non-documentation result/deployment files are byte-identical to the start of this cleanup.

After results cleanup, all 29 tests passed, including frozen Colab selection and baseline-reuse checks. Local documentation links and Git diff whitespace checks passed. No current source references to removed preparation paths remain.


## Colab-only results and complete-run import

The user requested only Colab results and supplied a downloaded complete run. The user then clarified that all datasets are already available and no local preparation is wanted. These instructions supersede the earlier retention of local review/audit tools and records.

The full run was imported from:

`/home/puneeth/Downloads/expanded_8_models_300_seed42_bcf65166-20261008T004700Z-1-001/expanded_8_models_300_seed42_bcf65166`

into `results/colab_run/` without the extra downloaded wrapper. All 132 original files were copied byte-for-byte; the original download was left unchanged. The run is complete with eight experiments. Its manifest hash, settings, CSV, per-model test metrics and eight best-checkpoint hashes match the current deployment. Three newly trained mixture folders include full histories, train/validation plots and periodic/resume checkpoints; their three `_test` folders contain evaluation plots. Five reused models include their original best checkpoints and metrics. The copied arguments retain original Colab paths as immutable historical metadata.

The original five-model Colab export moved from `results/history/comparison_300_blender_mixtures_2026-10-07/` to `results/colab_baselines/`. Current eight-model analysis and the unique Blender-only Colab diagnosis are retained. All other results directories are removed. Dataset provenance was initially relocated outside results, then removed when the user retired preparation entirely.

Removed retired non-Colab results:

- `results/history/README.md`
- `results/history/cleanup_manifest.json`
- `results/history/dataset_integrity_before.json`
- `results/history/models_v2_2026-10-07/README.md`
- `results/history/models_v2_2026-10-07/manifest.json`
- `results/history/models_v2_2026-10-07/comparison_metrics.csv`
- `results/history/models_v2_2026-10-07/deployment/registry.json`
- `results/history/models_v2_2026-10-07/synthetic_vs_real_cv/results/dashboard_tests/portable_validation.json`
- `results/history/models_v2_2026-10-07/synthetic_vs_real_cv/results/dashboard_tests/api_validation.json`
- `results/history/models_v2_2026-10-07/deployment/models/real_only_400.pt`
- `results/history/models_v2_2026-10-07/deployment/models/ai_only_400.pt`

Removed local preparation/audit source and supporting records:

- `data/provenance/colab_preparation_300/selection_report.json`
- `data/provenance/colab_preparation_300/dataset_manifest.json`
- `data/provenance/annotation_review_v5/single_01.jpg`
- `data/provenance/annotation_review_v5/real_train_06.jpg`
- `data/provenance/annotation_review_v5/sample_real_safety_500_v4.jpg`
- `data/provenance/annotation_review_v5/rendered_review_order.json`
- `data/provenance/annotation_review_v5/single_04.jpg`
- `data/provenance/annotation_review_v5/single_10.jpg`
- `data/provenance/annotation_review_v5/rendered_04.jpg`
- `data/provenance/annotation_review_v5/single_14.jpg`
- `data/provenance/annotation_review_v5/real_train_08.jpg`
- `data/provenance/annotation_review_v5/real_train_04.jpg`
- `data/provenance/annotation_review_v5/rendered_00.jpg`
- `data/provenance/annotation_review_v5/initial_review.jpg`
- `data/provenance/annotation_review_v5/real_train_09.jpg`
- `data/provenance/annotation_review_v5/rendered_05.jpg`
- `data/provenance/annotation_review_v5/real_train_11.jpg`
- `data/provenance/annotation_review_v5/real_train_10.jpg`
- `data/provenance/annotation_review_v5/single_15.jpg`
- `data/provenance/annotation_review_v5/spot_check_selection.json`
- `data/provenance/annotation_review_v5/real_train_00.jpg`
- `data/provenance/annotation_review_v5/holdout_02.jpg`
- `data/provenance/annotation_review_v5/real_train_review_order.json`
- `data/provenance/annotation_review_v5/holdout_01.jpg`
- `data/provenance/annotation_review_v5/real_train_02.jpg`
- `data/provenance/annotation_review_v5/sample_3d_rendered.jpg`
- `data/provenance/annotation_review_v5/single_02.jpg`
- `data/provenance/annotation_review_v5/sample_ai_generated_v4.jpg`
- `data/provenance/annotation_review_v5/input_audit.json`
- `data/provenance/annotation_review_v5/selection_report.json`
- `data/provenance/annotation_review_v5/rendered_decisions.json`
- `data/provenance/annotation_review_v5/real_decisions.json`
- `data/provenance/annotation_review_v5/holdout_04.jpg`
- `data/provenance/annotation_review_v5/rendered_03.jpg`
- `data/provenance/annotation_review_v5/real_train_01.jpg`
- `data/provenance/annotation_review_v5/rendered_01.jpg`
- `data/provenance/annotation_review_v5/single_16.jpg`
- `data/provenance/annotation_review_v5/holdout_05.jpg`
- `data/provenance/annotation_review_v5/single_03.jpg`
- `data/provenance/annotation_review_v5/single_06.jpg`
- `data/provenance/annotation_review_v5/single_review_order.json`
- `data/provenance/annotation_review_v5/single_13.jpg`
- `data/provenance/annotation_review_v5/single_11.jpg`
- `data/provenance/annotation_review_v5/single_07.jpg`
- `data/provenance/annotation_review_v5/single_17.jpg`
- `data/provenance/annotation_review_v5/real_train_03.jpg`
- `data/provenance/annotation_review_v5/single_08.jpg`
- `data/provenance/annotation_review_v5/single_00.jpg`
- `data/provenance/annotation_review_v5/rendered_02.jpg`
- `data/provenance/annotation_review_v5/holdout_00.jpg`
- `data/provenance/annotation_review_v5/holdout_06.jpg`
- `data/provenance/annotation_review_v5/real_train_07.jpg`
- `data/provenance/annotation_review_v5/real_train_05.jpg`
- `data/provenance/annotation_review_v5/holdout_03.jpg`
- `data/provenance/annotation_review_v5/single_12.jpg`
- `data/provenance/annotation_review_v5/single_05.jpg`
- `data/provenance/annotation_review_v5/holdout_review_order.json`
- `data/provenance/annotation_review_v5/dataset_manifest.json`
- `data/provenance/annotation_review_v5/single_09.jpg`
- `data/provenance/test_review/decisions.json`
- `data/provenance/annotation_audit_v3/decisions.json`
- `data/provenance/annotation_audit_v3/vest_flags.json`
- `data/provenance/annotation_audit_v3/predictions.jsonl`
- `data/provenance/annotation_audit_v3/candidates.json`
- `data/provenance/annotation_audit_v3/applied_changes.json`
- `data/provenance/full_visual_review_v4/decisions.json`
- `data/provenance/full_visual_review_v4/inventory.json`
- `data/provenance/full_visual_review_v4/review_status.json`
- `data/provenance/full_visual_review_v4/applied_changes.json`
- `data/provenance/full_visual_review_v4/dataset_cleanup.json`
- `data/provenance/real_ppe_v2_import/selected_records.json`
- `data/provenance/real_ppe_v2_import/ranked_scene_groups.json`
- `data/provenance/real_ppe_v2_import/records.json`
- `data/provenance/real_ppe_v2_import/report.json`
- `data/provenance/curation_600/source_annotation_manifest.json`
- `data/provenance/curation_600/final_validation.json`
- `data/provenance/curation_600/ranked_candidates.json`
- `data/provenance/curation_600/jpeg_repairs.json`
- `data/provenance/curation_600/original_voc_conversion_report.json`
- `data/provenance/curation_600/visual_decisions.json`
- `data/provenance/curation_600/deleted_old_datasets.json`
- `data/provenance/synthetic_label_correction/README.md`
- `data/provenance/synthetic_label_correction/duplicate_groups.json`
- `data/provenance/synthetic_label_correction/final_audit.json`
- `data/provenance/comparison_preparation_v4/inventory.json`
- `data/provenance/synthetic_label_correction/3d_rendered/final_records.json`
- `data/provenance/synthetic_label_correction/ai_generated/final_records.json`
- `src/data/__init__.py`
- `src/data/import_real_ppe_v2.py`
- `src/data/convert_voc_to_yolo.py`
- `src/data/rank_safety_600.py`
- `src/data/full_visual_review.py`
- `src/data/select_real_ppe_500.py`
- `src/data/review_real_3class.py`
- `src/data/apply_visual_review_v4.py`
- `src/data/audit_training_labels.py`
- `src/data/build_safety_600.py`
- `src/data/dataset_split.py`
- `src/data/annotate_real_3class.py`
- `src/data/review_closeups.py`
- `src/data/corrections/README.md`
- `src/data/corrections/refine_3d_labels.py`
- `src/data/corrections/refine_ai_labels.py`
- `src/data/corrections/apply_synthetic_corrections.py`
- `src/data/corrections/refine_vest_clip.py`
- `src/data/corrections/review_helmets.py`
- `src/data/corrections/render_corrected_synthetic.py`
- `scripts/prepare_colab.py`
- `scripts/create_colab_notebook.py`
- `scripts/create_colab_update.py`
- `docs/research/full_visual_review_v4.md`
- `docs/research/annotation_corrections_v3.md`

The deleted source includes `src/data/` and local dataset selection/notebook/update generators. The retained existing notebooks, ready-to-upload ZIPs, Colab training helper, existing-data reuse helper and completed-export importer remain usable. Three tests specific to deleted local selection/review logic were removed; baseline-reuse tests now read the existing frozen completed manifest instead of rebuilding a selection.

Source dataset manifest image lists, class mappings and review states are retained. Only optional path metadata (`audit`/`correction_log`) pointing to deleted review folders was removed. Dataset images, labels and split files were not changed. The active registry's optional retired-v2 archive locator was removed; its models, hashes and metrics are unchanged. The importer no longer writes that locator. Retired files have task-local recovery copies outside the repository under `/tmp/synthreal_retired_non_colab_results/` and `/tmp/synthreal_retired_preparation/`.

Runtime snapshots moved from results to `.runtime/dashboard_cache/`, keeping results Colab-only after inference. Git ignore rules allow best checkpoints and imported JPEG/PNG plots, while retaining larger periodic/last and initial pretrained checkpoints locally without including them in Git. Existing Colab ZIP/notebook bytes were not regenerated.

Validation:

- All 132 imported files match the source bytes.
- All 929 selected frozen images and matching labels exist locally and match their manifest hashes, verified without rebuilding datasets.
- The frozen manifest matches its run-specification hash; CSV agrees with all eight per-model measured scores.
- All 26 retained tests pass, covering Colab integrity/runtime relocation, baseline reuse, safe resume, launcher behavior, deployment confinement and PPE logic.
- All eight models pass checkpoint checksum, model loading and a small CPU inference check.
- Python syntax, local documentation links and Git diff whitespace checks pass.
- Best-model/plot inclusion and periodic-checkpoint exclusion rules pass.
- No live source references to retired preparation folders/tools remain.
- 3,195 protected non-documentation files match their starting hashes, including existing images, labels, splits, active deployment/evaluation, Colab bundles/notebooks and source assets. Source manifests and the live registry are excluded from that byte count only for the documented optional metadata-path changes.

No model retraining or new research evaluation was performed. Original run settings and annotation/holdout limitations remain unchanged. Fresh-clone dataset/bundle regeneration is intentionally unsupported after removal of local preparation; retain the existing bundles and datasets when relocating the project.
