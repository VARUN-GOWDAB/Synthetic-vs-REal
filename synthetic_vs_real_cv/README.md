# Evaluating the Effectiveness of Synthetic Data for Real-World Object Detection

This project implements a controlled research framework for studying whether synthetic training data can reduce the amount of real-world labeled data needed for an industrial object-detection task while preserving comparable performance on unseen real images.

The repository is intentionally structured as a scientific study, not only as a model-training demo. It separates:

- research design
- dataset screening and split logic
- synthetic-data generation plan
- critical review of methodological risks
- a minimal prototype pipeline
- reproducible experiment configuration
- placeholder results and a final research report

## Current workspace status

Use [the dataset guide](data/README.md) for the three active sources and [the training guide](reports/real_3class_training.md) for the current entry point. Older prototype/two-class utilities remain as research scaffolding; their generated demo outputs and stale prepared manifests are archived. See [folder organization](reports/folder_organization.md).

## Core research question

Can synthetic training data reduce the amount of real-world data required for an industrial object-detection task while maintaining comparable performance on unseen real-world images?

## Experimental design

The independent variable is the real-to-synthetic training ratio. Each model is trained using the same architecture and comparable hyperparameters while evaluating on the same unseen real-world test set.

The project is now aligned to the recommended practical approach:

- Real dataset candidate: Safety Helmet Wearing Dataset (SHWD)
- Synthetic dataset tool: Blender with Python domain randomization
- Current classes: 0 worker/person, 1 safety helmet/hard hat, 2 safety/high-visibility vest
- Aim: generate a medium-realism synthetic set of roughly 500â€“1000 images in a short, reproducible generation window

The primary study design is:

- 100% real
- 75% real + 25% synthetic
- 50% real + 50% synthetic
- 25% real + 75% synthetic
- 100% synthetic

## Critical-review loop

The project includes a skeptical reviewer stage that explicitly checks for:

- data leakage
- class imbalance
- synthetic-data bias
- domain gap
- confounding variables
- unfair comparisons
- test-set quality
- random variation
- annotation mismatch
- reproducibility issues

This is built into the review documents in the `reports/` directory.

## Repository layout

- `configs/` â€” experiment configuration and data controls
- `data/` â€” real, synthetic, and processed dataset folders
- `experiments/` â€” per-ratio experiment notes and trackers
- `reports/` â€” critical review, revised methodology, and final report
- `src/` â€” data handling, synthetic-data utilities, training, evaluation, and analysis code
- `results/` â€” machine-readable outputs and plots
- `notebooks/` â€” analysis notebooks

## Minimal prototype

A small end-to-end prototype is included to verify the pipeline logic:

- synthetic data generation
- data split
- training stub
- inference simulation
- evaluation metrics

This prototype is intentionally lightweight and used to validate the workflow rather than to claim a final research result.

## Reproduction workflow

1. Review the experiment configuration in `configs/experiment_config.yaml`.
2. Prepare the real-world dataset according to the discussed split strategy.
3. Generate or curate synthetic images that are diverse and representative.
4. Train each experiment configuration with the same YOLO model family.
5. Evaluate models on the fixed real-world test set.
6. Save metrics to CSV/JSON and generate plots in `results/`.

## Important scientific guardrail

This repository does not invent dataset sizes, results, or performance metrics. Where a full experiment has not been run, placeholder values are used explicitly and marked as such.

This project is designed to be methodologically defensible, even if the final conclusion is that synthetic data does not provide a convincing benefit under the tested conditions.

## Three-class real-data workflow

The requested real-data model uses class 0 worker/person, class 1 safety helmet/hard hat, and class 2 safety/high-visibility vest. The curated dataset is `data/real_safety_600`, with 600 image-label pairs and 420/90/90 train/validation/test splits. Automatic labels were visually screened and obvious false vest boxes corrected; they are not exhaustive manual ground truth. See [the three-class training guide](reports/real_3class_training.md) and `configs/yolo_real_3class.json`. The earlier two-class research configuration remains a separate experiment.
