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

## Core research question

Can synthetic training data reduce the amount of real-world data required for an industrial object-detection task while maintaining comparable performance on unseen real-world images?

## Experimental design

The independent variable is the real-to-synthetic training ratio. Each model is trained using the same architecture and comparable hyperparameters while evaluating on the same unseen real-world test set.

The project is now aligned to the recommended practical approach:

- Real dataset candidate: Safety Helmet Wearing Dataset (SHWD)
- Synthetic dataset tool: Blender with Python domain randomization
- Class focus for the first defensible version: worker/person and helmet
- Aim: generate a medium-realism synthetic set of roughly 500–1000 images in a short, reproducible generation window

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

- `configs/` — experiment configuration and data controls
- `data/` — real, synthetic, and processed dataset folders
- `experiments/` — per-ratio experiment notes and trackers
- `reports/` — critical review, revised methodology, and final report
- `src/` — data handling, synthetic-data utilities, training, evaluation, and analysis code
- `results/` — machine-readable outputs and plots
- `notebooks/` — analysis notebooks

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
