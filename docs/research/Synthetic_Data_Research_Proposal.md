# Evaluating the Effectiveness of Synthetic Data for Real-World Object Detection

*A Comparative Experimental Study of Synthetic and Real Training Data for Industrial Safety Object Detection*

**Team Name:** Neural Nomads

**Team Members:**
1. Varun Gowda B-1RN24CS285
2. Puneeth A N-1RN24CS193
3. Rishikesh P U-1RN24CS211
4. Pavan R Gowda-1RN24CS162

---

## 1. Problem Statement

Modern object-detection systems typically require large volumes of labeled training data. Collecting and manually annotating real-world images is expensive, time-consuming, and often impractical in industrial environments, where limited access, privacy concerns, safety constraints, and the difficulty of capturing diverse operating conditions restrict the scale of usable data.

Synthetic data offers a possible alternative or supplement. Images generated in controlled virtual environments can have object locations and annotations produced automatically, substantially reducing manual labeling effort. However, synthetic imagery may differ systematically from real photographs in lighting, texture, noise, and background complexity, creating a synthetic-to-real domain gap. The core problem this project addresses is that the practical value of synthetic data as a substitute or supplement for real training data has not been measured experimentally for the industrial safety object-detection setting, and cannot simply be assumed.

## 2. Research Question

***Can synthetic training data reduce the amount of real-world data required for an industrial object-detection task while maintaining comparable performance on unseen real-world images?***

This question is examined by training an object-detection model, such as a YOLO-based detector, on several controlled mixtures of real and synthetic images and evaluating every resulting model on the same unseen real-world test set.

## 3. Research Gap

A large body of prior work demonstrates that synthetic data can assist object-detection training in domains such as autonomous driving and robotics, but far less evidence exists for industrial workplace-safety detection tasks involving workers, helmets, safety vests, machinery, and restricted zones.

- Most existing studies report a single real/synthetic ratio rather than a systematic sweep across ratios, so the relationship between the proportion of synthetic data and real-world performance is not well characterized.
- Few studies isolate the real-to-synthetic ratio as the sole controlled variable while holding the model architecture, training procedure, and evaluation protocol fixed, which makes it difficult to attribute performance differences specifically to the data composition.
- Published results for industrial-safety scenarios specifically (rather than general-purpose detection benchmarks) are limited, and the practical range in which synthetic data supplements real data without unacceptable performance loss is not established.
- The interaction between dataset size, the synthetic-to-real domain gap, and detection performance is rarely quantified together in a single controlled experiment.

This project addresses these gaps by running a controlled, ratio-based comparison in an industrial-safety context, evaluating every configuration against a fixed unseen real-world test set.

## 4. Hypothesis

**A suitable mixture of synthetic and real training data can achieve object-detection performance comparable to training with a larger amount of real-world data alone.**

This is a hypothesis to be tested rather than an assumed outcome. The measured results may show that synthetic data helps, provides limited benefit, or reduces performance at high proportions; the actual experimental findings, not an expected direction, will determine the conclusion.

## 5. Proposed Methodology

The study follows a controlled experimental design in which the real-to-synthetic training ratio is the independent variable and all other factors are held constant.

### 5.1 Workflow

1. Define the industrial-safety detection task and object classes (e.g., workers, helmets, safety vests, machinery, restricted zones).
2. Obtain and inspect a suitable real-world dataset for the task.
3. Clean and standardize the real dataset and its annotations.
4. Split the real data into training, validation, and test portions, reserving the test set exclusively for final evaluation and preventing any leakage.
5. Create a synthetic industrial environment, or use an appropriate synthetic-data generation tool, to produce object-detection imagery.
6. Generate synthetic images with varied camera positions, object placements, lighting, backgrounds, and configurations, with corresponding annotations exported automatically.
7. Construct controlled real/synthetic training mixtures (e.g., 100R, 75R+25S, 50R+50S, 25R+75S, 100S).
8. Train the same YOLO-based configuration separately on each mixture, keeping architecture and hyperparameters fixed.
9. Evaluate every trained model on the identical unseen real-world test set.
10. Record mAP, precision, recall, per-class performance, and relevant training statistics.
11. Compare results using tables and graphs, and analyze where synthetic data helps and where the domain gap causes degradation.
12. Draw conclusions strictly from the measured results.

### 5.2 Experimental Variables

| Variable Type | Details |
|---|---|
| Independent variable | Proportion of synthetic data in the training set |
| Dependent variables | mAP, precision, recall, per-class AP, training behavior |
| Controlled variables | Object categories, test dataset, model architecture/version, training procedure and hyperparameters, evaluation metrics and protocol |

### 5.3 Conceptual Pipeline

*REAL DATA → REAL TRAINING POOL + REAL TEST SET | SYNTHETIC DATA GENERATION → SYNTHETIC TRAINING DATA | REAL/SYNTHETIC MIXING (100R, 75R+25S, 50R+50S, 25R+75S, 100S) → YOLO TRAINING → SAME UNSEEN REAL TEST SET → METRICS & COMPARISON → RESEARCH FINDINGS.*

## 6. Baselines

Two reference configurations anchor the comparison and give meaning to the intermediate real/synthetic mixtures:

- **100% Real baseline (100R):** the model trained only on the available real-image training pool, representing the conventional data-collection approach and the upper-bound reference against which synthetic-assisted configurations are judged.
- **100% Synthetic baseline (100S):** the model trained only on synthetic images, representing the lower-anchor case and directly exposing the magnitude of the synthetic-to-real domain gap when no real training data is used.

Intermediate mixtures (e.g., 75R+25S, 50R+50S, 25R+75S) are compared against these two baselines to determine whether partial substitution of real data with synthetic data can approach the 100R baseline's performance while using a smaller real-data pool. A reduced-real baseline (a smaller real-only subset matched in size to the real portion of a given mixture) may also be used to isolate whether any gain comes from the added synthetic images specifically, rather than simply from a larger total training set.

## 7. Dataset

### 7.1 Real-World Dataset

A real-world dataset for the industrial/workplace-safety domain will be obtained or curated, containing labeled instances of the target classes (e.g., workers, safety helmets, safety vests, machinery, restricted zones). The exact classes will be finalized after checking the availability and quality of real and synthetic data.

### 7.2 Synthetic Dataset

A synthetic dataset will be generated in a virtual industrial environment by placing virtual workers, helmets, safety vests, and machinery in varied positions, camera angles, lighting conditions, and backgrounds, then rendering images. Because object placement is known exactly by the generating system, bounding-box annotations can be produced automatically, reducing manual labeling effort.

### 7.3 Recommended Dataset Split

A portion of the real dataset is reserved as a fixed, unseen test set that is never used during training. As an illustrative example, if 2,000 real images are available, approximately 1,600 could form the real-data pool used across training experiments, with 400 held out as the final real-world test set. The exact split will be chosen based on the final dataset size and standard dataset-splitting practice. Every experiment is evaluated against the same real test set so that performance differences can be attributed, as far as possible, to the training-data composition rather than to differences in evaluation data.

## 8. Evaluation Metrics

All trained models are evaluated on the identical unseen real-world test set using standard object-detection metrics:

| Metric | Purpose |
|---|---|
| mAP (mean Average Precision) | Overall detection accuracy across all object classes and IoU thresholds |
| Precision | Proportion of predicted detections that are correct |
| Recall | Proportion of actual objects that are successfully detected |
| Per-class Average Precision (AP) | Detection accuracy broken down by individual object category |
| Training behavior / loss curves | Convergence characteristics and stability across the different data mixtures |

Where dataset size and the number of repeated runs permit, results will be reported with statistically appropriate analysis (e.g., averages across repeated runs) rather than single-run point estimates.

## 9. Research Contribution

This project contributes an empirical, ratio-controlled answer to how the proportion of synthetic training data affects real-world object-detection performance in an industrial-safety context. A particularly useful outcome would be identifying a practical range of real-to-synthetic mixing in which synthetic data meaningfully supplements or substitutes for real data without a large loss in detection performance, along with a characterization of the synthetic-to-real domain gap for this task.

Because the project is structured as a controlled experiment rather than a single-configuration application, it further contributes a reusable experimental protocol — fixed test set, controlled variables, and a systematic mixing schedule — that can be extended to other object classes, domains, or synthetic-data generation tools. The conclusions are drawn strictly from measured results rather than assumed in advance.

## 10. Ablation Study

Beyond the core real/synthetic ratio sweep, planned ablations isolate the contribution of individual factors to the observed performance differences:

- **Ratio granularity:** comparing the core five-point sweep (100R, 75R+25S, 50R+50S, 25R+75S, 100S) against additional intermediate ratios to check whether performance changes smoothly or has a threshold effect.
- **Synthetic data diversity:** training with a restricted range of synthetic lighting/backgrounds versus a wider range, to measure whether increased synthetic variability narrows the domain gap.
- **Dataset size matched to real portion:** comparing a mixture against a real-only baseline of the same total training size, to separate the effect of the synthetic images from the effect of simply having more training data.
- **Per-class breakdown:** examining whether synthetic data helps some object classes (e.g., helmets, rigid machinery) more than others (e.g., irregularly posed workers), which the aggregate mAP alone would not reveal.
- **Test-set sensitivity:** repeating evaluation on different held-out real test subsets to check that conclusions are not an artifact of one particular test split.

These ablations help confirm that any measured benefit of synthetic data is attributable to the data composition itself, and clarify the specific conditions under which synthetic data is most useful.
