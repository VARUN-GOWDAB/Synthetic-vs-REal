# Evaluating the Effectiveness of Synthetic Data for Real-World Object Detection

## 1. Title

Evaluating the Effectiveness of Synthetic Data for Real-World Object Detection

## 2. Abstract

This project studies whether synthetic training data can reduce the amount of real-world labeled data required for an industrial object-detection task while preserving performance on unseen real-world images. The study compares multiple real-to-synthetic training ratios using a fixed YOLO-based detector and a consistent real-world evaluation set. The final conclusion will depend on the measured outcomes rather than on the assumption that synthetic data is inherently helpful.

[ABSTRACT TO BE GENERATED AFTER RESULTS ARE AVAILABLE]

## 3. Introduction

Industrial object detection often requires large numbers of labeled images. Real-world collection is expensive and constrained by privacy, safety, and access limitations. Synthetic data offers a possible way to broaden the training set without increasing manual labeling costs, but any benefit must be demonstrated under a controlled experimental design.

## 4. Problem Statement

The central question is whether synthetic data can replace or supplement real-world labeled images in a way that maintains real-world detection performance. This must be tested experimentally using a fixed unseen real test set.

## 5. Research Question

Can synthetic training data reduce the amount of real-world data required for an industrial object-detection task while maintaining comparable performance on unseen real-world images?

## 6. Hypothesis

A suitable mixture of real and synthetic training data may achieve detection performance comparable to a larger real-only dataset, but this claim must be validated empirically.

## 7. Objectives

- evaluate the effect of real-to-synthetic training ratio on real-world object detection
- compare performance across the same YOLO architecture and evaluation set
- characterize the synthetic-to-real domain gap
- identify which object classes benefit most and least from synthetic augmentation

## 8. Literature Review

The literature indicates that synthetic data can support training in controlled domains, but performance depends strongly on realism, scene diversity, and the domain gap between rendered and real imagery. The industrial safety setting is less standardized than autonomous-driving benchmarks, and the conditions under which synthetic data helps remain domain-specific.

[ADD REFERENCES HERE]

## 9. Dataset

The real dataset must be selected based on object classes, image diversity, licensing, annotation quality, and relevance to industrial safety. The exact dataset is not fixed in this repository because the final choice depends on a licensing and quality review.

Real dataset characteristics to validate before use:

- object classes
- number of images
- annotation format
- licensing status
- lighting diversity
- camera viewpoints
- occlusion
- class balance
- legal academic usage

## 10. Synthetic Data Generation

The synthetic dataset should be generated using a renderer such as Blender, Omniverse, UE, or Unity. It should include lighting changes, camera variations, pose diversity, clutter, occlusion, and industrial backgrounds. Object annotations should be exported automatically whenever possible.

## 11. Experimental Methodology

The study uses a controlled design in which the proportion of synthetic data in the training set is the independent variable. All other major training factors are kept stable.

Experimental ratios:

- 100% real
- 75% real + 25% synthetic
- 50% real + 50% synthetic
- 25% real + 75% synthetic
- 100% synthetic

Each configuration is evaluated on the same real-world test set.

## 12. Model Architecture

A YOLO-based detector is selected because it is a widely used, strong baseline for real-time object detection with mature tooling and reproducible training.

Model details will be recorded once the final pilot configuration is chosen, including:

- YOLO version
- model size
- input resolution
- number of epochs
- batch size
- learning rate
- optimizer
- hardware and software environment

## 13. Training Procedure

The training procedure is designed to hold constant the architecture, data preprocessing, optimizer, batch size, learning rate, resolution, and evaluation set while varying only the real-to-synthetic ratio. If resources allow, the study should repeat each configuration across multiple random seeds.

## 14. Evaluation Metrics

The primary metrics include:

- mAP
- precision
- recall
- AP per class
- loss curves

Optional metrics include:

- mAP@0.5
- mAP@0.5:0.95
- F1 score
- inference time
- false positives
- false negatives

## 15. Results

[RESULTS TO BE GENERATED AFTER FULL EXPERIMENTS]

Example table format:

| Experiment | Real Data | Synthetic Data | mAP | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100R | 100% | 0% | [TBD] | [TBD] | [TBD] |
| 75R+25S | 75% | 25% | [TBD] | [TBD] | [TBD] |
| 50R+50S | 50% | 50% | [TBD] | [TBD] | [TBD] |
| 25R+75S | 25% | 75% | [TBD] | [TBD] | [TBD] |
| 100S | 0% | 100% | [TBD] | [TBD] | [TBD] |

## 16. Error Analysis

The error analysis will examine false positives, false negatives, occlusion, crowding, poor lighting, small objects, and domain mismatch. The goal is to determine whether synthetic data improves detection on some classes while degrading others.

## 17. Discussion

The discussion will interpret results in the context of the domain gap and the real proportion of synthetic data used. It will distinguish observed results from hypotheses and limitations.

## 18. Threats to Validity

- data leakage
- class imbalance
- synthetic dataset bias
- domain mismatch
- confounding data quantity with data composition
- random variation
- annotation mismatch
- limited external validity

## 19. Limitations

This project may be limited by the quality of the public real dataset, the realism of the synthetic generator, and computational constraints that restrict repeated runs.

## 20. Conclusion

The conclusion will be written only after the experimental results are available and will follow the measured data rather than a priori expectations.

## 21. Future Work

- investigate more industrial classes
- test additional synthetic generation pipelines
- add domain-adaptation strategies
- explore semi-supervised learning
- test transfer to other factory environments

## 22. References

[ADD REFERENCES HERE]

## 23. Research Integrity Statement

This project avoids fabricating performance claims and uses a critical-review loop to challenge its own methodology before final conclusions are made.
