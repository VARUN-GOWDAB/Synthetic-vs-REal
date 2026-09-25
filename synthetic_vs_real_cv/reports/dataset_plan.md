# Dataset Plan

## 1. Real dataset candidate

The primary real-dataset candidate is the Safety Helmet Wearing Dataset (SHWD). It is a practical starting choice because it is closely aligned with the industrial safety / PPE task, is widely used in helmet-detection research, and is easier to adopt than a custom dataset assembled from unrelated sources.

### Candidate class set

For the first defensible version of the study, the recommended classes are:

- worker / person
- helmet

This is a tighter and more reliable taxonomy than trying to include machinery, vests, or restricted-zone labels without a strong real-data foundation.

### Dataset screening criteria

Before the final experiment, the dataset must be checked for:

- number of images
- annotation quality
- class distribution
- license and usage rights
- diversity of lighting, camera angle, and background
- presence of duplicate or near-duplicate scenes
- legal use for academic research

## 2. Synthetic dataset plan

The synthetic dataset will be generated using Blender with a Python-driven domain-randomization loop.

### Synthetic scene classes

- warehouse
- factory floor
- loading bay
- assembly line

### Synthetic variation factors

- camera position
- camera angle
- worker density
- worker pose
- helmet placement and color
- background type
- lighting condition
- shadows
- occlusion
- scale variation
- clutter level

### Planned synthetic image count

Target: 500–1000 images.

This is realistic within a 15-day project window while preserving medium realism and variability. The exact final count will depend on the real dataset size and the chosen training split.

## 3. Split strategy

The real dataset will be split into train, validation, and test sets using a strict protocol designed to reduce leakage.

### Leakage control rules

- group by video, sequence, camera, or near-duplicate scene before splitting
- keep the test set completely hidden from model selection and training
- never use the same scene or camera location in both training and test
- if duplicate frames are found, keep them in the same group

### Recommended split

- train: 70%
- validation: 15%
- test: 15%

This split is consistent with the project configuration and is appropriate when the real dataset is large enough.

## 4. Controlled experimental data ratios

The main experiments are:

- 100% real
- 75% real + 25% synthetic
- 50% real + 50% synthetic
- 25% real + 75% synthetic
- 100% synthetic

The exact number of images in each subset will be matched to the real data volume after the final dataset screening.

## 5. Data balance requirement

Each experiment should keep:

- same model architecture
- same hyperparameters
- same resolution
- same augmentation policy
- same evaluation test set

The variable under study is the real-to-synthetic ratio, not the model or training procedure.

## 6. Reproducibility

Each run should log:

- dataset version
- image counts per split
- class counts
- synthetic generation parameters
- random seed
- render resolution
- model version
- hardware and software versions
- evaluation metrics

## 7. Main risk to monitor

The biggest risk is a synthetic-to-real domain gap. If the synthetic dataset is too clean or too repetitive, it may produce misleadingly strong or weak results. To reduce that risk, increase variability in scenes and render conditions before final training begins.
