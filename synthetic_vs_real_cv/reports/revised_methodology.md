# Revised Methodology

## Criticism to design decision mapping

### Criticism: Data leakage from frames or locations
- Is it valid? Yes.
- Why? If the same scene or camera is split across train and test, the model may memorize location-specific features rather than learning transferable object patterns.
- Proposed change: Group by video, scene, camera, or near-duplicate visual fingerprint before splitting.
- Effect on experiment: More defensible test performance and a cleaner estimate of real-world generalization.

### Criticism: Dataset imbalance across classes
- Is it valid? Potentially.
- Why? Some classes, such as helmets or vests, may occur far less often than workers.
- Proposed change: Report class counts, use stratified sampling where possible, and perform per-class analysis.
- Effect on experiment: Avoids false confidence from a model that performs well only on the dominant class.

### Criticism: Synthetic images are too unrealistic
- Is it valid? Very likely if the synthetic renderer does not model background complexity, shadows, occlusion, or texture noise.
- Why? A clean synthetic domain can artificially inflate performance compared with real conditions.
- Proposed change: Require diverse backgrounds, varied lighting, object poses, occlusion, and camera setup.
- Effect on experiment: Narrows the domain gap and makes the synthetic data more representative.

### Criticism: Confounding variables between experiments
- Is it valid? Yes, if the total number of training images or the augmentation policy changes across conditions.
- Why? Then we would be comparing more than just real/synthetic ratio.
- Proposed change: Fix architecture, resolution, optimizer, augmentation, and batch size across all experiments.
- Effect on experiment: The only major variable becomes the real-to-synthetic ratio.

### Criticism: Unfair comparisons due to inconsistent data size
- Is it valid? Yes.
- Why? A model trained on 1,000 total images is not equivalent to a model trained on 500 total images if the total training budget differs across conditions.
- Proposed change: Report both ratio experiments and total-image-matched controls.
- Effect on experiment: Distinguishes “more data” from “better data composition.”

### Criticism: Model choice may dominate the result
- Is it valid? Yes, but it is manageable.
- Why? A poor model or an unstable training procedure can obscure the effect of the data mixture.
- Proposed change: Keep the model family fixed and repeat training across seeds.
- Effect on experiment: Makes the data-mixing effect easier to interpret.

## Revised study design

1. Select a real-world industrial safety dataset only after license and class screening.
2. Split the real data into train/val/test with grouping by scene or video source.
3. Create a synthetic industrial environment with variable lighting, clutter, pose, and camera angles.
4. Use the same YOLO architecture and training settings for all experiments.
5. Run the primary ratio sweep on the same fixed real test set.
6. Add a data-budget-matched control so improvements are not simply due to more total images.
7. Repeat each configuration across multiple seeds when feasible.
8. Report average performance and variability.

## Final redesigned experiment matrix

- 100% Real
- 75% Real + 25% Synthetic
- 50% Real + 50% Synthetic
- 25% Real + 75% Synthetic
- 100% Synthetic
- Plus optional matched-data control for each synthetic fraction

This design is stricter than a naive “mix data and compare” approach and is more defensible scientifically.
