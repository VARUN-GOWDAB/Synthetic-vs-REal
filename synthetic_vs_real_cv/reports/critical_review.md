# Critical Review

## Strengths

- The study isolates the real-to-synthetic ratio as the main experimental variable, which is scientifically useful.
- The same unseen real test set is used across all experiments, which is the correct way to compare training-data composition fairly.
- The project uses a clear baseline structure with real-only and synthetic-only anchor conditions.
- The study includes per-class analysis and explicit attention to domain gap, which are often omitted in synthetic-data work.
- The methodology is explicit about avoiding a single-run conclusion when computational resources allow repeated seeds.

## Major Problems

1. Data leakage risk is real if frames from the same camera sequence or location appear in both the train and test split.
2. If the synthetic dataset is too clean or too uniform, the domain gap may be artificially small in favor of the synthetic data or artificially large in the opposite direction.
3. A single real test set may not be large or diverse enough to support strong conclusions across all classes.
4. If total dataset size changes between experiments while only the ratio changes, the study may be confounding data quantity with data composition.
5. Real and synthetic images may not be comparable in class definition, annotation granularity, or object visibility.
6. A single training run may yield misleading results if runs are noisy or highly sensitive to initialization.
7. Without a formal audit of dataset licenses, the project may be scientifically weak or legally risky.

## Minor Problems

- Some industrial classes may be underrepresented in the chosen real data.
- Image augmentations may differ across train and test conditions if not carefully controlled.
- Annotation quality needs to be checked for both the real and synthetic set.
- The final object classes should be finalized only after the real dataset is inspected and not assumed in advance.

## Threats to Validity

### Internal validity

The biggest internal-validity threat is confounding: if only the ratio changes, but the total images, augmentation policy, label quality, or scene distribution also change, then observed differences cannot be attributed cleanly to synthetic data alone.

### External validity

The result may not generalize beyond the selected industrial domain, camera setup, or object complexity. A real-world occupational site with different lighting, motion blur, or clutter may behave differently.

### Construct validity

The construct of interest is the usefulness of synthetic data for realistic industrial object detection. If synthetic images are not representative of the real environment, the experiment may not measure the right construct.

### Statistical conclusion validity

A single trial cannot support strong statistical claims. The reported differences must be checked against seed-to-seed variation and the uncertainty of the evaluation metric.

## Recommended Changes

- Pre-register the split policy and stop conditions before training begins.
- Group near-duplicate sequences before splitting.
- Use a fixed real test set and keep it untouched throughout model selection.
- Track the exact total number of training images per configuration.
- Validate synthetic realism with visual inspection and domain-statistic comparisons.
- Include at least three random seeds when feasible.
- Ensure identical annotation conventions and class definitions across real and synthetic data.

## Experimental Redesign

A stronger design is to use:

1. one real-only baseline
2. one real-data-matched control for each synthetic fraction
3. several synthetic diversity settings
4. repeated runs across seeds
5. a strict fixed test split that is never used for validation or model selection

This prevents the study from being misread as a simple case of “more images is better.”

## Final Research-Rigor Assessment

Needs improvement before final conclusions are drawn. The design is promising, but without strict leakage control, fixed data budget accounting, and repeated runs, the experiment could produce misleading evidence.
