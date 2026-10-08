# Four-hour preparation: Synthetic vs. Real

This guide uses the completed experiment records in this project. Read sections 1–5 first. Speak answers aloud; recognizing an answer on a page is different from explaining it.

## 1. Your project in plain English

Imagine a factory photo. We want a computer to draw boxes around workers, helmets, and safety vests. Training a detector normally needs many labeled real photos. Real photos can be difficult to collect and annotate, so we investigate whether artificial images can replace some of them.

We compare three sources: real photographs, AI-generated images, and images rendered from a Blender 3D scene. We fine-tune eight copies of the same pretrained YOLOv8n detector with different source mixtures. Each copy gets 300 training images. All copies use the same real validation and test images.

The strongest recorded overall test result is the mixture with 150 real, 75 AI, and 75 Blender images. It reaches 56.34% mAP50–95, versus 52.44% for 300 real images. This suggests that a mixture can be promising on this holdout. It does not prove that synthetic data always helps or that this mixture is best for all factories.

**One-sentence research question:** How does training-image source and mixture affect worker and PPE detection on real photographs?

**Application:** A prototype dashboard compares models and can flag possible missing PPE in video. The research comparison is the main contribution; the dashboard demonstrates the trained detectors.

## 2. Four-hour schedule

| Time from now | Task | What you should be able to do |
|---|---|---|
| 0–40 minutes | Sections 1, 3, 4 | Explain the problem, pipeline, and basic terms without reading |
| 40–80 minutes | Sections 5 and 6 | Explain the results and answer the main research objections |
| 80–120 minutes | Section 7, then rehearse section 8 | Explain training and deliver the presentation |
| 120–160 minutes | Section 9 | Run the demo and locate evidence files |
| 160–210 minutes | Answer questions in section 6 aloud without looking | Handle follow-up questions and identify remaining weak areas |
| 210–230 minutes | Repeat the opening, results, limitations, and closing | Present smoothly with correct numbers |
| 230–240 minutes | Check presentation files, power, and demo; take a short break | Arrive ready rather than starting new work |

If travel/setup takes time, subtract it from practice. Do not start retraining before evaluation.

## 3. The facts to memorize

| Item | Completed project |
|---|---|
| Task | Supervised object detection |
| Classes | 0 worker/person, 1 helmet/hard hat, 2 safety/high-visibility vest |
| Model | Pretrained YOLOv8n, fine-tuned separately for each experiment |
| Number of experiments | Eight |
| Training images | 300 distinct images per experiment |
| Shared validation | Seven real images |
| Shared test | 22 real images |
| Training settings | 50 epochs, image size 640, batch size 8, seed 42 |
| Optimizer / initial learning rate | AdamW / 0.001 |
| Training platform | Google Colab GPU |
| Local demonstration | Saved checkpoints, CPU inference, browser dashboard |
| Best recorded overall mixture | 50% real / 25% AI / 25% Blender |
| Its mAP50–95 | 56.34% |
| Real-only mAP50–95 | 52.44% |
| Difference | Approximately 3.90 percentage points, computed from unrounded saved scores |
| Largest limitations | Small correlated holdouts, incomplete label review, one seed, test-informed follow-up experiments |

**Dataset sizes are different concepts:** The source pools contain 500 real photos, 680 AI images, and 394 rendered images. The frozen comparison bundle contains 929 unique images: 300 real training, 300 AI, 300 Blender, seven real validation, and 22 real test. Different experiments reuse images from these pools. Eight experiments do not mean 2,400 unique training images.

Historical proposal/report files mention earlier plans and split sizes. Use the completed manifest and result files for the evaluated experiment.

## 4. ML basics, with examples

### Classification versus detection

Classification says an image contains a helmet. Detection says where each helmet is, by predicting a bounding box and its class. We need detection because one image can contain several workers and pieces of equipment.

### Supervised learning and ground truth

Supervised learning learns from examples with target labels. Here the targets are object classes and their boxes. These annotated targets are called ground truth, but annotations can still contain human or automatic errors.

### Train, validation, and test

Training updates model weights. Validation checks generalization during training and selects the saved best checkpoint. Test measures the selected model on held-out images. Normally the final test set stays untouched until choices are fixed. In this project, earlier test results influenced the three additional mixtures; we need a new independent test set for a final claim.

### Pretrained model and transfer learning

A pretrained detector has already learned useful visual patterns from a different dataset. Fine-tuning adapts those weights to our worker/helmet/vest labels. We did not design YOLO or train the whole detector from random initialization.

### CNN, weights, loss, and optimizer

A convolutional neural network learns image features such as edges, textures, and object parts. Weights are the learned numerical parameters. A loss measures prediction error. Backpropagation computes how weights contributed to that error. The optimizer uses those gradients to update weights. We use AdamW, which also applies decoupled weight decay.

### Epoch, batch, learning rate, seed

An epoch is one training pass through the dataset. A batch is a group of examples processed together; ours is eight. The learning rate controls update size; the initial value is 0.001. A seed controls random choices to improve repeatability. One seed does not show that rankings are stable across different random runs.

### Precision and recall

Precision = TP / (TP + FP): of the objects predicted, how many were correct?

Recall = TP / (TP + FN): of the labeled objects that exist, how many did we find?

Example: there are ten labeled helmets. The detector predicts eight helmets; six are correct. TP = 6, FP = 2, FN = 4. Precision = 6/8 = 75%; recall = 6/10 = 60%.

A correct detection must match the correct class and sufficiently overlap the labeled object, under the evaluation's matching rules. Duplicate predictions can count as false positives. Object detection does not have a simple useful count of all possible true-negative boxes, so ordinary classification accuracy is not our main metric.

### IoU: whether the predicted box is in the right place

Intersection over Union = overlap area / union area of the predicted and ground-truth boxes. If overlap area is 60 and union area is 100, IoU = 0.60. An IoU threshold of 0.50 requires at least that degree of overlap for a matched detection.

### AP and mAP

Changing the confidence threshold changes precision and recall. Average precision summarizes the precision–recall curve for a class. Mean average precision averages AP across classes.

mAP50 evaluates AP at IoU 0.50. mAP50–95 averages AP over ten IoU thresholds: 0.50, 0.55, ..., 0.95, and over the classes. It demands more accurate box placement, so it is usually lower than mAP50. Our best model's 56.34% mAP50–95 is not the percentage of images correctly classified and is not its precision.

### Confidence threshold versus IoU threshold

Confidence threshold filters weak predictions. Raising it generally trades recall for precision, though behavior must be measured. Evaluation IoU compares predictions with ground truth. The dashboard's IoU control is used for non-maximum suppression (NMS), which suppresses overlapping duplicate predictions; it is a separate use of IoU.

### Overfitting and domain gap

Overfitting means fitting training-specific details without generalizing well. A domain gap is a difference between training and deployment distributions, such as rendered versus real lighting, textures, object shapes, or backgrounds. A model can learn rendered images well but fail on photographs. Low training loss alone does not establish good real-image performance.

Metric definitions: [Ultralytics metrics documentation](https://docs.ultralytics.com/guides/yolo-performance-metrics/).

## 5. Results you can defend

| Training mixture | Real | AI | Blender | Test mAP50–95 (%) |
|---|---:|---:|---:|---:|
| Real 50 / AI 25 / Blender 25 | 150 | 75 | 75 | 56.34 |
| Real 75 / AI 25 | 225 | 75 | 0 | 53.75 |
| Real only | 300 | 0 | 0 | 52.44 |
| Real 50 / AI 50 | 150 | 150 | 0 | 51.77 |
| Real 25 / AI 50 / Blender 25 | 75 | 150 | 75 | 49.12 |
| Real 25 / AI 75 | 75 | 225 | 0 | 48.81 |
| AI only | 0 | 300 | 0 | 29.17 |
| Blender only | 0 | 0 | 300 | 0.64 |

The leading model's precision is 95.63%, recall 84.55%, and mAP50 92.40%. The 75% real / 25% AI model leads precision at 96.70% and mAP50 at 94.28%. Real-only leads recall at 87.86%. The best model depends on the metric and application requirement.

**Safe conclusion:** Some mixtures performed comparably to or better than the real-only baseline on this small real holdout, while synthetic-only models performed worse. We need a larger independent evaluation to know whether the differences persist.

**Do not claim:** synthetic always beats real; Blender's contribution is proven; 56.34% is accuracy; the result is statistically significant; the system is ready for autonomous safety enforcement; real-data collection costs were measured.

## 6. Tough viva questions and short answers

**1. What is new in your project?**
The contribution is an application-specific comparison of three training-data sources and mixtures under matched model settings, plus a prototype comparison dashboard. YOLO itself is an existing algorithm. I would not claim global novelty without a literature review.

**2. What exactly changes across experiments?**
Training source proportions and selected training-image membership. Architecture, training count, initialization, settings, and real holdouts are held constant. Class/object counts and image diversity are not necessarily matched, which remains a confounding factor.

**3. Why keep 300 training images for each model?**
To avoid comparing a mixture with more total images against a smaller baseline. It controls image count, not every property of the data.

**4. Why use only real validation and test images?**
The intended application receives real images, so we measure transfer to that domain. We did not also establish performance on a separate unseen synthetic holdout.

**5. Why YOLOv8n?**
The nano variant is a compact detector suitable for limited compute and a local prototype. Keeping one established detector fixed helps focus the study on data source. We did not prove YOLOv8n is superior to other architectures.

**6. Why not the latest YOLO or Faster R-CNN?**
Our completed study uses a fixed YOLOv8n baseline. A comparison of detector architectures would be a separate experiment. The current evidence only supports conclusions for this detector and setup.

**7. Did you train from scratch?**
No. We fine-tuned copies of the same pretrained YOLOv8n initialization. Using matching starting weights helps make the comparison more consistent.

**8. How do you know the comparison is fair?**
We matched model initialization, total training images, settings, and holdout membership, and saved file hashes and manifests. That improves reproducibility, but incomplete label review and unmatched diversity/object counts limit fairness.

**9. Why 50 epochs, batch eight, and image size 640?**
These are the fixed settings used for the completed comparison. They provide a manageable training setup. We did not perform a controlled study proving these values are optimal.

**10. Why is the validation set only seven images?**
The frozen dataset has a very small retained real validation split after exclusions. That makes checkpoint selection unstable. I acknowledge it as a weakness; a larger scene-independent validation set is needed.

**11. Are 22 test images enough?**
They support a preliminary comparison, not a robust deployment claim. Multiple boxes per image do not make those images independent, and related frames reduce effective diversity.

**12. Is the winning model significantly better?**
We have not established statistical significance. We used one seed and a small correlated test set. We should repeat seeds and evaluate on a larger independent set with uncertainty estimates.

**13. Is the improvement 3.90 percent?**
It is about 3.90 percentage points in mAP50–95, from the unrounded saved metrics. A relative percentage increase uses a different calculation. I use percentage points to avoid ambiguity.

**14. Does the best mixture prove Blender helps?**
No. Mixture ratios and selected images differ, so the comparison does not isolate Blender's effect. A matched ablation should replace the Blender portion while controlling the real subset, total count, and other conditions across repeated runs.

**15. Why is Blender-only so poor?**
The recorded training loss falls while real validation performance remains poor. This is consistent with a render-to-real domain gap and/or annotation differences. We have not isolated their individual contributions. More epochs alone are not supported as a fix.

**16. Why might synthetic data help in a mixture?**
It may add variation while real examples anchor learning to the target domain. That is a plausible explanation, not a mechanism proved by our experiment.

**17. What is data leakage? Did it happen here?**
Leakage occurs when evaluation information enters training or model selection improperly. The pipeline checks exact duplicate images and known scene-group overlap, but grouping is heuristic and cannot prove full scene independence. Also, test results influenced follow-up mixtures, so a new final test set is required.

**18. Why is looking at the test set a problem if you did not train on its images?**
Repeatedly choosing experiments from test performance indirectly adapts our decisions to that test set. The reported ranking remains descriptive, but the set no longer provides an untouched final assessment.

**19. How were labels created?**
The completed experiments use existing YOLO-format labels. The real dataset documentation records mapping person, helmet, and vest into IDs 0, 1, and 2. The repository includes Blender annotation-generation utilities, but the exact provenance of every selected label must be verified before claiming a particular generation process. Labels were not exhaustively manually reviewed.

**20. Which website supplied the real dataset? Which AI model generated the images?**
The files I reviewed do not establish the original provider or exact image-generator model. I should verify those from the original download and generation records rather than invent a source or license.

**21. Does structural label validation prove the boxes are right?**
No. It can verify valid classes, coordinate ranges, matching files, and hashes. Only visual review can assess whether objects were missed, mislabeled, or inaccurately boxed.

**22. What is one YOLO annotation line?**
`class_id center_x center_y width height`, normalized by image dimensions. For example, `1 0.5 0.2 0.1 0.1` describes a helmet centered halfway across and one-fifth down the image, with width and height each one-tenth of the corresponding image dimension.

**23. Why is mAP50 high but mAP50–95 much lower?**
Boxes can match an object at IoU 0.50 yet fail stricter overlap thresholds. The difference suggests box localization becomes a challenge under tighter criteria; it does not mean half the images are wrong.

**24. Is high precision sufficient for safety?**
No. Missed workers or equipment matter too, so recall and actual violation-detection behavior require evaluation. Detection metrics alone do not validate the whole alert system.

**25. Does not detecting a helmet mean the worker is not wearing one?**
No. The helmet may be hidden, small, or missed. The dashboard flags a possible violation, not proof of absence.

**26. How does your alert system work?**
It associates helmet and vest detections with approximate head and torso regions inside a detected worker box. It tracks workers between frames using box overlap. By default, the same missing-PPE condition must persist for at least two seconds and three observations. Small or edge-truncated people are marked uncertain. This is heuristic logic, not a trained violation classifier or a sophisticated tracking model.

**27. Can it handle crowds or occlusion?**
Those are difficult cases. Geometric equipment association can assign PPE to the wrong nearby person, and overlap-based tracking can switch identities. We have not established reliable performance in those conditions.

**28. Is it real-time?**
The dashboard supports live frame inference, but speed depends on hardware, frame size, and model selection. I should report measured latency from the demo instead of claiming a guaranteed FPS. Local inference runs on the CPU.

**29. Why select best.pt instead of the final checkpoint?**
The best checkpoint is selected using validation performance. The last epoch is not necessarily the best generalizing model. The saved test metrics evaluate the selected checkpoint.

**30. What would you improve first?**
Audit labels and obtain larger genuinely independent real validation/test sets. Predefine comparisons, repeat seeds, report variation, control object counts and image diversity, and separately evaluate PPE alerts. Improve synthetic realism and diversity, then measure whether that helps.

**31. Did AI write this project? What did you personally contribute?**
Answer honestly. A suitable starting point is: “I used AI assistance extensively for implementation and documentation. I am responsible for understanding the setup, checking the results, and explaining its limitations.” Only claim specific collection, coding, training, or review work that you actually performed, and follow your course's disclosure rules.

**32. What do you say when you do not know?**
“I have not verified that detail. The result I can support is ____. To answer the other part, I would need to inspect ____ or run ____.” Give the relevant known fact and a concrete way to investigate.

## 7. Model and code walkthrough

YOLO means You Only Look Once. YOLOv8 is a single-stage detector. Its backbone extracts image features; its neck combines features across scales; its detection head predicts boxes and class scores. YOLOv8 uses an anchor-free split head. You do not need to claim you implemented these internal layers.

Training uses box localization, classification, and distribution focal loss components. Box loss concerns position/size; classification loss concerns the class; distribution focal loss helps train box-coordinate distributions. Explain their purpose before attempting equations you cannot derive.

Architecture reference: [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/).

**Training pipeline:** image/label pools → frozen manifest and split checks → select source mixture → initialize pretrained YOLOv8n → train on Colab → select best validation checkpoint → evaluate on real test → export checkpoints and metrics.

**Demonstration pipeline:** browser image/video frame → local Python server → selected saved YOLO model → predicted classes/boxes/confidences → display boxes → optional PPE association and temporal alerts.

| File | Purpose |
|---|---|
| `scripts/colab_training.py` | Validates frozen data, trains/resumes experiments, evaluates saved best checkpoints |
| `colab/Train_SynthReal_on_Colab.ipynb` | Colab entry point for the original workflow |
| `deployment/evaluation/dataset_manifest.json` | Exact images, labels, split memberships, hashes, ratios, and limitations |
| `deployment/evaluation/run_specification.json` | Recorded common training settings |
| `results/colab_run/comparison_metrics.csv` | Saved comparison numbers |
| `results/colab_run/<experiment>/test_metrics.json` | Per-model metrics and checkpoint evidence |
| `deployment/registry.json` | Maps available models to checkpoints and recorded metrics |
| `src/dashboard/server.py` | Loads models, runs CPU inference, serves browser dashboard |
| `src/dashboard/alarms.py` | Heuristic PPE association and overlap-based temporal tracking |
| `src/dashboard/static/` | Browser interface |
| `src/synthetic/` | Optional synthetic generation utilities; not required to run the trained models |

Some generation files are placeholders or optional tools. In particular, `generate_synthetic_manifest` creates metadata, not actual rendered images. Do not describe every source file as part of the completed training path.

Hashes verify that files match saved records; they do not prove data quality, correct labels, or scientific validity. Software tests likewise do not establish model accuracy.

## 8. Presentation speaking notes

Use these as topic notes for your slides. Practice in your own words; do not memorize sounds without understanding.

**Problem and objective:** “Our project studies worker and personal protective equipment detection. The practical challenge is that collecting and labeling real training photos can be difficult. We ask whether AI-generated or Blender-rendered images can replace part of the real training data while preserving performance on real images.”

**Data and detector:** “We detect three classes: worker, helmet, and vest. We fine-tuned pretrained YOLOv8n models. Eight experiments used different mixtures of real, AI, and Blender images, with 300 training images each.”

**Experimental design:** “We kept initialization and training settings consistent: 50 epochs, image size 640, batch eight, and seed 42. Every experiment used the same seven real validation images and 22 real test images. Validation selected the best checkpoint, and the saved test metrics report its performance.”

**Metrics:** “We report precision, recall, mAP50, and mAP50–95. Precision measures how reliable predicted detections are; recall measures how many labeled objects are found. mAP summarizes detection performance across classes. mAP50–95 also tests stricter box-overlap requirements.”

**Results:** “The highest recorded mAP50–95 was 56.34%, using 50% real, 25% AI, and 25% Blender images. Real-only scored 52.44%. AI-only scored 29.17%, and Blender-only 0.64%. This shows weak transfer for synthetic-only models in this setup. Mixing some real images was more promising.”

**Demonstration:** “The dashboard loads saved models to compare detections on an image. It also supports video frames and a prototype alert for possible missing PPE. Alerts use geometric association and persistence across observations; they do not prove a violation.”

**Limitations and conclusion:** “These results are preliminary. The holdouts are small and correlated, labels were not exhaustively reviewed, and the study used one seed. Earlier test scores also influenced follow-up mixtures, so we need a new independent final test. Our conclusion is that synthetic mixtures deserve further investigation, not that one mixture is universally optimal.”

## 9. Demo rehearsal

1. From the project folder, run `python3 run_dashboard.py` and open `http://127.0.0.1:8765`.
2. Start with one understandable real image that was not used for training. If you use a saved test image, identify it as a test-set illustration; it adds no new evaluation evidence.
3. Compare real-only with the leading three-source mixture. Explain the colored boxes, class labels, and confidence scores.
4. Show the recorded comparison table. Distinguish the test-set metrics from a single-image demonstration.
5. If showing live inference, explain that PPE alerts require visible workers and persistent observations. A webcam view without relevant helmets/vests is not an informative PPE benchmark.
6. Be ready to show a missed detection and explain uncertainty. Never select one attractive example and claim it proves the overall winner.
7. If the demo fails, present saved metrics and artifacts, state the runtime problem, and avoid pretending a screenshot is live inference.

An optional model check is `python3 run_dashboard.py --check`. It verifies model files, loading, and small predictions; it does not rerun research evaluation.

## 10. Last-minute oral checklist

Close this file and answer aloud:

- What question does the project investigate?
- What does one model receive during training, and what does it predict?
- Why are training, validation, and test different?
- What do precision, recall, IoU, and mAP mean?
- Which mixture leads mAP50–95, and how does it compare with real-only?
- Why does the result not prove Blender helps?
- Why is synthetic-only performance weak in this setup?
- Why do we need a new independent test set?
- How do PPE alerts work, and how can they fail?
- What did you actually do personally, and what was AI-assisted?

Answer pattern: direct answer → reason → project evidence → limitation. Keep the first answer short, then expand if the teacher asks.
