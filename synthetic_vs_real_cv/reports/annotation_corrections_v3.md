# Training annotation corrections v3

Audited 400 real training images and all 656 selected AI training-source images using an independent YOLO-World detector (768px). Visually reviewed all 358 proposed additions/geometry changes and 163 flagged vest crops. Accepted 262 detector proposals and removed 32 false vest labels, changing 162 label files. Rejected ambiguous proposals, combined-person detections and machinery mistaken for people. Existing boxes not flagged by this review remain unchanged; this is targeted visual review, not exhaustive manual annotation.

Original datasets and completed models are retained. Corrected copies: data/real_safety_500_v3 and data/ai_generated_v3. Image content and experiment train membership are unchanged. Classes remain 0 worker, 1 helmet, 2 vest. Ordinary shirts, jackets and full coveralls are not labeled as vests merely because they are workwear.

Real validation and test images AND labels are byte-for-byte unchanged. They were not relabeled from model predictions. Their existing annotation limitations remain; corrected training labels alone do not establish accurate ground truth or guarantee improved measured scores. All five new experiments use the same 50-image test set as v2, allowing direct score comparison with that caveat. Source-prefixed filenames do not determine actual split membership.

Each experiment starts from the same original yolov8n.pt weights, 50 epochs, 640px, batch 8, AdamW, seed 42. This isolates label changes from architecture/hyperparameter changes. New run directory: results/runs/comparison_v3. Order: real only, AI only, B mixture, Blender only, C mixture. The old v2 mixed run was stopped at a verified checkpoint and retained.

Completed v3 models are exported automatically after evaluation, alongside the two completed v2 models. The website labels new entries as corrected labels. It continues serving v2 while v3 training is incomplete.

Audit evidence and exact change log: results/annotation_audit_v3/decisions.json, applied_changes.json, review_*.jpg and vest_flags_*.jpg.
