# Full visual annotation review — in progress

All 1,156 selected images have received a first visual pass: 500 real and 656 unique AI images. This is not a certification that all objects or boxes are correct.

The decisions ledger records proposed removals, replacements and additions using original label indices and normalized XYXY coordinates. Saved explicit changes have now been applied to isolated data/real_safety_500_v4 and data/ai_generated_v4 copies: 66 real and 11 AI reviewed label files changed. Annotations were also propagated to 24 exact duplicate AI images. Older unversioned and v3 real/AI dataset folders were deleted at user request after verifying every image exists unchanged in v4. Active configuration now points to v4; training is gated while review is incomplete. The current models remain based on their earlier annotations.

Many images still require enlarged inspection or correction. Typical issues are missed background workers/PPE, duplicate person boxes, bare heads or equipment labeled as helmets, printed portraits labeled as people, and loose/truncated boxes. Some original images are too blurred to resolve small objects confidently. Such cases stay unresolved.

Do not start training from this review until every unresolved entry is addressed, all modified overlays are checked, the remaining corrections are applied and validated, and train/test leakage is assessed. Keep the held-out real test membership fixed unless a documented leakage repair is needed. Corrected test annotations also require reevaluating comparison baselines.

Progress: `results/full_visual_review_v4/review_status.json`
Decisions: `results/full_visual_review_v4/decisions.json`
Inventory: `results/full_visual_review_v4/inventory.json`
Render enlarged original/proposed overlays: `python src/data/review_closeups.py ID [ID ...] [--proposed]`
