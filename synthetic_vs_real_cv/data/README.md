# Datasets

The current [Colab workflow](../../colab/README.md) samples **300 training images
per experiment** from these sources and uses the existing labels without further
review. Its manifest is `../results/colab_preparation_300/dataset_manifest.json`.
The legacy local research configuration remains `../configs/comparison_v4.json`.

| Source directory | Images in this checkout | YOLO label files |
| --- | ---: | ---: |
| `real_safety_500_v4/` | 500 | 500 |
| `ai_generated_v4/` | 680 | 680 |
| `3d_rendered/` | 394 | 400 |

The AI source manifest selects 656 unique images from 680 files. The real source
has fixed 400/50/50 train/validation/test membership. The intended 3D source has
400 pairs, but six images are missing in this checkout: `synthetic_000394`,
`synthetic_000395`, `synthetic_000397`, `synthetic_000398`, `synthetic_000399`, and
`synthetic_000400`. Restore these images before preparing the 400-image rendered
experiment; keep their labels and audit records.

Each source uses `images/` and `labels/` with matching stems. Labels use normalized
YOLO boxes (`class_id center_x center_y width height`), with class IDs **0 worker**,
**1 helmet**, and **2 vest**. Honor source annotation manifests when selecting
training inputs. Preserve real holdouts and prevent overlap with training data.

The full v4 sources contain partial annotation corrections, and the legacy
review-gated runner remains blocked. The separate Colab workflow records the
user-requested existing-label policy explicitly. See `../results/full_visual_review_v4/review_status.json` and
[the review report](../reports/full_visual_review_v4.md). Saved split paths and
some manifests refer to the original Windows machine and need regeneration for
local training.

`incoming/` is reserved for new source deliveries; `processed/` holds generated
experiment splits. Both are created as needed and ignored by Git. Older
preparation scripts may target retired dataset names; inspect their inputs before
running them. Historical selection and correction records remain in `../results/`.
