# Existing research datasets

These are the source images and annotations used by the project. The datasets and Colab upload bundles are already prepared; **no local preparation is required** to run the dashboard or reuse the supplied Colab workflow.

## Source folders

| Folder | Image files | YOLO label files | Source |
| --- | ---: | ---: | --- |
| `real_safety_500_v4/` | 500 | 500 | Real photographs |
| `ai_generated_v4/` | 680 | 680 | AI-generated images |
| `3d_rendered/` | 394 | 400 | Blender renders |

These counts describe the source pools, not the training sample used by each experiment. Each dataset keeps `images/` and `labels/`; corresponding image and label files share a filename stem. Existing split files and manifests are retained as provenance.

The rendered source pool has six labels whose images are missing. Those missing images are excluded from the selected Colab bundle; the completed run's selected image/label pairs are available.

## Labels and class IDs

| ID | Meaning |
| --- | --- |
| 0 | Worker/person |
| 1 | Helmet/hard hat |
| 2 | Safety/high-visibility vest |

A YOLO detection label contains one object per line:

```text
class_id center_x center_y width height
```

Coordinates are normalized to the image dimensions. Existing labels were used without exhaustive manual review, so file integrity does not establish annotation accuracy.

## Which images were actually used?

The frozen comparison bundle contains **929 unique images**: 300 real training images, 300 AI images, 300 rendered images, 7 real validation images and 22 real test images. Each of the eight experiments selects 300 training images from these pools according to its source ratio; validation and test memberships are shared.

Use the [completed run's dataset manifest](../results/colab_run/dataset_manifest.json) for exact memberships and image/label hashes. The [deployment copy](../deployment/evaluation/dataset_manifest.json) records the same completed comparison. Source-pool counts or historical source split files should not be substituted for these frozen experiment memberships.

Preserve the selected labels, memberships and holdouts when resuming an existing run. See the [Colab guide](../colab/README.md) for using the ready-made bundles, and the [comparison report](../results/colab_analysis_8_models/README.md) for the study's limitations. The dashboard itself reads trained models from `deployment/` and does not require these datasets.

## Storage and GitHub

The datasets are stored in ZIP files in Google Drive. All local dataset subfolders, including images, labels, split files and source manifests, are ignored by Git. Only this data guide is included from `data/`. The frozen evaluation manifests remain versioned under `deployment/evaluation/` and `results/` as research evidence.

A clone will not include the source pools listed above. Use the existing matching ZIP in Drive for Colab, or download it if local dataset access is needed. Your existing local dataset files remain intact.
