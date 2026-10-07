# Workspace cleanup — 2026-10-06

Current data: 600 real, 680 AI-generated, and 400 rendered image-label pairs. Image and annotation contents are preserved. Synthetic exact-duplicate exclusions remain in source manifests. Original synthetic labels remain backed up separately.

## Changes

- Grouped Blender meshes under assets/blender and research documents under docs/research.
- Moved the factory generator to src/synthetic and its YOLO-World checkpoint to weights.
- Updated live generator/annotation model paths; generator output now targets an incoming directory.
- Archived 1,294 obsolete/intermediate files with byte verification before deleting loose copies: demo processed datasets, stale experiment manifests, trial audit outputs, runtime scripts, repeated staged labels, and old review crops.
- Removed disposable root Python bytecode cache.
- Kept current final audit records, final overview sheets, curation decisions, all source code, pretrained weights, environment, and original annotation backups.
- Added dataset and archive guides and ignore rules for local bulk datasets/generated media.

## Remaining project work

The three-class real training entry point is current. Older experiment/prototype utilities still assume earlier data layouts or class definitions and need adaptation before comparative experiments. Prepare source-aware synthetic/mixed splits using a fixed real holdout; do not use archived sample manifests. Labels have targeted visual review, not exhaustive manual ground-truth verification.

The Blender vest source is a .blend file, while its importer expects FBX; the missing vest FBX was documented. Blender execution and YOLO training were not run as part of this organization task.
