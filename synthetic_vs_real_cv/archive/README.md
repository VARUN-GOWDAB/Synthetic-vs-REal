# Historical files

`cleanup_2026-10-06.zip` stores obsolete demo processed data, stale run manifests, annotation trial outputs, working scripts, and redundant staged label copies. ZIP entries preserve paths relative to the workspace. Contents were byte-verified against source files before removal.

`cleanup_manifest.json` lists each archived file with SHA-256 and size. `dataset_integrity_before.json` records active image/label hashes before cleanup.

Restore individual entries only when needed; archived configurations and scripts contain historical paths and are not active training inputs. Original synthetic annotations remain separately available at `../results/synthetic_label_correction/original_labels.zip`.
