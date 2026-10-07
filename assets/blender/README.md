# Blender source assets

- `worker.fbx`: worker mesh.
- `Construction_Helmet.fbx`: hard-hat mesh.
- `Safety Vest.blend`: vest Blender source.

Generator: `../../synthetic_vs_real_cv/src/synthetic/industrial_factory_generator.py`.
Run with Blender's Python environment. Its asset paths now resolve from this workspace; fresh outputs go to `data/incoming/3d_rendered`, protecting curated images. The vest importer expects `Safety Vest.fbx`, which is not supplied: export the .blend asset to FBX before using that importer. The Blender runtime was not exercised during cleanup.
