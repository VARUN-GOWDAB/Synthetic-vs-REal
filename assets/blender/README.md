# Blender source assets

These assets support optional synthetic scene generation. The existing rendered dataset is already available; **these assets are not needed to run the dashboard or use the completed Colab comparison**.

| File | Purpose |
| --- | --- |
| [worker.fbx](worker.fbx) | Worker mesh |
| [Construction_Helmet.fbx](Construction_Helmet.fbx) | Hard-hat mesh |
| [Safety Vest.blend](Safety%20Vest.blend) | Vest Blender source |

## Optional generator

The [industrial factory generator](../../src/synthetic/industrial_factory_generator.py) runs in Blender's Python environment because it uses `bpy`. Its asset paths resolve from the repository, and fresh output goes to `data/incoming/3d_rendered/` rather than overwriting the curated rendered dataset.

`GENERATE_DATASET` defaults to `False`: the script builds/saves a scene without starting an image batch. Review its settings before deliberately generating another dataset.

The vest importer expects `Safety Vest.fbx`, but only the `.blend` source is supplied here. Export the vest to that FBX filename before using the importer. This optional Blender workflow was not exercised during repository cleanup; the missing FBX does not affect existing data or deployed models.
