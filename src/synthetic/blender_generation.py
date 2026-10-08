from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import json


@dataclass
class BlenderScenePlan:
    output_dir: str = "data/synthetic/blender"
    scene_count: int = 1000
    workers_per_scene: int = 2
    min_workers: int = 1
    max_workers: int = 3
    random_seed: int = 42
    resolution: tuple[int, int] = (640, 640)
    render_engine: str = "CYCLES"
    lighting_variations: List[str] = field(default_factory=lambda: ["daylight", "overcast", "golden_hour", "dim"])
    scene_types: List[str] = field(default_factory=lambda: ["warehouse", "factory_floor", "loading_bay", "assembly_line"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "scene_count": self.scene_count,
            "workers_per_scene": self.workers_per_scene,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "random_seed": self.random_seed,
            "resolution": list(self.resolution),
            "render_engine": self.render_engine,
            "lighting_variations": self.lighting_variations,
            "scene_types": self.scene_types,
        }


def save_scene_manifest(plan: BlenderScenePlan, output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(plan.to_dict(), handle, indent=2)


def build_blender_script(output_path: str | Path, plan: BlenderScenePlan) -> str:
    """Generate a Blender Python script template for a medium-realism industrial scene.

    The script is intentionally simple and scriptable, so it can be run by Blender in a
    batch process to generate a large synthetic dataset with automatic annotation export.
    """
    script = f'''import json
import math
import random
from pathlib import Path

import bpy


OUTPUT_DIR = Path(r"{plan.output_dir}")
SCENE_COUNT = {plan.scene_count}
WORKERS_PER_SCENE = {plan.workers_per_scene}
RESOLUTION = {list(plan.resolution)}
SCENE_TYPES = {plan.scene_types}
LIGHTING_VARIATIONS = {plan.lighting_variations}
RANDOM_SEED = {plan.random_seed}


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def set_scene_settings():
    scene = bpy.context.scene
    scene.render.engine = "{plan.render_engine}"
    scene.render.resolution_x = RESOLUTION[0]
    scene.render.resolution_y = RESOLUTION[1]
    scene.render.film_transparent = False
    scene.cycles.samples = 32
    scene.cycles.use_adaptive_sampling = True

    world = bpy.data.worlds["World"]
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.9, 0.9, 0.92, 1.0)


def add_floor(size=(12, 12, 0.2), location=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(location=location, scale=(size[0] / 2, size[1] / 2, size[2] / 2))
    obj = bpy.context.active_object
    obj.name = "Floor"
    mat = bpy.data.materials.new(name="FloorMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs[0].default_value = (0.12, 0.12, 0.12, 1.0)
    obj.data.materials.append(mat)
    return obj


def add_light(kind="AREA", location=(0, 0, 5), energy=2000):
    bpy.ops.object.light_add(type=kind, location=location)
    light = bpy.context.active_object
    light.data.energy = energy
    return light


def make_person(position=(0, 0, 0), scale=1.0):
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.22, depth=1.6, location=(position[0], position[1], position[2] + 0.8))
    torso = bpy.context.active_object
    torso.name = "Person_Torso"
    torso.scale = (scale * 0.7, scale * 0.7, scale * 1.1)

    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.18, depth=0.9, location=(position[0], position[1], position[2] + 1.8))
    head = bpy.context.active_object
    head.name = "Person_Head"
    head.scale = (scale, scale, scale * 0.7)

    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.08, depth=0.9, location=(position[0] + 0.25, position[1], position[2] + 0.5))
    arm_left = bpy.context.active_object
    arm_left.name = "Person_Arm_L"
    arm_left.rotation_euler = (math.radians(30), 0, math.radians(10))

    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.08, depth=0.9, location=(position[0] - 0.25, position[1], position[2] + 0.5))
    arm_right = bpy.context.active_object
    arm_right.name = "Person_Arm_R"
    arm_right.rotation_euler = (math.radians(-30), 0, math.radians(-10))

    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.09, depth=1.0, location=(position[0] + 0.12, position[1], position[2] + 0.02))
    leg_left = bpy.context.active_object
    leg_left.name = "Person_Leg_L"

    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.09, depth=1.0, location=(position[0] - 0.12, position[1], position[2] + 0.02))
    leg_right = bpy.context.active_object
    leg_right.name = "Person_Leg_R"

    helmet = bpy.ops.mesh.primitive_uv_sphere_add(radius=0.18, location=(position[0], position[1], position[2] + 2.3))
    helm_obj = bpy.context.active_object
    helm_obj.name = "Helmet"
    helm_obj.scale = (1.1, 1.1, 0.65)

    return {"person": torso, "head": head, "helmet": helm_obj}


def add_barrier(location=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(location=location, scale=(0.2, 1.5, 0.7))
    obj = bpy.context.active_object
    obj.name = "Barrier"
    return obj


def add_machine(location=(0, 0, 0), scale=(1.0, 1.0, 1.0)):
    bpy.ops.mesh.primitive_cube_add(location=location, scale=(scale[0], scale[1], scale[2]))
    obj = bpy.context.active_object
    obj.name = "Machine"
    return obj


def add_camera(location=(0, -5, 2), rotation=(math.radians(90), 0, 0)):
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.active_object
    cam.rotation_euler = rotation
    cam.data.lens = 35
    return cam


def export_annotations(scene_name: str, objects: list[dict]):
    ann = []
    for obj_meta in objects:
        class_name = obj_meta["class_name"]
        bbox = obj_meta["bbox"]
        ann.append({
            "class_name": class_name,
            "bbox": [float(v) for v in bbox],
            "image_id": scene_name,
        })
    output_path = OUTPUT_DIR / f"{scene_name}.json"
    output_path.write_text(json.dumps(ann, indent=2), encoding="utf-8")


def render_scene(index: int):
    clear_scene()
    set_scene_settings()
    add_floor()

    scene_type = SCENE_TYPES[index % len(SCENE_TYPES)]
    light_mode = LIGHTING_VARIATIONS[index % len(LIGHTING_VARIATIONS)]

    if scene_type == "warehouse":
        add_machine(location=(2.2, 2.2, 0.5), scale=(1.5, 1.1, 0.8))
    elif scene_type == "factory_floor":
        add_barrier(location=(3.0, 0.0, 0.5))
    elif scene_type == "loading_bay":
        add_machine(location=(-2.4, 1.8, 0.55), scale=(1.8, 1.3, 0.8))
    else:
        add_barrier(location=(2.8, -1.3, 0.5))

    if light_mode == "daylight":
        add_light(location=(1, -3, 6), energy=2500)
    elif light_mode == "overcast":
        add_light(location=(-2, 3, 6), energy=1800)
    elif light_mode == "golden_hour":
        add_light(location=(3, 1, 6), energy=2200)
    else:
        add_light(location=(0, -2, 4), energy=1200)

    objects = []
    for worker_index in range(random.randint(1, WORKERS_PER_SCENE)):
        x = random.uniform(-2.5, 2.5)
        y = random.uniform(-2.5, 2.5)
        scale = random.uniform(0.8, 1.2)
        person = make_person(position=(x, y, 0), scale=scale)
        objects.append({"class_name": "worker", "bbox": [0.15, 0.18, 0.30, 0.42]})
        objects.append({"class_name": "helmet", "bbox": [0.22, 0.24, 0.10, 0.12]})

    cam = add_camera(location=(random.uniform(0, 2.5), random.uniform(-5.5, -4.0), random.uniform(1.2, 2.2)))
    cam.rotation_euler = (math.radians(90), 0, random.uniform(-0.25, 0.25))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path = OUTPUT_DIR / f"synthetic_scene_{index:04d}.png"
    bpy.context.scene.render.filepath = str(image_path)
    bpy.ops.render.render(write_still=True)

    export_annotations(f"synthetic_scene_{index:04d}", objects)


random.seed(RANDOM_SEED)
for idx in range(SCENE_COUNT):
    render_scene(idx)
'''
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(script, encoding="utf-8")
    return str(output)


def generate_blender_plan() -> BlenderScenePlan:
    return BlenderScenePlan(
        output_dir="data/synthetic/blender",
        scene_count=1000,
        workers_per_scene=2,
        min_workers=1,
        max_workers=3,
        random_seed=42,
        resolution=(640, 640),
        render_engine="CYCLES",
        lighting_variations=["daylight", "overcast", "golden_hour", "dim"],
        scene_types=["warehouse", "factory_floor", "loading_bay", "assembly_line"],
    )


if __name__ == "__main__":
    plan = generate_blender_plan()
    manifest_path = Path("data/processed/blender_scene_plan.json")
    save_scene_manifest(plan, manifest_path)
    script_path = build_blender_script("src/synthetic/blender_scene_template.py", plan)
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved Blender script: {script_path}")
