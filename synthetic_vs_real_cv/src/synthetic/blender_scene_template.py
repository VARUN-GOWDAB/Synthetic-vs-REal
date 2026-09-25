import json
import math
import random
from pathlib import Path

import bpy


OUTPUT_DIR = Path(r"data/synthetic/blender")
SCENE_COUNT = 1000
WORKERS_PER_SCENE = 2
RESOLUTION = [640, 640]
SCENE_TYPES = ["warehouse", "factory_floor", "loading_bay", "assembly_line"]
LIGHTING_VARIATIONS = ["daylight", "overcast", "golden_hour", "dim"]
RANDOM_SEED = 42


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def set_scene_settings():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
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

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.18, location=(position[0], position[1], position[2] + 2.3))
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


def export_annotations(scene_name: str, objects):
    ann = []
    for obj_meta in objects:
        ann.append({
            "class_name": obj_meta["class_name"],
            "bbox": [float(v) for v in obj_meta["bbox"]],
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
        make_person(position=(x, y, 0), scale=scale)
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
