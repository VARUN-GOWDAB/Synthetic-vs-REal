"""Procedural industrial factory and synthetic PPE dataset generator.

Run from Blender 4.x/4.5 via the Scripting workspace or:
    blender --background --python industrial_factory_generator.py

Worker/PPE placeholders are intentionally empty. See register_worker_asset() below
for connecting imported realistic meshes to randomization and visible segmentation.
"""

import bpy
import math
import os
import random
from mathutils import Euler, Matrix, Quaternion, Vector

# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------
NUM_IMAGES = 500
OUTPUT_DIR = "C:/Users/rnsit_coe/Desktop/factory/synthetic_dataset/"
SEED = 42
GENERATE_DATASET = False  # Build/save the scene without starting the image batch.
AUTO_IMPORT_WORKER = True
WORKER_FBX_PATH = "C:/Users/rnsit_coe/Desktop/factory/worker.fbx"
WORKER_TARGET_HEIGHT = 1.8
AUTO_IMPORT_PPE = True
HELMET_FBX_PATH = "C:/Users/rnsit_coe/Desktop/factory/Construction_Helmet.fbx"
VEST_FBX_PATH = "C:/Users/rnsit_coe/Desktop/factory/FBX.fbx"
HELMET_TARGET_HEIGHT = 0.15
VEST_TARGET_HEIGHT = 0.62
USE_CYCLES = False
RENDER_WIDTH = 1280
RENDER_HEIGHT = 720
RENDER_PERCENTAGE = 100
NUM_WORKERS = 3
NUM_UPPER_FLOOR_WORKERS = 3
HELMET_PROBABILITY = 0.8
VEST_PROBABILITY = 0.8
RANDOMIZE_CAMERA = True
RANDOMIZE_LIGHTING = True
RANDOMIZE_MACHINERY = True
RANDOMIZE_WORKER_COUNT = False
RANDOMIZE_BACKGROUND_OBJECTS = True
BACKGROUND_HIDE_PROBABILITY = 0.08
WORKER_POSITION_JITTER = 0.65
RANDOMIZE_WORKER_POSE = True
CAMERA_POSITION_JITTER = 1.8
LIGHT_ENERGY_VARIATION = (0.68, 1.38)
LIGHT_DIRECTION_JITTER = 1.1
SAVE_BLEND_FILE = True
BLEND_FILENAME = "industrial_factory_synthetic.blend"

FACTORY_LENGTH = 60.0
FACTORY_WIDTH = 40.0
FACTORY_HEIGHT = 12.0
MEZZANINE_SURFACE_Z = 6.30

# -----------------------------------------------------------------------------
# Shared state and scene helpers
# -----------------------------------------------------------------------------
RNG = random.Random(SEED)
COLLECTIONS = {}
MATERIALS = {}
MACHINE_ROOTS = []
PROP_ROOTS = []
CAMERA_BASELINES = {}
LIGHT_BASELINES = {}
LABEL_OBJECTS = {0: {}, 1: {}, 2: {}}
NEXT_PASS_INDEX = 1


def collection(name, parent=None):
    coll = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(coll)
    COLLECTIONS[name] = coll
    return coll


def move_to_collection(obj, coll):
    for old in list(obj.users_collection):
        old.objects.unlink(obj)
    coll.objects.link(obj)
    return obj


def make_empty(name, location, coll, display='PLAIN_AXES', size=0.6):
    obj = bpy.data.objects.new(name, None)
    coll.objects.link(obj)
    obj.location = location
    obj.empty_display_type = display
    obj.empty_display_size = size
    return obj


def make_material(name, color, metallic=0.0, roughness=0.5, noise=0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if noise:
        tex = nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = 3.8
        tex.inputs["Detail"].default_value = 2.0
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = noise
        bump.inputs["Distance"].default_value = 0.08
        mat.node_tree.links.new(tex.outputs["Fac"], bump.inputs["Height"])
        mat.node_tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def cube(name, location, dimensions, material, coll, bevel=0.025, rotation=None):
    dx, dy, dz = (max(float(d) / 2.0, 0.001) for d in dimensions)
    verts = [(-dx, -dy, -dz), (-dx, -dy, dz), (-dx, dy, -dz), (-dx, dy, dz),
             (dx, -dy, -dz), (dx, -dy, dz), (dx, dy, -dz), (dx, dy, dz)]
    faces = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6),
             (0, 1, 3, 2), (4, 6, 7, 5)]
    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.materials.append(material)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    coll.objects.link(obj)
    obj.location = location
    if rotation:
        obj.rotation_euler = rotation
    if bevel > 0:
        modifier = obj.modifiers.new("Soft manufactured edges", "BEVEL")
        modifier.width = min(bevel, min(dimensions) * 0.18)
        modifier.segments = 2
        modifier = obj.modifiers.new("Weighted corner normals", "WEIGHTED_NORMAL")
    return obj


def cylinder(name, start, end, radius, material, coll, vertices=16):
    a, b = Vector(start), Vector(end)
    direction = b - a
    obj = bpy.data.objects.new(name, bpy.data.meshes.new(name + "_Mesh"))
    coll.objects.link(obj)
    # Build a unit cylinder and scale it along its local Z axis.
    from mathutils import Quaternion
    verts = []
    faces = []
    for z in (-0.5, 0.5):
        for i in range(vertices):
            angle = 2.0 * math.pi * i / vertices
            verts.append((math.cos(angle), math.sin(angle), z))
    faces.append(tuple(range(vertices - 1, -1, -1)))
    faces.append(tuple(range(vertices, vertices * 2)))
    for i in range(vertices):
        nxt = (i + 1) % vertices
        faces.append((i, nxt, vertices + nxt, vertices + i))
    obj.data.from_pydata(verts, [], faces)
    obj.data.materials.append(material)
    obj.location = (a + b) * 0.5
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = direction.to_track_quat('Z', 'Y')
    obj.scale = (radius, radius, max(direction.length, 0.001))
    return obj


def parent_new_objects(root, before):
    for obj in set(bpy.data.objects) - before:
        if obj != root:
            obj.parent = root
            obj.matrix_parent_inverse = root.matrix_world.inverted()


def set_root_elevation(root, z):
    root.location.z = z
    root["base_location"] = [root.location.x, root.location.y, z]


def add_text(name, body, location, rotation, size, material, coll):
    curve = bpy.data.curves.new(name + "_Text", 'FONT')
    curve.body = body
    curve.size = size
    curve.extrude = 0.001
    obj = bpy.data.objects.new(name, curve)
    coll.objects.link(obj)
    obj.location = location
    obj.rotation_euler = rotation
    obj.data.materials.append(material)
    return obj


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll, do_unlink=True)
    COLLECTIONS.clear()
    MACHINE_ROOTS.clear()
    PROP_ROOTS.clear()
    LABEL_OBJECTS[0].clear()
    LABEL_OBJECTS[1].clear()
    LABEL_OBJECTS[2].clear()


def create_collections():
    env = collection("Factory_Environment")
    for name in ("Floor", "Walls", "Ceiling", "Structural_Beams", "Windows", "Doors",
                 "Pipes", "Cable_Trays", "Safety_Features"):
        collection(name, env)
    collection("Mezzanine_Level_02", env)
    machinery = collection("Industrial_Machinery")
    for name in ("CNC_Machines", "Conveyors", "Robots", "Tanks", "Control_Panels", "Storage"):
        collection(name, machinery)
    props = collection("Props")
    for name in ("Pallets", "Boxes", "Containers", "Tools", "Barriers"):
        collection(name, props)
    workers = collection("SYNTHETIC_WORKERS")
    collection("Worker_Models", workers)
    collection("Worker_PPE", workers)
    collection("Cameras")
    collection("Lights")
    collection("Synthetic_Data")


def create_materials():
    definitions = {
        "Concrete": ((0.31, 0.33, 0.32), 0.0, 0.83, 0.12),
        "Concrete_Light": ((0.48, 0.49, 0.46), 0.0, 0.78, 0.07),
        "Paint_White": ((0.72, 0.75, 0.72), 0.08, 0.48, 0.015),
        "Paint_OffWhite": ((0.57, 0.60, 0.58), 0.08, 0.54, 0.02),
        "Machine_Blue": ((0.075, 0.20, 0.27), 0.48, 0.31, 0.02),
        "Machine_Green": ((0.12, 0.25, 0.19), 0.42, 0.34, 0.015),
        "Safety_Yellow": ((0.92, 0.52, 0.045), 0.18, 0.38, 0.0),
        "Safety_Black": ((0.035, 0.042, 0.045), 0.32, 0.42, 0.0),
        "Steel": ((0.34, 0.39, 0.41), 0.82, 0.29, 0.0),
        "Dark_Steel": ((0.105, 0.13, 0.14), 0.78, 0.34, 0.0),
        "Stainless": ((0.62, 0.68, 0.69), 0.91, 0.22, 0.0),
        "Rubber": ((0.025, 0.031, 0.033), 0.0, 0.79, 0.0),
        "Glass": ((0.17, 0.37, 0.42), 0.16, 0.16, 0.0),
        "Plastic": ((0.16, 0.19, 0.19), 0.0, 0.48, 0.0),
        "Wood": ((0.37, 0.22, 0.105), 0.0, 0.69, 0.04),
        "Emergency_Red": ((0.63, 0.035, 0.022), 0.12, 0.36, 0.0),
        "LED": ((0.86, 0.91, 0.89), 0.05, 0.21, 0.0),
        "Marking_White": ((0.82, 0.83, 0.78), 0.0, 0.62, 0.0),
        "Marking_Yellow": ((0.92, 0.57, 0.06), 0.0, 0.52, 0.0),
        "Cable": ((0.045, 0.055, 0.058), 0.05, 0.55, 0.0),
    }
    for name, (color, metallic, roughness, noise) in definitions.items():
        MATERIALS[name] = make_material(name, color, metallic, roughness, noise)


def M(name):
    return MATERIALS[name]

# -----------------------------------------------------------------------------
# Building shell and infrastructure
# -----------------------------------------------------------------------------
def create_column(x, y, height=11.5, coll=None):
    coll = coll or COLLECTIONS["Structural_Beams"]
    cube("HSS structural column", (x, y, height / 2), (0.48, 0.48, height), M("Dark_Steel"), coll, 0.035)
    cube("Column base plate", (x, y, 0.16), (0.82, 0.82, 0.22), M("Steel"), coll, 0.018)
    for dx in (-0.27, 0.27):
        for dy in (-0.27, 0.27):
            cylinder("Anchor bolt", (x + dx, y + dy, 0.26), (x + dx, y + dy, 0.34), 0.035, M("Stainless"), coll, 10)


def create_window(x, y, z, width, height, coll=None, side='back'):
    coll = coll or COLLECTIONS["Windows"]
    cube("Industrial glazing", (x, y, z), (width, 0.12, height), M("Glass"), coll, 0.01)
    frame_mat = M("Dark_Steel")
    for dx in (-width / 2, width / 2):
        cube("Window jamb", (x + dx, y, z), (0.12, 0.2, height + 0.14), frame_mat, coll, 0.012)
    for dz in (-height / 2, height / 2):
        cube("Window rail", (x, y, z + dz), (width + 0.12, 0.2, 0.12), frame_mat, coll, 0.012)
    count = max(1, int(width / 2.0))
    for index in range(1, count):
        xx = x - width / 2 + width * index / count
        cube("Glazing mullion", (xx, y, z), (0.055, 0.16, height), frame_mat, coll, 0.008)


def create_pipe(start, end, radius=0.1, material=None, coll=None):
    return cylinder("Process pipe", start, end, radius, material or M("Stainless"), coll or COLLECTIONS["Pipes"], 20)


def create_factory_structure():
    floor = COLLECTIONS["Floor"]
    cube("Reinforced concrete slab", (0, 0, -0.2), (60, 40, 0.4), M("Concrete"), floor, 0.0)
    # Expansion joints laid as fine, dark inlays in the slab.
    for x in range(-27, 30, 6):
        cube("Floor expansion joint", (x, 0, 0.006), (0.025, 39.5, 0.012), M("Dark_Steel"), floor, 0.0)
    for y in range(-18, 20, 6):
        cube("Floor expansion joint", (0, y, 0.007), (59.5, 0.025, 0.012), M("Dark_Steel"), floor, 0.0)
    for y in (-19.1, 19.1):
        cube("Drain channel", (0, y, 0.015), (57.0, 0.16, 0.03), M("Dark_Steel"), floor, 0.01)
        for index in range(87):
            x = -28 + index * 0.65
            cube("Drain grate slot", (x, y, 0.034), (0.035, 0.13, 0.012), M("Steel"), floor, 0.0)
    # Marked travel aisles and machine exclusion zones.
    safety = COLLECTIONS["Safety_Features"]
    for y in (-14.8, -8.0, 8.0, 14.8):
        cube("Aisle line", (0, y, 0.012), (57, 0.095, 0.016), M("Marking_Yellow"), safety, 0.0)
    for x in (-25, -11, 3, 17, 27):
        for i in range(7):
            cube("Hazard stripe", (x + i * 0.42, -7.0, 0.02), (0.25, 1.35, 0.025),
                 M("Safety_Yellow" if i % 2 == 0 else "Safety_Black"), safety, 0.0)
    walls = COLLECTIONS["Walls"]
    # Lower perimeter is concrete; upper cladding is painted insulated metal panel.
    cube("North concrete wall", (0, 19.72, 2.25), (60, 0.55, 4.5), M("Concrete_Light"), walls, 0.0)
    cube("South concrete wall", (0, -19.72, 2.25), (60, 0.55, 4.5), M("Concrete_Light"), walls, 0.0)
    cube("West concrete wall", (-29.72, 0, 2.25), (0.55, 40, 4.5), M("Concrete_Light"), walls, 0.0)
    cube("East concrete wall", (29.72, 0, 2.25), (0.55, 40, 4.5), M("Concrete_Light"), walls, 0.0)
    for y in (-19.72, 19.72):
        cube("Upper insulated wall cladding", (0, y, 8.0), (60, 0.35, 7.0), M("Paint_OffWhite"), walls, 0.0)
    for x in (-29.72, 29.72):
        cube("Upper insulated wall cladding", (x, 0, 8.0), (0.35, 40, 7.0), M("Paint_OffWhite"), walls, 0.0)
    # Broad north clerestory glazing admits cool daylight above work-cell height.
    for x in (-22, -11, 0, 11, 22):
        create_window(x, 19.48, 7.55, 8.2, 4.3)
    doors = COLLECTIONS["Doors"]
    for x in (-19, 19):
        cube("Roll-up industrial door", (x, -19.35, 3.15), (6.0, 0.24, 6.0), M("Dark_Steel"), doors, 0.04)
        for z in [0.4 + i * 0.48 for i in range(12)]:
            cube("Door slat", (x, -19.19, z), (5.82, 0.04, 0.045), M("Steel"), doors, 0.008)
    for x in (-27.0, 27.0):
        cube("Emergency exit leaf", (x, 19.35, 2.15), (1.25, 0.22, 4.1), M("Emergency_Red"), doors, 0.025)
        cube("Exit push bar", (x, 19.18, 2.0), (0.76, 0.12, 0.08), M("Stainless"), doors, 0.015)
        add_text("EXIT sign", "EXIT", (x - 0.42, 19.2, 4.25), (math.pi / 2, 0, 0), 0.32, M("LED"), doors)
    # Roof deck and portal framing.
    cube("Roof deck underside", (0, 0, 11.75), (60, 40, 0.45), M("Dark_Steel"), COLLECTIONS["Ceiling"], 0.0)
    beams = COLLECTIONS["Structural_Beams"]
    for y in range(-18, 19, 6):
        cube("Roof primary girder", (0, y, 11.15), (60, 0.48, 0.72), M("Dark_Steel"), beams, 0.025)
        for x in range(-27, 30, 3):
            cube("Roof purlin", (x, y, 11.56), (0.12, 5.9, 0.18), M("Steel"), beams, 0.012)
    for x in (-27, -15, -3, 9, 21, 27):
        for y in (-17, -5, 7, 17):
            create_column(x, y)
    # Concrete wall piers and steel kick protection at column bases.
    for x in range(-24, 25, 12):
        cube("Wall pier", (x, 19.2, 2.1), (0.48, 0.8, 4.2), M("Concrete"), walls, 0.02)
    for y in (-19.0, 19.0):
        create_pipe((-28, y, 4.8), (28, y, 4.8), 0.12, M("Emergency_Red"), COLLECTIONS["Pipes"])
        for x in range(-27, 29, 3):
            create_pipe((x, y, 4.8), (x, y, 4.35), 0.035, M("Emergency_Red"), COLLECTIONS["Pipes"])
    # Ceiling services: compressed air, water, extraction, cable ladder.
    for y, z, radius, mat in [(-16.0, 10.4, 0.13, "Stainless"), (-14.8, 10.0, 0.09, "Machine_Blue"),
                               (15.7, 10.5, 0.16, "Dark_Steel"), (17.0, 9.9, 0.07, "Safety_Yellow")]:
        create_pipe((-28, y, z), (28, y, z), radius, M(mat), COLLECTIONS["Pipes"])
        for x in range(-27, 29, 6):
            create_pipe((x, y, z), (x, y, 11.2), 0.025, M("Steel"), COLLECTIONS["Pipes"])
    cable = COLLECTIONS["Cable_Trays"]
    for y in (-12.6, 12.6):
        cube("Cable tray bottom", (0, y, 9.65), (56, 0.62, 0.055), M("Dark_Steel"), cable, 0.012)
        for dy in (-0.31, 0.31):
            cube("Cable tray side", (0, y + dy, 9.79), (56, 0.055, 0.28), M("Steel"), cable, 0.012)
        for index in range(94):
            x = -27 + index * 0.6
            cube("Cable tray rung", (x, y, 9.73), (0.045, 0.58, 0.025), M("Steel"), cable, 0.0)
    # Wall-mounted electrical boxes and small wayfinding / warning plates.
    for x in (-24, -8, 8, 24):
        cube("Wall electrical junction box", (x, -19.1, 3.3), (0.55, 0.24, 0.72), M("Machine_Blue"), safety, 0.035)
        cube("Junction box indicator", (x, -18.96, 3.48), (0.1, 0.035, 0.07), M("Safety_Yellow"), safety, 0.01)
    for x in (-13, 0, 13):
        cube("Safety information sign", (x, -19.36, 5.7), (1.1, 0.08, 0.7), M("Safety_Yellow"), safety, 0.025)
        add_text("Warning plate text", "PPE REQUIRED", (x - 0.46, -19.30, 5.62), (math.pi / 2, 0, 0), 0.105, M("Safety_Black"), safety)


    def create_mezzanine():
        """Build a partial upper production deck with perimeter protection and stairs."""
        coll = COLLECTIONS["Mezzanine_Level_02"]
        x_min, x_max = -26.0, -2.0
        y_min, y_max = 3.5, 14.5
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        deck_z = MEZZANINE_SURFACE_Z - 0.15

        for x in (x_min + 0.8, -14.0, x_max - 0.8):
           for y in (y_min + 0.8, y_max - 0.8):
              create_column(x, y, deck_z - 0.15, coll)
        for y in (y_min + 0.8, center_y, y_max - 0.8):
           cube("Mezzanine primary girder", (center_x, y, deck_z - 0.28),
               (x_max - x_min, 0.28, 0.32), M("Dark_Steel"), coll, 0.025)
        for x in (-24.0, -20.0, -16.0, -12.0, -8.0, -4.0):
           cube("Mezzanine cross beam", (x, center_y, deck_z - 0.27),
               (0.2, y_max - y_min, 0.3), M("Steel"), coll, 0.02)
        cube("Mezzanine composite floor", (center_x, center_y, deck_z),
            (x_max - x_min, y_max - y_min, 0.3), M("Concrete_Light"), coll, 0.025)
        cube("Mezzanine edge fascia", (center_x, y_min, deck_z - 0.14),
            (x_max - x_min, 0.12, 0.32), M("Dark_Steel"), coll, 0.015)

        rail_z = MEZZANINE_SURFACE_Z + 0.98
        for x in (x_min, x_max):
           for y in (y_min + 0.35, y_max - 0.35):
              cube("Mezzanine guardrail top", (x, y, rail_z), (0.08, y_max - y_min - 0.7, 0.08),
                  M("Safety_Yellow"), coll, 0.018)
              cube("Mezzanine guardrail mid", (x, y, rail_z - 0.48), (0.055, y_max - y_min - 0.7, 0.055),
                  M("Steel"), coll, 0.012)
              cube("Mezzanine toe board", (x, y, MEZZANINE_SURFACE_Z + 0.12),
                  (0.07, y_max - y_min - 0.7, 0.22), M("Safety_Yellow"), coll, 0.012)
           for y in (y_min + 0.7, y_max - 0.7):
              cube("Mezzanine rail upright", (x, y, rail_z - 0.48), (0.06, 0.06, 1.0), M("Steel"), coll, 0.012)

        stair_x = -23.0
        stair_start_y = -4.1
        step_count = 28
        tread_run = (y_min - stair_start_y) / step_count
        riser = (MEZZANINE_SURFACE_Z - 0.22) / step_count
        for index in range(step_count):
           y = stair_start_y + (index + 0.5) * tread_run
           z = 0.22 + (index + 0.5) * riser
           cube("Mezzanine stair tread", (stair_x, y, z), (1.25, tread_run + 0.025, 0.075),
               M("Steel"), coll, 0.012)
        stair_top_y = y_min
        cylinder("Mezzanine stair stringer left",
               (stair_x - 0.68, stair_start_y, 0.12),
               (stair_x - 0.68, stair_top_y, MEZZANINE_SURFACE_Z - 0.05),
               0.09, M("Dark_Steel"), coll, 12)
        cylinder("Mezzanine stair stringer right",
               (stair_x + 0.68, stair_start_y, 0.12),
               (stair_x + 0.68, stair_top_y, MEZZANINE_SURFACE_Z - 0.05),
               0.09, M("Dark_Steel"), coll, 12)
        for side in (-0.72, 0.72):
           cylinder("Mezzanine stair handrail",
                  (stair_x + side, stair_start_y, 1.05),
                  (stair_x + side, stair_top_y, MEZZANINE_SURFACE_Z + 1.05),
                  0.035, M("Safety_Yellow"), coll, 12)
        # Leave a clear 1.7 m opening in the south guardrail for stair access.
        for start_x, end_x in ((x_min, stair_x - 0.9), (stair_x + 0.9, x_max)):
           length = end_x - start_x
           mid_x = (start_x + end_x) / 2
           for z, material, thickness in ((rail_z, "Safety_Yellow", 0.08),
                                    (rail_z - 0.48, "Steel", 0.055),
                                    (MEZZANINE_SURFACE_Z + 0.12, "Safety_Yellow", 0.22)):
              cube("Mezzanine south guardrail", (mid_x, y_min, z),
                  (length, 0.07, thickness), M(material), coll, 0.012)
           for x in (start_x + 0.5, end_x - 0.5):
              cube("Mezzanine south rail post", (x, y_min, rail_z - 0.48),
                  (0.06, 0.06, 1.0), M("Steel"), coll, 0.012)
        # Upper-floor wayfinding and marked maintenance boundary.
        cube("Level two floor marker", (-14.0, 13.2, MEZZANINE_SURFACE_Z + 0.012),
            (5.0, 0.08, 0.02), M("Marking_Yellow"), coll, 0.0)
        add_text("Level two identification", "LEVEL 02 - PRODUCTION", (-16.3, 13.05, MEZZANINE_SURFACE_Z + 0.03),
               (0, 0, 0), 0.22, M("Safety_Black"), coll)
    create_mezzanine()

# -----------------------------------------------------------------------------
# Machine and prop builders
# -----------------------------------------------------------------------------
def create_machine(name, x, y, width=3.4, depth=2.8, height=3.1, kind="cnc", facing=0.0):
    coll_name = "CNC_Machines" if kind == "cnc" else "CNC_Machines"
    if kind == "press":
        coll_name = "CNC_Machines"
    coll = COLLECTIONS[coll_name]
    root = make_empty(name + "_ROOT", (x, y, 0), coll, 'CUBE', 0.3)
    root["randomizable_kind"] = "machine"
    root["base_location"] = [x, y, 0.0]
    MACHINE_ROOTS.append(root)
    before = set(bpy.data.objects)
    paint = M("Machine_Blue" if RNG.random() < 0.65 else "Machine_Green")
    cube(name + " plinth", (x, y, 0.28), (width + 0.18, depth + 0.18, 0.55), M("Dark_Steel"), coll, 0.06)
    cube(name + " cabinet", (x, y, 1.45), (width, depth, 2.2), paint, coll, 0.09)
    if kind == "cnc":
        cube(name + " front work opening", (x, y - depth / 2 - 0.035, 1.66), (width * 0.63, 0.09, 1.15), M("Dark_Steel"), coll, 0.025)
        cube(name + " safety glass", (x, y - depth / 2 - 0.09, 1.75), (width * 0.55, 0.025, 0.72), M("Glass"), coll, 0.008)
        cube(name + " sliding door", (x - width * 0.34, y - depth / 2 - 0.12, 1.75), (0.09, 0.08, 0.88), M("Stainless"), coll, 0.012)
        for dx in (-width * 0.36, width * 0.36):
            cylinder(name + " door guide", (x + dx, y - depth / 2 - 0.08, 1.19),
                     (x + dx, y - depth / 2 - 0.08, 2.28), 0.035, M("Stainless"), coll, 12)
        cube(name + " spindle housing", (x, y + 0.08, 2.55), (0.56, 0.48, 0.3), M("Stainless"), coll, 0.035)
        cylinder(name + " spindle", (x, y - 0.15, 1.98), (x, y - 0.15, 2.35), 0.07, M("Steel"), coll, 16)
    else:
        for dx in (-width * 0.28, width * 0.28):
            cube(name + " press upright", (x + dx, y, 3.0), (0.28, depth * 0.72, 2.7), M("Dark_Steel"), coll, 0.035)
        cube(name + " press crown", (x, y, 4.2), (width, depth * 0.82, 0.55), paint, coll, 0.045)
        cube(name + " press platen", (x, y - 0.1, 2.22), (width * 0.72, depth * 0.6, 0.24), M("Stainless"), coll, 0.025)
        cylinder(name + " hydraulic ram", (x, y, 2.35), (x, y, 3.86), 0.19, M("Stainless"), coll, 20)
    # Side electrical enclosure, status stack, vents, handles and feet.
    px = x + width / 2 + 0.35
    cube(name + " electrical cabinet", (px, y, 1.5), (0.58, 0.8, 1.7), M("Paint_OffWhite"), coll, 0.04)
    cube(name + " operator screen", (px, y - 0.43, 2.15), (0.42, 0.06, 0.34), M("Dark_Steel"), coll, 0.025)
    cube(name + " HMI display", (px, y - 0.47, 2.16), (0.32, 0.025, 0.23), M("Glass"), coll, 0.008)
    for i, color in enumerate(("Emergency_Red", "Safety_Yellow", "Machine_Green")):
        cylinder(name + " stack light", (px, y - 0.44, 2.42 + i * 0.15),
                 (px, y - 0.44, 2.53 + i * 0.15), 0.055, M(color), coll, 12)
    for z in (0.83, 0.98, 1.13):
        cube(name + " ventilation slot", (x, y - depth / 2 - 0.045, z), (width * 0.37, 0.025, 0.035), M("Dark_Steel"), coll, 0.0)
    for dx in (-width * 0.38, width * 0.38):
        for dy in (-depth * 0.35, depth * 0.35):
            cube(name + " leveling foot", (x + dx, y + dy, 0.08), (0.18, 0.18, 0.16), M("Steel"), coll, 0.015)
    root.rotation_euler[2] = facing
    parent_new_objects(root, before)
    return root


def create_conveyor(name, x, y, length=8.0, angle=0.0, z=1.0):
    coll = COLLECTIONS["Conveyors"]
    root = make_empty(name + "_ROOT", (x, y, 0), coll, 'CUBE', 0.25)
    root["randomizable_kind"] = "machine"
    root["base_location"] = [x, y, 0.0]
    MACHINE_ROOTS.append(root)
    before = set(bpy.data.objects)
    c, s = math.cos(angle), math.sin(angle)
    def point(along, across, height):
        return (x + along * c - across * s, y + along * s + across * c, height)
    cube(name + " conveyor frame", point(0, 0, z), (length, 1.05, 0.28), M("Dark_Steel"), coll, 0.045, (0, 0, angle))
    cube(name + " belt", point(0, 0, z + 0.17), (length - 0.12, 0.82, 0.055), M("Rubber"), coll, 0.018, (0, 0, angle))
    count = max(5, int(length / 0.42))
    for i in range(count + 1):
        t = -length / 2 + i * length / count
        p = point(t, 0, z + 0.13)
        roller_start = point(t, -0.42, p[2])
        roller_end = point(t, 0.42, p[2])
        cylinder(name + " roller", roller_start, roller_end, 0.065, M("Stainless"), coll, 12)
    for side in (-0.52, 0.52):
        cube(name + " guard rail", point(0, side, z + 0.34), (length, 0.06, 0.22), M("Safety_Yellow"), coll, 0.02, (0, 0, angle))
    for along in (-length * 0.35, length * 0.35):
        for across in (-0.34, 0.34):
            p = point(along, across, z / 2)
            cube(name + " support leg", p, (0.12, 0.12, z), M("Steel"), coll, 0.015, (0, 0, angle))
    # Drive motor and end drums make the conveyor read as production equipment.
    motor = point(-length / 2 - 0.38, 0, z - 0.1)
    cylinder(name + " drive motor", (motor[0], motor[1], motor[2] - 0.2),
             (motor[0], motor[1], motor[2] + 0.2), 0.27, M("Machine_Blue"), coll, 20)
    parent_new_objects(root, before)
    return root


def create_robotic_arm(name, x, y, facing=0.0):
    coll = COLLECTIONS["Robots"]
    root = make_empty(name + "_ROOT", (x, y, 0), coll, 'CUBE', 0.22)
    root["randomizable_kind"] = "machine"
    root["base_location"] = [x, y, 0.0]
    MACHINE_ROOTS.append(root)
    before = set(bpy.data.objects)
    base = (x, y, 0.45)
    shoulder = (x + 0.15, y, 1.45)
    elbow = (x + 0.9, y - 0.05, 2.45)
    wrist = (x + 1.65, y - 0.2, 2.1)
    cube(name + " robot pedestal", (x, y, 0.25), (0.95, 0.95, 0.5), M("Dark_Steel"), coll, 0.04)
    cylinder(name + " shoulder joint", (x, y, 0.45), (x, y, 1.15), 0.4, M("Safety_Yellow"), coll, 24)
    cylinder(name + " upper arm", shoulder, elbow, 0.22, M("Safety_Yellow"), coll, 20)
    cylinder(name + " forearm", elbow, wrist, 0.17, M("Safety_Yellow"), coll, 20)
    for joint in (shoulder, elbow, wrist):
        cube(name + " joint housing", joint, (0.48, 0.48, 0.48), M("Dark_Steel"), coll, 0.12)
    cylinder(name + " tool spindle", wrist, (wrist[0] + 0.2, wrist[1] - 0.35, wrist[2] - 0.25), 0.065, M("Stainless"), coll, 12)
    root.rotation_euler[2] = facing
    parent_new_objects(root, before)
    return root


def create_storage_rack(name, x, y, width=5.0, depth=1.1, levels=4):
    coll = COLLECTIONS["Storage"]
    root = make_empty(name + "_ROOT", (x, y, 0), coll, 'CUBE', 0.2)
    before = set(bpy.data.objects)
    for dx in (-width / 2, width / 2):
        for dy in (-depth / 2, depth / 2):
            cube(name + " upright", (x + dx, y + dy, 2.55), (0.12, 0.12, 5.1), M("Safety_Yellow"), coll, 0.02)
    for level in range(levels):
        z = 0.65 + level * 1.35
        cube(name + " shelf beam", (x, y, z), (width + 0.2, depth, 0.13), M("Dark_Steel"), coll, 0.018)
        cube(name + " shelf deck", (x, y, z + 0.1), (width - 0.12, depth - 0.08, 0.08), M("Steel"), coll, 0.008)
        if level < levels - 1:
            for item in range(2):
                box_x = x + (item - 0.5) * width * 0.48
                cube(name + " stored tote", (box_x, y, z + 0.48), (width * 0.38, depth * 0.72, 0.62),
                     M("Plastic" if item else "Wood"), coll, 0.025)
    for dx in (-width / 2, width / 2):
        for dz in (1.0, 2.4, 3.8):
            cube(name + " diagonal brace", (x + dx, y, dz), (0.07, depth + 0.05, 1.15), M("Steel"), coll, 0.012,
                 (0.42 if dz == 2.4 else -0.42, 0, 0))
    root["randomizable_kind"] = "prop"
    root["base_location"] = [x, y, 0.0]
    PROP_ROOTS.append(root)
    parent_new_objects(root, before)
    return root


def create_tank(name, x, y, radius=0.85, height=3.5):
    coll = COLLECTIONS["Tanks"]
    root = make_empty(name + "_ROOT", (x, y, 0), coll, 'CUBE', 0.2)
    before = set(bpy.data.objects)
    cylinder(name + " vessel", (x, y, 0.35), (x, y, height), radius, M("Stainless"), coll, 32)
    cylinder(name + " top head", (x, y, height - 0.1), (x, y, height + 0.08), radius * 0.88, M("Steel"), coll, 32)
    for z in (0.55, height * 0.62, height - 0.32):
        cylinder(name + " reinforcing band", (x, y, z), (x, y, z + 0.08), radius * 1.04, M("Dark_Steel"), coll, 32)
    create_pipe((x + radius * 0.7, y, 1.0), (x + radius * 1.7, y, 1.0), 0.085, M("Steel"), coll)
    for dx, dy in ((-0.55, -0.55), (0.55, -0.55), (-0.55, 0.55), (0.55, 0.55)):
        cube(name + " tank leg", (x + dx, y + dy, 0.23), (0.16, 0.16, 0.46), M("Dark_Steel"), coll, 0.015)
    root["randomizable_kind"] = "machine"
    root["base_location"] = [x, y, 0.0]
    MACHINE_ROOTS.append(root)
    parent_new_objects(root, before)
    return root


def create_control_panel(name, x, y, z=1.55, wall_mount=False):
    coll = COLLECTIONS["Control_Panels"]
    before = set(bpy.data.objects)
    cube(name + " enclosure", (x, y, z), (0.9, 0.34, 1.55), M("Paint_OffWhite"), coll, 0.055)
    cube(name + " door", (x, y - 0.19, z), (0.78, 0.07, 1.39), M("Machine_Blue"), coll, 0.04)
    cube(name + " screen bezel", (x - 0.14, y - 0.24, z + 0.28), (0.42, 0.035, 0.32), M("Dark_Steel"), coll, 0.02)
    cube(name + " screen", (x - 0.14, y - 0.264, z + 0.28), (0.34, 0.018, 0.24), M("Glass"), coll, 0.008)
    for i, mat in enumerate(("Emergency_Red", "Safety_Yellow", "Machine_Green", "Safety_Black")):
        cylinder(name + " selector", (x + 0.18, y - 0.27, z + 0.36 - i * 0.22),
                 (x + 0.18, y - 0.34, z + 0.36 - i * 0.22), 0.055, M(mat), coll, 12)
    for i in range(5):
        cube(name + " ventilation louver", (x, y - 0.24, z - 0.43 - i * 0.075), (0.42, 0.025, 0.025), M("Dark_Steel"), coll, 0.004)
    if wall_mount:
        for dx in (-0.32, 0.32):
            cube(name + " wall bracket", (x + dx, y + 0.23, z), (0.08, 0.42, 0.12), M("Steel"), coll, 0.012)
    return list(set(bpy.data.objects) - before)


def create_control_panel_pedestal(name, x, y, panel_center_z, floor_z):
    panel_bottom = panel_center_z - 1.55 / 2
    post_height = panel_bottom - floor_z
    if post_height <= 0:
        raise ValueError(f"{name} panel is at or below its support floor")
    cube(name + " pedestal post", (x, y + 0.27, (panel_bottom + floor_z) / 2),
         (0.16, 0.18, post_height), M("Dark_Steel"), COLLECTIONS["Control_Panels"], 0.018)
    cube(name + " pedestal base plate", (x, y + 0.27, floor_z + 0.05),
         (0.48, 0.48, 0.1), M("Steel"), COLLECTIONS["Control_Panels"], 0.015)
    cube(name + " pedestal head plate", (x, y + 0.27, panel_bottom - 0.035),
         (0.42, 0.36, 0.07), M("Steel"), COLLECTIONS["Control_Panels"], 0.012)


def create_workbench(name, x, y, width=2.2, depth=0.85):
    coll = COLLECTIONS["Tools"]
    root = make_empty(name + "_ROOT", (x, y, 0), coll, 'CUBE', 0.18)
    before = set(bpy.data.objects)
    cube(name + " worktop", (x, y, 1.0), (width, depth, 0.14), M("Wood"), coll, 0.025)
    for dx in (-width * 0.43, width * 0.43):
        for dy in (-depth * 0.38, depth * 0.38):
            cube(name + " leg", (x + dx, y + dy, 0.49), (0.09, 0.09, 0.95), M("Dark_Steel"), coll, 0.015)
    cube(name + " lower stretcher", (x, y, 0.25), (width * 0.85, depth * 0.7, 0.08), M("Steel"), coll, 0.012)
    cube(name + " drawer cabinet", (x + width * 0.2, y, 0.68), (0.72, depth * 0.8, 0.5), M("Machine_Blue"), coll, 0.025)
    for z in (0.58, 0.73):
        cube(name + " drawer front", (x + width * 0.2, y - depth * 0.42, z), (0.61, 0.035, 0.1), M("Steel"), coll, 0.008)
    cube(name + " pegboard", (x, y + depth * 0.55, 1.85), (width * 0.8, 0.08, 1.45), M("Dark_Steel"), coll, 0.02)
    for i in range(9):
        cube(name + " pegboard slot", (x - 0.72 + i * 0.18, y + depth * 0.5, 1.85), (0.035, 0.025, 1.16), M("Steel"), coll, 0.008)
    for i in range(3):
        cylinder(name + " hand tool handle", (x - 0.55 + i * 0.42, y - 0.04, 1.08),
                 (x - 0.55 + i * 0.42, y - 0.04, 1.38), 0.025, M("Safety_Yellow"), coll, 8)
    root["randomizable_kind"] = "prop"
    root["base_location"] = [x, y, 0.0]
    PROP_ROOTS.append(root)
    parent_new_objects(root, before)
    return root


def create_pallet(name, x, y, z=0.08, size=(1.2, 1.0)):
    coll = COLLECTIONS["Pallets"]
    root = make_empty(name + "_ROOT", (x, y, 0), coll, 'CUBE', 0.16)
    before = set(bpy.data.objects)
    width, depth = size
    for i in range(6):
        cube(name + " top deck board", (x - width / 2 + (i + 0.5) * width / 6, y, z + 0.06),
             (width / 6 - 0.025, depth, 0.1), M("Wood"), coll, 0.012)
    for dx in (-width * 0.38, 0, width * 0.38):
        cube(name + " runner", (x + dx, y, z - 0.025), (0.14, depth * 0.92, 0.12), M("Wood"), coll, 0.012)
    root["randomizable_kind"] = "prop"
    root["base_location"] = [x, y, 0.0]
    PROP_ROOTS.append(root)
    parent_new_objects(root, before)
    return root


def create_barrier(name, x, y, length=2.2, angle=0.0):
    coll = COLLECTIONS["Barriers"]
    root = make_empty(name + "_ROOT", (x, y, 0), coll, 'CUBE', 0.18)
    before = set(bpy.data.objects)
    cube(name + " weighted foot", (x, y, 0.12), (0.42, 0.42, 0.24), M("Dark_Steel"), coll, 0.04)
    cube(name + " yellow rail", (x, y, 0.82), (length, 0.12, 0.12), M("Safety_Yellow"), coll, 0.035, (0, 0, angle))
    cube(name + " black warning band", (x, y - 0.065, 0.82), (length * 0.26, 0.018, 0.12), M("Safety_Black"), coll, 0.006, (0, 0, angle))
    for dx in (-length * 0.43, length * 0.43):
        cube(name + " upright", (x + dx, y, 0.48), (0.09, 0.09, 0.72), M("Safety_Yellow"), coll, 0.02)
    root.rotation_euler[2] = angle
    root["randomizable_kind"] = "prop"
    root["base_location"] = [x, y, 0.0]
    PROP_ROOTS.append(root)
    parent_new_objects(root, before)
    return root


def create_crate(name, x, y, width=0.9, depth=0.8, height=0.75, material=None):
    coll = COLLECTIONS["Boxes"]
    root = make_empty(name + "_ROOT", (x, y, 0), coll, 'CUBE', 0.12)
    before = set(bpy.data.objects)
    mat = material or M("Wood")
    cube(name + " crate body", (x, y, height / 2 + 0.05), (width, depth, height), mat, coll, 0.025)
    for z in (0.16, height * 0.48, height - 0.03):
        cube(name + " crate band", (x, y - depth / 2 - 0.012, z), (width * 0.94, 0.025, 0.07), M("Steel"), coll, 0.008)
    cube(name + " shipping label", (x + width * 0.18, y - depth / 2 - 0.025, height * 0.62),
         (width * 0.28, 0.018, height * 0.25), M("Marking_White"), coll, 0.006)
    root["randomizable_kind"] = "prop"
    root["base_location"] = [x, y, 0.0]
    PROP_ROOTS.append(root)
    parent_new_objects(root, before)
    return root


def create_fire_extinguisher(name, x, y, z=0.8, coll=None):
    coll = coll or COLLECTIONS["Safety_Features"]
    cylinder(name + " cylinder", (x, y, z - 0.48), (x, y, z + 0.48), 0.16, M("Emergency_Red"), coll, 20)
    cube(name + " bracket", (x, y + 0.17, z), (0.36, 0.08, 0.62), M("Dark_Steel"), coll, 0.02)
    cylinder(name + " valve", (x, y, z + 0.48), (x, y, z + 0.59), 0.07, M("Steel"), coll, 12)
    create_pipe((x + 0.08, y, z + 0.55), (x + 0.32, y, z + 0.25), 0.025, M("Rubber"), coll)
    cube(name + " instruction label", (x, y - 0.16, z + 0.05), (0.17, 0.025, 0.28), M("Marking_White"), coll, 0.005)


def create_machinery():
    for name, x, y, kind, size in [
        ("CNC_01", -20, -11.5, "cnc", (3.7, 3.1, 3.2)),
        ("CNC_02", -11, -11.5, "cnc", (3.2, 2.8, 3.0)),
        ("CNC_03", 7, -11.3, "cnc", (4.0, 3.2, 3.3)),
        ("CNC_04", 17, -11.2, "press", (3.5, 3.2, 4.4)),
        ("CNC_05", -20, 10.8, "press", (3.3, 3.0, 4.2)),
        ("CNC_06", -10.5, 10.7, "cnc", (3.8, 3.0, 3.2)),
        ("CNC_07", 8, 10.8, "cnc", (3.5, 3.2, 3.1)),
        ("CNC_08", 18, 10.8, "cnc", (3.0, 2.8, 2.9)),
    ]:
        create_machine(name, x, y, *size, kind=kind, facing=math.pi if y > 0 else 0)
    create_conveyor("Main assembly conveyor", -1.0, -1.2, 15.0, 0.0, 1.05)
    create_conveyor("Transfer conveyor west", -10.5, 1.4, 7.0, math.pi / 2, 0.95)
    create_conveyor("Transfer conveyor east", 9.0, 1.5, 7.5, math.pi / 2, 1.0)
    create_conveyor("Packing conveyor", 19.0, 2.0, 7.0, math.pi / 2, 0.92)
    create_robotic_arm("Robot_01", -5.4, -1.0, 0.0)
    create_robotic_arm("Robot_02", 5.2, -1.1, math.pi)
    create_robotic_arm("Robot_03", 14.4, 2.1, math.pi / 2)
    create_tank("Coolant tank A", 25.0, -8.0, 0.85, 3.6)
    create_tank("Hydraulic reservoir", 25.0, -3.8, 0.7, 2.8)
    create_tank("Process tank", -26.0, 3.2, 1.15, 4.4)
    create_storage_rack("Raw stock rack north", -24, 6.8, 6.2, 1.2, 4)
    create_storage_rack("Finished goods rack east", 25.0, 9.3, 5.6, 1.2, 4)
    for x in (-25.5, -2.5, 3.0, 23.8):
        create_control_panel("Line control panel", x, -18.95, 1.75, True)


def create_mezzanine_machinery():
    """Add a compact second production cell without exceeding the roof clearance."""
    deck_z = MEZZANINE_SURFACE_Z
    for name, x, y, width, depth in (
        ("Upper CNC cell A", -22.0, 7.4, 3.0, 2.5),
        ("Upper CNC cell B", -14.0, 7.4, 3.2, 2.6),
        ("Upper CNC cell C", -6.0, 7.4, 3.0, 2.5),
    ):
        root = create_machine(name, x, y, width, depth, 2.8, kind="cnc", facing=math.pi)
        set_root_elevation(root, deck_z)
    conveyor = create_conveyor("Upper inspection conveyor", -14.0, 11.6, 9.0, 0.0, 0.9)
    set_root_elevation(conveyor, deck_z)
    bench = create_workbench("Upper maintenance bench", -22.0, 12.0, 2.2, 0.8)
    set_root_elevation(bench, deck_z)
    barrier = create_barrier("Upper cell safety barrier", -9.2, 5.1, 2.2)
    set_root_elevation(barrier, deck_z)
    create_control_panel("Upper production control panel", -3.5, 12.7, deck_z + 1.65)
    create_control_panel_pedestal("Upper production control panel", -3.5, 12.7,
                                  deck_z + 1.65, deck_z)
    pallet = create_pallet("Upper parts pallet", -5.0, 12.4, size=(1.0, 0.85))
    set_root_elevation(pallet, deck_z)
    crate = create_crate("Upper parts crate", -5.0, 12.4, 0.7, 0.65, 0.55)
    set_root_elevation(crate, deck_z)


def create_props():
    for name, x, y in [("Bench A", -17.0, -4.0), ("Bench B", 2.3, -5.0), ("Bench C", 21.0, 5.4)]:
        create_workbench(name, x, y, 2.5 if name != "Bench B" else 2.0, 0.9)
    for i, (x, y) in enumerate([(-23, -5.2), (-15, -6.0), (-5, 5.0), (3, 6.2), (12, -5.4), (22, -5.3), (25, 5.5)]):
        create_pallet("Timber pallet", x, y, size=(1.25, 1.05))
        if i in (0, 2, 4, 6):
            create_crate("Parts crate", x, y, 0.85, 0.78, RNG.uniform(0.55, 0.9), M("Wood"))
    for x, y, angle in [(-18.5, -7.3, 0), (-7.0, -7.7, 0), (11.2, -7.4, 0), (20.0, 7.0, math.pi / 2), (-1.0, 7.0, 0)]:
        create_barrier("Machine perimeter barrier", x, y, RNG.uniform(1.8, 2.7), angle)
    # Reusable industrial bins, waste containers, air compressors and tool cabinets.
    for i, (x, y) in enumerate([(-27, -1.5), (27, 1.8), (-3.0, 16.8)]):
        cube("Steel scrap bin", (x, y, 0.62), (1.1, 0.9, 1.2), M("Dark_Steel"), COLLECTIONS["Containers"], 0.045)
        cube("Bin rim", (x, y, 1.25), (1.18, 0.98, 0.09), M("Steel"), COLLECTIONS["Containers"], 0.018)
        if i == 1:
            cube("Compressor receiver", (x - 1.1, y + 1.2, 0.7), (1.8, 0.72, 1.2), M("Machine_Blue"), COLLECTIONS["Tools"], 0.09)
            cylinder("Compressor motor", (x - 1.1, y + 1.2, 1.3), (x - 1.1, y + 1.2, 1.75), 0.32, M("Dark_Steel"), COLLECTIONS["Tools"], 20)
    for x, y in [(-28, -19.0), (0, -19.0), (28, -19.0), (-29.0, 6.0), (29.0, -10.0)]:
        create_fire_extinguisher("Portable fire extinguisher", x, y, 0.9)
    safety = COLLECTIONS["Safety_Features"]
    for x, y in [(-13.0, -9.0), (1.0, 9.0), (22.5, -1.5)]:
        cube("Emergency light housing", (x, y, 8.9), (0.7, 0.24, 0.22), M("Emergency_Red"), safety, 0.03)
        cube("Emergency light lens", (x, y - 0.14, 8.88), (0.52, 0.025, 0.08), M("LED"), safety, 0.018)
    # Catwalk across the rear process zone, with handrails and access stairs.
    cat = COLLECTIONS["Safety_Features"]
    cube("Service catwalk deck", (-1.5, 16.0, 5.4), (23.0, 2.0, 0.18), M("Dark_Steel"), cat, 0.02)
    for x in (-13.0, 10.0):
        cube("Catwalk support", (x, 16.0, 2.7), (0.2, 0.2, 5.4), M("Steel"), cat, 0.02)
    for y in (15.05, 16.95):
        cube("Catwalk top rail", (-1.5, y, 6.55), (23.0, 0.07, 0.08), M("Safety_Yellow"), cat, 0.018)
        cube("Catwalk mid rail", (-1.5, y, 5.95), (23.0, 0.05, 0.06), M("Steel"), cat, 0.012)
        cube("Catwalk toe board", (-1.5, y, 5.53), (23.0, 0.06, 0.22), M("Safety_Yellow"), cat, 0.012)
    for x in range(-12, 10, 1):
        cube("Catwalk rail upright", (x, 15.05, 6.1), (0.055, 0.055, 0.95), M("Steel"), cat, 0.012)
        cube("Catwalk rail upright", (x, 16.95, 6.1), (0.055, 0.055, 0.95), M("Steel"), cat, 0.012)
    # Industrial stair flight from floor to catwalk; treads and twin stringers.
    for i in range(13):
        xx = 11.0 + i * 0.32
        zz = 0.24 + i * 0.42
        cube("Steel stair tread", (xx, 16.0, zz), (0.42, 1.3, 0.09), M("Steel"), cat, 0.012)
    cylinder("Stair stringer near", (10.7, 15.45, 0.1), (15.2, 15.45, 5.35), 0.09, M("Dark_Steel"), cat, 12)
    cylinder("Stair stringer far", (10.7, 16.55, 0.1), (15.2, 16.55, 5.35), 0.09, M("Dark_Steel"), cat, 12)
    for y in (15.32, 16.68):
        cylinder("Stair handrail", (10.7, y, 1.05), (15.2, y, 6.1), 0.035, M("Safety_Yellow"), cat, 12)

# -----------------------------------------------------------------------------
# Worker placeholders and dataset tagging
# -----------------------------------------------------------------------------
def bind_existing_scene():
    """Rebuild script registries from a generated scene already open in Blender."""
    COLLECTIONS.clear()
    COLLECTIONS.update({coll.name: coll for coll in bpy.data.collections})
    MATERIALS.clear()
    MATERIALS.update({mat.name: mat for mat in bpy.data.materials})
    MACHINE_ROOTS[:] = [obj for obj in bpy.data.objects
                        if obj.get("randomizable_kind") == "machine"]
    PROP_ROOTS[:] = [obj for obj in bpy.data.objects
                     if obj.get("randomizable_kind") == "prop"]
    CAMERA_BASELINES.clear()
    LIGHT_BASELINES.clear()
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            CAMERA_BASELINES[obj.name] = (obj.location.copy(), obj.rotation_euler.copy())
        elif obj.type == 'LIGHT':
            LIGHT_BASELINES[obj.name] = (obj.location.copy(), obj.rotation_euler.copy(),
                                         obj.data.energy, obj.data.color[:])
    for instances in LABEL_OBJECTS.values():
        instances.clear()
    for obj in bpy.data.objects:
        if obj.type not in {'MESH', 'ARMATURE'} or "synthetic_class" not in obj or "worker_key" not in obj:
            continue
        class_id = int(obj["synthetic_class"])
        if obj.type == 'ARMATURE' and class_id != 0:
            continue
        if class_id in LABEL_OBJECTS:
            LABEL_OBJECTS[class_id].setdefault(obj["worker_key"], []).append(obj)


def register_worker_asset(worker_root, body_objects, helmet_objects=None, vest_objects=None):
    """Tag imported mesh objects for randomization, PPE toggles and YOLO masks.

    Import the realistic rig and PPE, parent the mesh objects beneath the matching
    Worker_XX / Helmet_XX / SafetyVest_XX empties, then call this function with
    the body, helmet and vest mesh objects. Keep PPE meshes separate from body meshes.
    The worker body gets class 0, helmet class 1, and vest class 2.
    """
    helmet_objects = helmet_objects or []
    vest_objects = vest_objects or []
    if "Worker_PPE" not in COLLECTIONS:
        bind_existing_scene()
    worker_key = worker_root.name
    ppe_collection = COLLECTIONS["Worker_PPE"]
    suffix = worker_key.split("_")[-1]
    helmet_anchor = bpy.data.objects.get(worker_root.get("helmet_anchor", "Helmet_" + suffix))
    vest_anchor = bpy.data.objects.get(worker_root.get("vest_anchor", "SafetyVest_" + suffix))

    def attach_asset_hierarchy(objects, parent):
        object_set = set(objects)
        roots = [obj for obj in objects if obj.parent not in object_set]
        for obj in roots:
            world_matrix = obj.matrix_world.copy()
            obj.parent = parent
            obj.matrix_parent_inverse = parent.matrix_world.inverted()
            obj.matrix_world = world_matrix

    attach_asset_hierarchy(body_objects, worker_root)
    if helmet_anchor is None:
        helmet_anchor = make_empty("Helmet_" + worker_key.split("_")[-1], (0, 0, 1.72), ppe_collection, 'SPHERE', 0.27)
        helmet_anchor.parent = worker_root
    if vest_anchor is None:
        vest_anchor = make_empty("SafetyVest_" + worker_key.split("_")[-1], (0, 0, 1.15), ppe_collection, 'CUBE', 0.4)
        vest_anchor.parent = worker_root
    attach_asset_hierarchy(helmet_objects, helmet_anchor)
    attach_asset_hierarchy(vest_objects, vest_anchor)
    for obj in body_objects:
        if obj.type != 'ARMATURE':
            continue
        for pose_bone in obj.pose.bones:
            pose_bone.rotation_mode = 'QUATERNION'
            if "factory_base_rotation" not in pose_bone:
                pose_bone["factory_base_rotation"] = list(pose_bone.rotation_quaternion)
    worker_root["synthetic_worker"] = True
    worker_root["worker_key"] = worker_key
    worker_root["base_location"] = list(worker_root.location)
    for obj in body_objects:
        obj["synthetic_class"] = 0
        obj["worker_key"] = worker_key
        obj["asset_part"] = "body"
        LABEL_OBJECTS[0].setdefault(worker_key, []).append(obj)
    for obj in helmet_objects:
        obj["synthetic_class"] = 1
        obj["worker_key"] = worker_key
        obj["asset_part"] = "helmet"
        LABEL_OBJECTS[1].setdefault(worker_key, []).append(obj)
    for obj in vest_objects:
        obj["synthetic_class"] = 2
        obj["worker_key"] = worker_key
        obj["asset_part"] = "vest"
        LABEL_OBJECTS[2].setdefault(worker_key, []).append(obj)


def duplicate_worker_to_anchor(source_root, target_root):
    """Duplicate the same rig and meshes at a second worker anchor."""
    source_objects = list(LABEL_OBJECTS[0].get(source_root.name, []))
    if not source_objects:
        return None
    source_world = {obj: obj.matrix_world.copy() for obj in source_objects}
    duplicates = {}
    worker_collection = COLLECTIONS["Worker_Models"]
    offset = target_root.matrix_world.translation - source_root.matrix_world.translation

    for source in source_objects:
        duplicate = source.copy()
        if source.data:
            duplicate.data = source.data
        worker_collection.objects.link(duplicate)
        duplicates[source] = duplicate

    for source, duplicate in duplicates.items():
        if source.parent == source_root:
            duplicate.parent = target_root
        else:
            duplicate.parent = duplicates.get(source.parent, source.parent)
        world_matrix = source_world[source].copy()
        world_matrix.translation += offset
        duplicate.matrix_world = world_matrix
        for modifier in duplicate.modifiers:
            if modifier.type == 'ARMATURE' and modifier.object in duplicates:
                modifier.object = duplicates[modifier.object]
        for constraint in duplicate.constraints:
            if constraint.target in duplicates:
                constraint.target = duplicates[constraint.target]

    duplicate_rig = duplicates.get(next((obj for obj in source_objects if obj.type == 'ARMATURE'), None))
    duplicate_meshes = [duplicates[obj] for obj in source_objects if obj.type == 'MESH']
    register_worker_asset(target_root, [duplicate_rig] + duplicate_meshes, [], [])
    target_root["asset_source"] = source_root.get("asset_source", WORKER_FBX_PATH)
    return duplicate_rig


def apply_worker_pose(rig, pose_index, phase=0.0):
    """Apply a repeatable small work-pose cycle, based on the imported rest pose."""
    for pose_bone in rig.pose.bones:
        base_rotation = pose_bone.get("factory_base_rotation")
        if base_rotation:
            pose_bone.rotation_mode = 'QUATERNION'
            pose_bone.rotation_quaternion = base_rotation

    profiles = (
        ("relaxed", -1.08, 1.08, 0.02, -0.02, 0.0, 0.0),
        ("walking", -1.08, 1.08, 0.11, -0.11, -0.13, 0.13),
        ("walking_stride", -1.08, 1.08, -0.11, 0.11, 0.13, -0.13),
        ("light_work", -1.02, 1.02, 0.06, -0.05, -0.05, 0.05),
    )
    name, left_arm_z, right_arm_z, left_swing, right_swing, left_leg_swing, right_leg_swing = profiles[pose_index % len(profiles)]
    pose_bone_offsets = {
        "LeftArm": (left_swing, 0.0, left_arm_z),
        "RightArm": (right_swing, 0.0, right_arm_z),
        "LeftForeArm": (0.05 if pose_index % 4 == 3 else 0.0, 0.0, 0.08 if pose_index % 4 == 3 else 0.0),
        "RightForeArm": (-0.05 if pose_index % 4 == 3 else 0.0, 0.0, -0.08 if pose_index % 4 == 3 else 0.0),
        "LeftUpLeg": (left_leg_swing, 0.0, 0.0),
        "RightUpLeg": (right_leg_swing, 0.0, 0.0),
        "Spine1": (0.0, 0.025 * math.sin(phase), 0.0),
    }
    for bone_name, angles in pose_bone_offsets.items():
        pose_bone = rig.pose.bones.get(bone_name)
        if pose_bone is None:
            continue
        base_rotation = pose_bone.get("factory_base_rotation", (1.0, 0.0, 0.0, 0.0))
        pose_bone.rotation_mode = 'QUATERNION'
        pose_bone.rotation_quaternion = Euler(angles, 'XYZ').to_quaternion() @ Quaternion(base_rotation)
    rig["factory_pose"] = name


def assign_worker_fallback_material(obj, material_name, color, roughness):
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = make_material(material_name, color, metallic=0.0, roughness=roughness)
    else:
        material.diffuse_color = (*color, 1.0)
        material.use_nodes = True
        shader = material.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (*color, 1.0)
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = 0.0
    MATERIALS[material_name] = material
    obj.data.materials.clear()
    obj.data.materials.append(material)


def import_ppe_fbx(filepath, worker_root, ppe_kind):
    """Import and fit one separate helmet or vest asset to its worker anchor."""
    if not os.path.isfile(filepath):
        print(f"{ppe_kind.title()} FBX not found; skipped import: {filepath}")
        return []
    anchor_key = "helmet_anchor" if ppe_kind == "helmet" else "vest_anchor"
    default_anchor = "Helmet_" if ppe_kind == "helmet" else "SafetyVest_"
    anchor_name = worker_root.get(anchor_key, default_anchor + worker_root.name.split("_")[-1])
    anchor = bpy.data.objects.get(anchor_name)
    if anchor is None:
        raise RuntimeError(f"{ppe_kind.title()} anchor does not exist: {anchor_name}")

    before_import = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=filepath, use_anim=False)
    imported_objects = set(bpy.data.objects) - before_import
    mesh_objects = [obj for obj in imported_objects if obj.type == 'MESH']
    if not mesh_objects:
        raise RuntimeError(f"No mesh found in PPE FBX: {filepath}")
    for obj in imported_objects:
        move_to_collection(obj, COLLECTIONS["Worker_PPE"])

    depsgraph = bpy.context.evaluated_depsgraph_get()

    def mesh_bounds():
        bpy.context.view_layer.update()
        points = []
        for mesh_obj in mesh_objects:
            evaluated = mesh_obj.evaluated_get(depsgraph)
            points.extend(evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box)
        minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
        maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
        return minimum, maximum

    roots = [obj for obj in imported_objects if obj.parent not in imported_objects]
    minimum, maximum = mesh_bounds()
    source_height = maximum.z - minimum.z
    target_height = HELMET_TARGET_HEIGHT if ppe_kind == "helmet" else VEST_TARGET_HEIGHT
    if source_height <= 0.001:
        raise RuntimeError(f"Invalid {ppe_kind} height in {filepath}: {source_height}")
    factor = target_height / source_height
    source_center = (minimum + maximum) * 0.5
    target_center = anchor.matrix_world.translation.copy()
    if ppe_kind == "helmet":
        target_center.z += 0.025
    anchor_rotation = anchor.matrix_world.to_quaternion()
    transform = (Matrix.Translation(target_center)
                 @ anchor_rotation.to_matrix().to_4x4()
                 @ Matrix.Scale(factor, 4)
                 @ Matrix.Translation(-source_center))
    for root in roots:
        root.matrix_world = transform @ root.matrix_world
    bpy.context.view_layer.update()

    for root in roots:
        world_matrix = root.matrix_world.copy()
        root.parent = anchor
        root.matrix_parent_inverse = anchor.matrix_world.inverted()
        root.matrix_world = world_matrix
    bpy.context.view_layer.update()

    if ppe_kind == "helmet":
        helmet_mat = make_material("Synthetic_Helmet_Yellow", (0.96, 0.62, 0.035), 0.05, 0.34)
        strap_mat = make_material("Synthetic_Helmet_Straps", (0.045, 0.052, 0.055), 0.0, 0.72)
        for mesh_obj in mesh_objects:
            mat = strap_mat if "belt" in mesh_obj.name.lower() else helmet_mat
            mesh_obj.data.materials.clear()
            mesh_obj.data.materials.append(mat)
    else:
        vest_mat = make_material("Synthetic_HighVis_Vest", (0.72, 0.9, 0.045), 0.0, 0.46)
        for mesh_obj in mesh_objects:
            mesh_obj.data.materials.clear()
            mesh_obj.data.materials.append(vest_mat)

    class_id = 1 if ppe_kind == "helmet" else 2
    label_instances = LABEL_OBJECTS[class_id].setdefault(worker_root.name, [])
    for mesh_obj in mesh_objects:
        mesh_obj["synthetic_class"] = class_id
        mesh_obj["worker_key"] = worker_root.name
        mesh_obj["asset_part"] = ppe_kind
        mesh_obj["ppe_kind"] = ppe_kind
        mesh_obj.hide_render = False
        if mesh_obj not in label_instances:
            label_instances.append(mesh_obj)
    print(f"Imported {ppe_kind} for {worker_root.name}: fitted to {target_height:.2f} m target height")
    return mesh_objects


def face_worker_toward_equipment(worker_root, x=None, y=None, jitter=0.0):
    """Orient the worker's local forward (-Y) toward the nearest machine on their level."""
    x = worker_root.location.x if x is None else x
    y = worker_root.location.y if y is None else y
    worker_z = worker_root.location.z
    same_level = [
        root for root in MACHINE_ROOTS
        if abs(float(root.get("base_location", list(root.location))[2]) - worker_z) < 1.0
    ]
    if not same_level:
        same_level = MACHINE_ROOTS
    if not same_level:
        return
    target = min(
        same_level,
        key=lambda root: (float(root.get("base_location", list(root.location))[0]) - x) ** 2
                         + (float(root.get("base_location", list(root.location))[1]) - y) ** 2,
    )
    target_base = target.get("base_location", list(target.location))
    dx = float(target_base[0]) - x
    dy = float(target_base[1]) - y
    if dx * dx + dy * dy < 0.04:
        return
    worker_root.rotation_euler.z = math.atan2(dx, -dy) + jitter


def import_worker_fbx(worker_name="Worker_01"):
    """Import the configured FBX, fit it to its floor marker, and register it."""
    if not AUTO_IMPORT_WORKER:
        print("Worker FBX auto-import disabled")
        return None
    if not os.path.isfile(WORKER_FBX_PATH):
        print(f"Worker FBX not found; skipped import: {WORKER_FBX_PATH}")
        return None

    worker_root = bpy.data.objects.get(worker_name)
    if worker_root is None:
        raise RuntimeError(f"Worker anchor does not exist: {worker_name}")
    before_import = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=WORKER_FBX_PATH, use_anim=True)
    imported_objects = set(bpy.data.objects) - before_import
    armatures = [obj for obj in imported_objects if obj.type == 'ARMATURE']
    if len(armatures) != 1:
        raise RuntimeError(f"Expected one armature in {WORKER_FBX_PATH}, found {len(armatures)}")
    rig = armatures[0]
    body_meshes = [obj for obj in imported_objects if obj.type == 'MESH' and obj.parent == rig]
    if not body_meshes:
        raise RuntimeError(f"No skinned body meshes found in {WORKER_FBX_PATH}")

    worker_models = COLLECTIONS["Worker_Models"]
    for obj in imported_objects:
        move_to_collection(obj, worker_models)
        if obj.type == 'MESH' and obj.parent != rig:
            obj.hide_render = True
            obj.hide_set(True)
            print(f"Excluded unparented FBX mesh: {obj.name}")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    bpy.context.view_layer.update()

    def mesh_bounds(mesh_objects):
        points = []
        for mesh_obj in mesh_objects:
            evaluated = mesh_obj.evaluated_get(depsgraph)
            points.extend(evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box)
        minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
        maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
        return minimum, maximum

    minimum, maximum = mesh_bounds(body_meshes)
    original_height = maximum.z - minimum.z
    if original_height <= 0:
        raise RuntimeError(f"Imported worker has invalid height: {original_height}")
    fit_scale = WORKER_TARGET_HEIGHT / original_height
    rig.scale = tuple(component * fit_scale for component in rig.scale)
    bpy.context.view_layer.update()
    minimum, maximum = mesh_bounds(body_meshes)

    anchor = worker_root.matrix_world.translation.copy()
    rig.location.x += anchor.x - (minimum.x + maximum.x) * 0.5
    rig.location.y += anchor.y - (minimum.y + maximum.y) * 0.5
    rig.location.z += anchor.z - minimum.z
    bpy.context.view_layer.update()

    fallback_materials = {
        "male_worksuit01Mesh": ("FBX_Workwear_Navy", (0.055, 0.085, 0.11), 0.82),
        "short02Mesh": ("FBX_Workwear_Trousers", (0.10, 0.12, 0.13), 0.86),
        "shoes05Mesh": ("FBX_Work_Shoes", (0.025, 0.03, 0.032), 0.78),
        "worker_1Mesh": ("FBX_Skin", (0.43, 0.24, 0.16), 0.62),
        "eyebrow001Mesh": ("FBX_Hair_Brows", (0.075, 0.035, 0.02), 0.88),
        "high-polyMesh": ("FBX_Eyes", (0.045, 0.025, 0.015), 0.35),
    }
    for mesh_obj in body_meshes:
        fallback = fallback_materials.get(mesh_obj.name)
        if fallback:
            assign_worker_fallback_material(mesh_obj, *fallback)

    register_worker_asset(worker_root, [rig] + body_meshes, [], [])
    final_minimum, final_maximum = mesh_bounds(body_meshes)
    worker_root["asset_source"] = WORKER_FBX_PATH
    print(f"Imported worker {rig.name}: height {final_maximum.z - final_minimum.z:.2f} m at {worker_name}")
    print("Worker body meshes:", ", ".join(mesh_obj.name for mesh_obj in body_meshes))
    print("No helmet/vest meshes found in FBX; PPE classes need separate assets")

    for target_root in sorted(
        (obj for obj in bpy.data.objects
         if obj.get("synthetic_worker") and obj != worker_root),
        key=lambda obj: obj.name,
    ):
        duplicate_worker_to_anchor(worker_root, target_root)
    for anchor_name, objects in LABEL_OBJECTS[0].items():
        target_root = bpy.data.objects.get(anchor_name)
        if target_root:
            face_worker_toward_equipment(target_root)
        armature = next((obj for obj in objects if obj.type == 'ARMATURE'), None)
        if armature:
            apply_worker_pose(armature, 0)
    if AUTO_IMPORT_PPE:
        for target_root in sorted(
            (obj for obj in bpy.data.objects if obj.get("synthetic_worker")),
            key=lambda obj: obj.name,
        ):
            import_ppe_fbx(HELMET_FBX_PATH, target_root, "helmet")
            import_ppe_fbx(VEST_FBX_PATH, target_root, "vest")
    print(f"Duplicated the same worker to {len(LABEL_OBJECTS[0])} worker anchors")
    return rig


def setup_worker_placeholders():
    workers = COLLECTIONS["SYNTHETIC_WORKERS"]
    models = COLLECTIONS["Worker_Models"]
    ppe = COLLECTIONS["Worker_PPE"]
    # Clear placeholder registrations if the script is rerun in the same process.
    for index in range(1, NUM_WORKERS + 1):
        suffix = f"{index:02d}"
        x = [-3.5, 2.0, 11.5, -14.0, 18.0][(index - 1) % 5]
        y = [-2.8, 3.0, -4.2, 4.0, 0.2][(index - 1) % 5]
        root = make_empty("Worker_" + suffix, (x, y, 0), models, 'CIRCLE', 0.9)
        root["synthetic_worker"] = True
        root["level"] = 1
        root["worker_key"] = root.name
        root["base_location"] = [x, y, 0.0]
        # Separate non-rendering insertion anchors: replace/parent actual imported assets here.
        helmet = make_empty("Helmet_" + suffix, (0, 0, 1.72), ppe, 'SPHERE', 0.27)
        helmet.parent = root
        helmet["ppe_kind"] = "helmet"
        helmet["worker_key"] = root.name
        helmet["placeholder_only"] = True
        vest = make_empty("SafetyVest_" + suffix, (0, 0, 1.15), ppe, 'CUBE', 0.4)
        vest.parent = root
        vest["ppe_kind"] = "vest"
        vest["worker_key"] = root.name
        vest["placeholder_only"] = True
        # A marked floor footprint makes the import zone apparent in the viewport.
        cube("Worker import zone " + suffix, (x, y, 0.018), (2.5, 2.5, 0.025),
             M("Marking_White"), COLLECTIONS["Safety_Features"], 0.0)
        root["helmet_anchor"] = helmet.name
        root["vest_anchor"] = vest.name
    # 'workers' is a top-level organization collection, with model/PPE subcollections.
    return workers


def setup_upper_floor_workers():
    """Create three separate worker/PPE import anchors on the level-two deck."""
    models = COLLECTIONS["Worker_Models"]
    ppe = COLLECTIONS["Worker_PPE"]
    positions = ((-23.0, 11.0), (-14.0, 5.7), (-6.0, 11.7))
    for index in range(1, NUM_UPPER_FLOOR_WORKERS + 1):
        suffix = f"{index:02d}"
        worker_name = "Worker_Upper_" + suffix
        x, y = positions[(index - 1) % len(positions)]
        root = make_empty(worker_name, (x, y, MEZZANINE_SURFACE_Z), models, 'CIRCLE', 0.9)
        root["synthetic_worker"] = True
        root["worker_key"] = root.name
        root["level"] = 2
        root["base_location"] = [x, y, MEZZANINE_SURFACE_Z]
        helmet_name = "Helmet_Upper_" + suffix
        vest_name = "SafetyVest_Upper_" + suffix
        helmet = make_empty(helmet_name, (0, 0, 1.72), ppe, 'SPHERE', 0.27)
        helmet.parent = root
        helmet["ppe_kind"] = "helmet"
        helmet["worker_key"] = root.name
        helmet["placeholder_only"] = True
        vest = make_empty(vest_name, (0, 0, 1.15), ppe, 'CUBE', 0.4)
        vest.parent = root
        vest["ppe_kind"] = "vest"
        vest["worker_key"] = root.name
        vest["placeholder_only"] = True
        root["helmet_anchor"] = helmet_name
        root["vest_anchor"] = vest_name
        cube("Upper worker import zone " + suffix, (x, y, MEZZANINE_SURFACE_Z + 0.018),
             (2.2, 2.2, 0.025), M("Marking_White"), COLLECTIONS["Mezzanine_Level_02"], 0.0)

# -----------------------------------------------------------------------------
# Lighting, camera, randomization
# -----------------------------------------------------------------------------
def aim_at(obj, point):
    direction = Vector(point) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def create_area_light(name, location, target, energy, color=(0.82, 0.88, 1.0), size=5.0):
    data = bpy.data.lights.new(name, 'AREA')
    data.energy = energy
    data.color = color
    data.shape = 'RECTANGLE'
    data.size = size
    data.size_y = size * 0.48
    obj = bpy.data.objects.new(name, data)
    COLLECTIONS["Lights"].objects.link(obj)
    obj.location = location
    aim_at(obj, target)
    LIGHT_BASELINES[obj.name] = (obj.location.copy(), obj.rotation_euler.copy(), data.energy, data.color[:])
    return obj


def create_lighting():
    scene = bpy.context.scene
    world = bpy.data.worlds.new("Factory ambient world") if not bpy.data.worlds else bpy.data.worlds[0]
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.34, 0.39, 0.43, 1)
    bg.inputs["Strength"].default_value = 0.32
    # Cool high-bay industrial luminaires distributed above work cells.
    for x in (-22, -11, 0, 11, 22):
        for y in (-12, 0, 12):
            cube("High-bay LED housing", (x, y, 10.85), (2.4, 0.75, 0.16), M("Dark_Steel"), COLLECTIONS["Ceiling"], 0.045)
            cube("High-bay diffuser", (x, y, 10.75), (2.1, 0.58, 0.045), M("LED"), COLLECTIONS["Ceiling"], 0.025)
            create_area_light("High bay area", (x, y, 10.55), (x, y, 0.5), 1150, (0.82, 0.88, 1.0), 3.5)
    create_area_light("North window daylight", (0, 17.6, 8.7), (0, -2, 1.0), 2600, (0.72, 0.83, 1.0), 12.0)
    create_area_light("West fill", (-27, 0, 8.0), (0, 0, 2.0), 1500, (1.0, 0.82, 0.63), 9.0)
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'AgX - Medium High Contrast'
    scene.view_settings.exposure = -0.2
    scene.view_settings.gamma = 1.0
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.resolution_percentage = RENDER_PERCENTAGE
    scene.render.use_file_extension = True
    if USE_CYCLES:
        scene.render.engine = 'CYCLES'
    else:
        engines = scene.render.bl_rna.properties["engine"].enum_items.keys()
        scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engines else 'BLENDER_EEVEE'
    if USE_CYCLES:
        scene.cycles.samples = 32
        scene.cycles.use_denoising = True
    else:
        if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = 32
    scene.render.resolution_percentage = RENDER_PERCENTAGE


def create_camera(name, location, target, lens=30.0, sensor_width=36.0):
    data = bpy.data.cameras.new(name)
    data.lens = lens
    data.sensor_width = sensor_width
    data.clip_start = 0.1
    data.clip_end = 180.0
    obj = bpy.data.objects.new(name, data)
    COLLECTIONS["Cameras"].objects.link(obj)
    obj.location = location
    aim_at(obj, target)
    CAMERA_BASELINES[name] = (obj.location.copy(), obj.rotation_euler.copy())
    return obj


def create_cameras():
    cameras = [
        create_camera("Camera_Main_Wide", (-27.0, -17.0, 8.1), (0.0, 0.0, 3.1), 26),
        create_camera("Camera_Worker_Level", (-13.0, -5.8, 1.72), (2.0, 0.0, 1.6), 29),
        create_camera("Camera_Conveyor_CCTV", (12.5, -7.0, 5.8), (0.0, 0.0, 1.1), 36),
        create_camera("Camera_Machine_Area", (-2.0, 13.8, 4.2), (0.0, 0.0, 1.4), 32),
        create_camera("Camera_High_Angle", (25.0, -14.0, 10.6), (1.0, 1.0, 1.0), 28),
        create_camera("Camera_Upper_Floor_CCTV", (-25.0, -2.5, 10.0), (-13.0, 9.5, 7.4), 34),
    ]
    bpy.context.scene.camera = cameras[0]


def setup_randomization(image_index):
    RNG.seed(SEED + image_index)
    for root in MACHINE_ROOTS:
        base = root.get("base_location", list(root.location))
        root.location = base
        root.rotation_euler[2] = root.get("base_rotation_z", root.rotation_euler[2])
        if RANDOMIZE_MACHINERY:
            # Restrained jitter only; keeps equipment on its aisle and does not break attachments.
            root.location.x += RNG.uniform(-0.18, 0.18)
            root.location.y += RNG.uniform(-0.14, 0.14)
            root.rotation_euler[2] += RNG.uniform(-0.025, 0.025)
    for root in PROP_ROOTS:
        base = root.get("base_location", list(root.location))
        root.location = base
        hidden = RANDOMIZE_BACKGROUND_OBJECTS and RNG.random() < BACKGROUND_HIDE_PROBABILITY
        root.hide_render = hidden
        for child in root.children_recursive:
            child.hide_render = hidden
        if RANDOMIZE_MACHINERY:
            root.location.x += RNG.uniform(-0.12, 0.12)
            root.location.y += RNG.uniform(-0.12, 0.12)
    workers = sorted((obj for obj in bpy.data.objects if obj.get("synthetic_worker")), key=lambda obj: obj.name)
    registered_workers = [worker for worker in workers if LABEL_OBJECTS[0].get(worker.name)]
    randomization_pool = registered_workers or workers
    active_workers = RNG.randint(1, len(randomization_pool)) if randomization_pool and RANDOMIZE_WORKER_COUNT else len(workers)
    active_population = randomization_pool if registered_workers and RANDOMIZE_WORKER_COUNT else workers
    active_worker_keys = {worker.name for worker in RNG.sample(active_population, active_workers)} if active_population else set()
    floor_slots = {
        1: [(-25, -4), (-19, 3), (-13, -5), (-8, 5), (-2, -4), (4, 5),
            (10, -5), (16, 4), (22, -4), (26, 5), (-21, 7), (12, 7)],
        2: [(-24, 5.8), (-21, 12.5), (-17, 5.8), (-13, 12.5),
            (-9, 5.8), (-5, 12.5), (-4, 9.2)],
    }
    available_slots = {level: list(slots) for level, slots in floor_slots.items()}
    active_targets = []
    for worker_index, worker in enumerate(workers):
        base = worker.get("base_location", list(worker.location))
        active = worker.name in active_worker_keys
        if active:
            level = int(worker.get("level", 2 if base[2] > 1.0 else 1))
            slots = available_slots.setdefault(level, list(floor_slots[1]))
            if not slots:
                slots.extend(floor_slots.get(level, floor_slots[1]))
            x, y = slots.pop(RNG.randrange(len(slots)))
            x += RNG.uniform(-WORKER_POSITION_JITTER, WORKER_POSITION_JITTER)
            y += RNG.uniform(-WORKER_POSITION_JITTER, WORKER_POSITION_JITTER)
            worker.location = (x, y, base[2])
            face_worker_toward_equipment(worker, x, y, RNG.uniform(-0.20, 0.20))
            scale = RNG.uniform(0.88, 1.12)
            worker.scale = (scale, scale, scale)
            active_targets.append(Vector((x, y, base[2] + 1.2)))
        else:
            worker.location = base
            worker.scale = (1.0, 1.0, 1.0)
        for obj in LABEL_OBJECTS[0].get(worker.name, []):
            obj.hide_render = not active
            if obj.type == 'ARMATURE' and RANDOMIZE_WORKER_POSE:
                apply_worker_pose(obj, image_index + worker_index, RNG.uniform(0.0, math.tau))
    # Independently sample helmet and vest presence by worker key.
    for obj in bpy.data.objects:
        kind = obj.get("ppe_kind")
        if kind:
            active = obj.get("worker_key") in active_worker_keys
            present = active and RNG.random() < (HELMET_PROBABILITY if kind == "helmet" else VEST_PROBABILITY)
            obj["ppe_present"] = present
            obj.hide_render = True  # Empty insertion anchors are never renderable.
            for meshes in LABEL_OBJECTS[1 if kind == "helmet" else 2].values():
                for mesh in meshes:
                    if mesh.get("worker_key") == obj.get("worker_key"):
                        mesh.hide_render = not present
    # Imported meshes inherit worker motion through parenting; avoid resetting their world transforms.
    if RANDOMIZE_CAMERA:
        cameras = [obj for obj in COLLECTIONS["Cameras"].objects if obj.type == 'CAMERA']
        cameras = sorted(cameras, key=lambda obj: obj.name)
        camera = cameras[image_index % len(cameras)]
        base_location, base_rotation = CAMERA_BASELINES[camera.name]
        camera.location = base_location
        camera.rotation_euler = base_rotation
        camera.location.x += RNG.uniform(-CAMERA_POSITION_JITTER, CAMERA_POSITION_JITTER)
        camera.location.y += RNG.uniform(-CAMERA_POSITION_JITTER, CAMERA_POSITION_JITTER)
        camera.location.z += RNG.uniform(-0.65, 0.65)
        if active_targets:
            target = RNG.choice(active_targets).copy()
        else:
            target = Vector((RNG.uniform(-18, 18), RNG.uniform(-8, 8), RNG.uniform(1, 8)))
        target.x += RNG.uniform(-5.0, 5.0)
        target.y += RNG.uniform(-3.0, 3.0)
        target.z += RNG.uniform(-0.8, 0.8)
        aim_at(camera, target)
        bpy.context.scene.camera = camera
    if RANDOMIZE_LIGHTING:
        for name, (location, rotation, energy, color) in LIGHT_BASELINES.items():
            obj = bpy.data.objects.get(name)
            if not obj:
                continue
            obj.location = location
            obj.data.color = color
            obj.data.energy = energy * RNG.uniform(*LIGHT_ENERGY_VARIATION)
            obj.location.x += RNG.uniform(-0.55, 0.55)
            obj.location.y += RNG.uniform(-0.55, 0.55)
            obj.location.z += RNG.uniform(-0.2, 0.2)
            base_direction = rotation.to_quaternion() @ Vector((0, 0, -1))
            target = location + base_direction * 10.0
            target.x += RNG.uniform(-LIGHT_DIRECTION_JITTER, LIGHT_DIRECTION_JITTER)
            target.y += RNG.uniform(-LIGHT_DIRECTION_JITTER, LIGHT_DIRECTION_JITTER)
            target.z += RNG.uniform(-0.6, 0.6)
            aim_at(obj, target)

# -----------------------------------------------------------------------------
# Segmentation-derived YOLO labels
# -----------------------------------------------------------------------------
def assign_instance_ids():
    global NEXT_PASS_INDEX
    NEXT_PASS_INDEX = 1
    index_to_label = {}
    for class_id in (0, 1, 2):
        for worker_key, objects in LABEL_OBJECTS[class_id].items():
            # One stable segmentation ID per semantic instance, shared by its visible meshes.
            pass_index = NEXT_PASS_INDEX
            NEXT_PASS_INDEX += 1
            for obj in objects:
                if obj.type != 'MESH':
                    continue
                obj.pass_index = pass_index
            index_to_label[pass_index] = (class_id, worker_key)
    bpy.context.view_layer.use_pass_object_index = True
    return index_to_label


def create_mask_material(name, pass_index):
    red = pass_index & 255
    green = (pass_index >> 8) & 255
    blue = (pass_index >> 16) & 255

    def srgb_to_linear(channel):
        value = channel / 255.0
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (srgb_to_linear(red), srgb_to_linear(green), srgb_to_linear(blue), 1.0)
    output = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def render_instance_mask(scene, output_path, index_to_label):
    """Render unique emissive IDs; ordinary scene geometry remains an opaque occluder."""
    beauty_settings = {
        "engine": scene.render.engine,
        "filepath": scene.render.filepath,
        "file_format": scene.render.image_settings.file_format,
        "color_mode": scene.render.image_settings.color_mode,
        "color_depth": scene.render.image_settings.color_depth,
        "film_transparent": scene.render.film_transparent,
        "use_compositing": scene.render.use_compositing,
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "exposure": scene.view_settings.exposure,
        "gamma": scene.view_settings.gamma,
    }
    mesh_objects = [obj for obj in scene.objects if obj.type == 'MESH']
    data_materials = {obj.data: list(obj.data.materials) for obj in mesh_objects}
    object_slots = {
        obj: [(slot.link, slot.material) for slot in obj.material_slots]
        for obj in mesh_objects
    }
    temporary_materials = []
    black = create_mask_material("Synthetic_Mask_Background", 0)
    temporary_materials.append(black)
    instance_materials = {
        pass_index: create_mask_material(f"Synthetic_Mask_{pass_index:06d}", pass_index)
        for pass_index in index_to_label
    }
    temporary_materials.extend(instance_materials.values())
    try:
        for mesh_data in data_materials:
            if mesh_data.materials:
                for slot_index in range(len(mesh_data.materials)):
                    mesh_data.materials[slot_index] = black
            else:
                mesh_data.materials.append(black)
        for obj in mesh_objects:
            for slot in obj.material_slots:
                slot.link = 'DATA'
        for class_instances in LABEL_OBJECTS.values():
            for objects in class_instances.values():
                for obj in objects:
                    if obj.type != 'MESH':
                        continue
                    mask_material = instance_materials[obj.pass_index]
                    for slot in obj.material_slots:
                        slot.link = 'OBJECT'
                        slot.material = mask_material

        scene.render.filepath = output_path
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.color_depth = '8'
        scene.render.film_transparent = False
        scene.render.use_compositing = False
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        bpy.ops.render.render(write_still=True)
    finally:
        for mesh_data, original_materials in data_materials.items():
            mesh_data.materials.clear()
            for mat in original_materials:
                mesh_data.materials.append(mat)
        for obj, slots in object_slots.items():
            for index, (link, material) in enumerate(slots):
                if index >= len(obj.material_slots):
                    continue
                slot = obj.material_slots[index]
                slot.link = link
                if link == 'OBJECT':
                    slot.material = material
        scene.render.engine = beauty_settings["engine"]
        scene.render.filepath = beauty_settings["filepath"]
        scene.render.image_settings.file_format = beauty_settings["file_format"]
        scene.render.image_settings.color_mode = beauty_settings["color_mode"]
        scene.render.image_settings.color_depth = beauty_settings["color_depth"]
        scene.render.film_transparent = beauty_settings["film_transparent"]
        scene.render.use_compositing = beauty_settings["use_compositing"]
        scene.view_settings.view_transform = beauty_settings["view_transform"]
        scene.view_settings.look = beauty_settings["look"]
        scene.view_settings.exposure = beauty_settings["exposure"]
        scene.view_settings.gamma = beauty_settings["gamma"]
        for mat in temporary_materials:
            bpy.data.materials.remove(mat, do_unlink=True)


def generate_yolo_labels(label_path, index_to_label, segmentation_path):
    """Read rendered material IDs; bounds include visible, unclipped pixels only."""
    image = bpy.data.images.load(segmentation_path, check_existing=False)
    image.colorspace_settings.name = 'Non-Color'
    width, height = image.size
    channels = image.channels
    pixels = image.pixels[:]
    if channels < 3:
        bpy.data.images.remove(image)
        raise RuntimeError("Segmentation mask must contain RGB pixel data")
    bounds = {}
    # RGB channels encode a 24-bit instance ID; image pixels are bottom-up.
    for pixel in range(width * height):
        offset = pixel * channels
        red = int(round(pixels[offset] * 255.0))
        green = int(round(pixels[offset + 1] * 255.0))
        blue = int(round(pixels[offset + 2] * 255.0))
        value = red | (green << 8) | (blue << 16)
        if value not in index_to_label:
            continue
        px = pixel % width
        py = pixel // width
        if value not in bounds:
            bounds[value] = [px, py, px, py]
        else:
            box = bounds[value]
            box[0] = min(box[0], px)
            box[1] = min(box[1], py)
            box[2] = max(box[2], px)
            box[3] = max(box[3], py)
    with open(label_path, "w", encoding="utf-8") as label_file:
        for object_index in sorted(bounds):
            class_id, worker_key = index_to_label[object_index]
            xmin, ymin, xmax, ymax = bounds[object_index]
            # Pixel coordinates are clipped by construction to the rendered frame.
            x_center = ((xmin + xmax + 1) / 2.0) / width
            y_center = 1.0 - ((ymin + ymax + 1) / 2.0) / height
            box_width = (xmax - xmin + 1) / width
            box_height = (ymax - ymin + 1) / height
            label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
    bpy.data.images.remove(image)


def render_dataset():
    bind_existing_scene()
    scene = bpy.context.scene
    output_dir = bpy.path.abspath(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "labels")
    os.makedirs(data_dir, exist_ok=True)
    segmentation_dir = os.path.join(output_dir, ".segmentation")
    os.makedirs(segmentation_dir, exist_ok=True)
    generated = 0
    for index in range(NUM_IMAGES):
        scene.frame_set(index + 1)
        setup_randomization(index)
        index_to_label = assign_instance_ids()
        filename = f"synthetic_{index + 1:06d}"
        segmentation_path = os.path.join(segmentation_dir, filename + "_mask.png")
        render_instance_mask(scene, segmentation_path, index_to_label)
        scene.render.filepath = os.path.join(output_dir, filename + ".png")
        bpy.ops.render.render(write_still=True)
        generate_yolo_labels(os.path.join(data_dir, filename + ".txt"), index_to_label, segmentation_path)
        os.remove(segmentation_path)
        generated += 1
        print(f"Rendered {generated}/{NUM_IMAGES}: {filename}")
    os.rmdir(segmentation_dir)
    return output_dir, generated

# -----------------------------------------------------------------------------
# Project save and entry point
# -----------------------------------------------------------------------------
def save_blend_file():
    if not SAVE_BLEND_FILE:
        return
    if bpy.data.filepath:
        filepath = bpy.path.abspath("//" + BLEND_FILENAME)
    else:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), BLEND_FILENAME)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=filepath)


def main():
    clear_scene()
    create_collections()
    create_materials()
    create_factory_structure()
    create_machinery()
    create_mezzanine_machinery()
    create_props()
    create_lighting()
    create_cameras()
    setup_worker_placeholders()
    setup_upper_floor_workers()
    import_worker_fbx("Worker_01")
    # Ensure camera / lighting baselines and deterministic initial transforms are captured.
    for root in MACHINE_ROOTS:
        root["base_rotation_z"] = root.rotation_euler.z
    scene = bpy.context.scene
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.resolution_percentage = RENDER_PERCENTAGE
    if USE_CYCLES:
        scene.render.engine = 'CYCLES'
    else:
        engines = scene.render.bl_rna.properties["engine"].enum_items.keys()
        scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engines else 'BLENDER_EEVEE'
    scene.view_layers[0].use_pass_object_index = True
    if GENERATE_DATASET:
        output_dir, generated = render_dataset()
    else:
        output_dir = bpy.path.abspath(OUTPUT_DIR)
        generated = 0
    machine_count = len(MACHINE_ROOTS)
    prop_count = len(PROP_ROOTS)
    camera_count = len(COLLECTIONS["Cameras"].objects)
    save_blend_file()
    print("Factory created successfully")
    print(f"Number of machines: {machine_count}")
    print(f"Number of props: {prop_count}")
    print(f"Number of cameras: {camera_count}")
    print(f"Output directory: {output_dir}")
    print(f"Number of synthetic images generated: {generated}")


if __name__ == "__main__":
    main()
