# src/warehouse_environments.py
# Scene Classification Environments — Autonomous Inventory Robot
# Capstone Days 8-12 | Intern: Shivani | Purple AI Labs Ltd
#
# 5 scene classes:
#   0 = Open Aisle
#   1 = Narrow Aisle
#   2 = Pick Station
#   3 = Blocked Path
#   4 = Cross-Aisle Junction
#
# CHANGELOG:
#   v1  — Initial design (blue floor marker, junction at x=4, FOV=60)
#   v2  — Pick Station: blue->orange, added vertical panel + bigger post
#          Junction: opening moved to x=2, FOV widened to 90
#   v3  — Junction: robot spawned INSIDE junction centre (x=4,y=0) so lateral
#          branches fill left/right FOV immediately. Hazard corner posts added
#          as near-field unique cue. FOV restored to 70 (90 showed too much
#          robot body). All other classes unchanged.
#   v4  — Junction: branch shelves start at y=±2, shelf-end cap panels added,
#          hazard posts moved to y=±1.8, floor cross recentered at x=4.0.
#   v5  — __main__: single consistent camera for ALL classes (same FOV=70,
#          same offsets) — matches the one physical camera used in live demo.
#          Robot hidden via changeVisualShape alpha=0 on all links for all
#          classes — robot body must never appear in training frames.

import pybullet as p
import pybullet_data
import random
import math


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def _setup_base(seed):
    """Load ground plane and set gravity. Call at start of every environment."""
    random.seed(seed)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    plane_id = p.loadURDF('plane.urdf')
    return plane_id


def create_box(pos, half_extents, color, orientation=None):
    """Create a static box obstacle."""
    orn = orientation if orientation is not None else [0, 0, 0, 1]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        baseOrientation=orn
    )


def create_cylinder(pos, radius, height, color):
    """Create a static cylinder (scanner posts, hazard posts)."""
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos
    )


def _load_robot(start_pos=None, start_yaw=0.0):
    """Load the R2D2 robot at the given position."""
    if start_pos is None:
        start_pos = [0, 0, 0.5]
    orn = p.getQuaternionFromEuler([0, 0, start_yaw])
    return p.loadURDF('r2d2.urdf', start_pos, orn)


def _add_shelf_row(x_start, x_end, y_pos, shelf_height, shelf_color, step=1.0):
    """Add a continuous row of shelf units along the x-axis."""
    ids = []
    x = x_start
    while x <= x_end:
        ids.append(create_box(
            [x, y_pos, shelf_height / 2],
            [step / 2 - 0.05, 0.3, shelf_height / 2],
            shelf_color
        ))
        x += step
    return ids


def _add_floor_tape(x_start, x_end, y_pos, color, tape_width=0.12):
    """Add flat floor tape strips lying on the ground."""
    ids = []
    x = x_start
    while x <= x_end:
        ids.append(create_box(
            [x, y_pos, 0.01],
            [0.5, tape_width / 2, 0.01],
            color
        ))
        x += 1.0
    return ids


# ===========================================================================
# CLASS 0 — OPEN AISLE
# ===========================================================================

def create_open_aisle(seed=42):
    """
    Wide corridor, grey shelves on both sides, yellow floor tape on edges.
    Visual signature: Yellow tape + wide 4-unit gap + clear depth.
    Robot response: Full speed (0.15 units/step).
    """
    _setup_base(seed)

    shelf_height = random.uniform(0.7, 1.0)
    shelf_gray   = random.uniform(0.45, 0.65)
    shelf_color  = [shelf_gray, shelf_gray, shelf_gray, 1.0]
    aisle_length = 18

    _add_shelf_row(-1, aisle_length, -2.0, shelf_height, shelf_color)
    _add_shelf_row(-1, aisle_length,  2.0, shelf_height, shelf_color)

    yellow = [1.0, 0.85, 0.0, 1.0]
    _add_floor_tape(0, aisle_length, -1.6, yellow)
    _add_floor_tape(0, aisle_length,  1.6, yellow)

    create_box([aisle_length + 0.5, 0, shelf_height / 2],
               [0.1, 3.0, shelf_height / 2], [0.5, 0.5, 0.5, 1.0])

    return _load_robot(start_pos=[0, 0, 0.5], start_yaw=0.0)


# ===========================================================================
# CLASS 1 — NARROW AISLE
# ===========================================================================

def create_narrow_aisle(seed=42):
    """
    Tight tunnel, tall shelves very close on both sides, no floor tape.
    Visual signature: Shelves dominating left+right FOV, narrow 1.8-unit gap.
    Robot response: Half speed (0.08), center-seeking.
    """
    _setup_base(seed)

    shelf_height     = random.uniform(1.1, 1.5)
    shelf_gray       = random.uniform(0.35, 0.55)
    shelf_color      = [shelf_gray, shelf_gray + 0.05, shelf_gray, 1.0]
    aisle_half_width = random.uniform(0.85, 1.0)
    aisle_length     = 18

    _add_shelf_row(-1, aisle_length, -(aisle_half_width + 0.3), shelf_height, shelf_color)
    _add_shelf_row(-1, aisle_length,   aisle_half_width + 0.3,  shelf_height, shelf_color)

    box_colors = [
        [0.8, 0.3, 0.1, 1.0],
        [0.2, 0.5, 0.8, 1.0],
        [0.7, 0.7, 0.2, 1.0],
    ]
    for x in range(1, aisle_length, 3):
        color  = random.choice(box_colors)
        y_side = random.choice([-(aisle_half_width + 0.3), aisle_half_width + 0.3])
        create_box([x, y_side, shelf_height + 0.15], [0.2, 0.2, 0.15], color)

    create_box([aisle_length + 0.5, 0, shelf_height / 2],
               [0.1, aisle_half_width + 0.8, shelf_height / 2], [0.4, 0.4, 0.4, 1.0])

    return _load_robot(start_pos=[0, 0, 0.5], start_yaw=0.0)


# ===========================================================================
# CLASS 2 — PICK STATION
# ===========================================================================

def create_pick_station(seed=42):
    """
    Orange vertical panel + scanner post + close shelf face ahead.

    Orange is unique across all classes:
      yellow = Open Aisle tape
      red    = Blocked Path tape
      grey   = shelves everywhere
      orange = Pick Station only  <-- unambiguous CNN signal

    Visual signature: Orange vertical panel + tall scanner post + shelf face.
    Robot response: Slow approach (0.04), full stop at marker.
    """
    _setup_base(seed)

    shelf_height = random.uniform(0.9, 1.2)
    shelf_color  = [random.uniform(0.4, 0.6)] * 3 + [1.0]

    for y in [-1.5, -0.5, 0.5, 1.5]:
        create_box([4.5, y, shelf_height / 2], [0.3, 0.45, shelf_height / 2], shelf_color)

    item_colors = [
        [0.8, 0.2, 0.2, 1.0],
        [0.2, 0.7, 0.3, 1.0],
        [0.9, 0.6, 0.1, 1.0],
    ]
    for i, y in enumerate([-1.2, 0.0, 1.2]):
        create_box([4.2, y, shelf_height * 0.6], [0.15, 0.2, 0.2], item_colors[i % 3])

    ORANGE      = [1.0, 0.45, 0.0, 1.0]
    panel_width = random.uniform(0.5, 0.7)

    create_box([2.0, 0, 0.35], [0.05, panel_width, 0.25], ORANGE)
    create_box([2.0, 0, 0.015], [0.55, 0.55, 0.015], ORANGE)

    post_y = random.uniform(-0.3, 0.3)
    create_cylinder([2.2, post_y, 0.6], radius=0.07, height=1.2,
                    color=[0.15, 0.15, 0.15, 1.0])
    create_box([2.2, post_y, 1.25], [0.09, 0.09, 0.07], [0.9, 0.45, 0.0, 1.0])

    for x in range(0, 5):
        create_box([x, -2.2, 0.3], [0.05, 0.05, 0.3], [0.5, 0.5, 0.5, 1.0])
        create_box([x,  2.2, 0.3], [0.05, 0.05, 0.3], [0.5, 0.5, 0.5, 1.0])

    return _load_robot(start_pos=[0, 0, 0.5], start_yaw=0.0)


# ===========================================================================
# CLASS 3 — BLOCKED PATH
# ===========================================================================

def create_blocked_path(seed=42):
    """
    Red floor tape + dense cardboard boxes filling the aisle. Impassable.
    Visual signature: Red tape + ground-level clutter + no clear depth.
    Robot response: Full stop (0.0), trigger reroute signal.
    """
    _setup_base(seed)

    shelf_height = random.uniform(0.7, 1.0)
    shelf_color  = [0.5, 0.5, 0.5, 1.0]
    aisle_length = 12

    _add_shelf_row(-1, aisle_length, -2.5, shelf_height, shelf_color)
    _add_shelf_row(-1, aisle_length,  2.5, shelf_height, shelf_color)

    red = [0.9, 0.1, 0.1, 1.0]
    _add_floor_tape(1, 8, -1.5, red)
    _add_floor_tape(1, 8,  0.0, red)
    _add_floor_tape(1, 8,  1.5, red)

    box_colors_pool = [
        [0.6, 0.5, 0.35, 1.0],
        [0.55, 0.5, 0.3, 1.0],
        [0.4, 0.4, 0.4, 1.0],
        [0.7, 0.6, 0.3, 1.0],
    ]

    placed, attempts = 0, 0
    num_boxes = random.randint(8, 14)
    while placed < num_boxes and attempts < 100:
        attempts += 1
        x = random.uniform(1.5, 6.5)
        y = random.uniform(-1.8, 1.8)
        w = random.uniform(0.2, 0.5)
        d = random.uniform(0.2, 0.4)
        h = random.uniform(0.2, 0.55)
        create_box([x, y, h / 2], [w / 2, d / 2, h / 2], random.choice(box_colors_pool))
        placed += 1

    for _ in range(random.randint(2, 4)):
        x  = random.uniform(2.0, 5.0)
        y  = random.uniform(-1.2, 1.2)
        hb = random.uniform(0.3, 0.5)
        ht = random.uniform(0.2, 0.35)
        c  = random.choice(box_colors_pool)
        create_box([x, y, hb / 2],      [0.3, 0.25, hb / 2], c)
        create_box([x, y, hb + ht / 2], [0.25, 0.2, ht / 2], c)

    return _load_robot(start_pos=[0, 0, 0.5], start_yaw=0.0)


# ===========================================================================
# CLASS 4 — CROSS-AISLE JUNCTION
# ===========================================================================

def create_cross_aisle_junction(seed=42):
    """
    Robot spawns INSIDE the junction centre so lateral branch aisles
    immediately fill the left and right sides of the camera frame.

    Visual signature:
      - Shelf-end cap panels immediately left and right
      - Hazard corner posts (yellow bands on dark cylinders) close in frame
      - White floor cross underfoot
      - Forward aisle (plus) or end wall (T) visible ahead
      - Approach aisle visible behind

    Robot response: Reduced speed (0.06), yield logic activated.
    Seed variation: plus-junction (even seed) vs T-junction (odd seed).
    """
    _setup_base(seed)

    shelf_height  = random.uniform(0.75, 1.0)
    shelf_color   = [random.uniform(0.45, 0.6)] * 3 + [1.0]
    junction_type = 'plus' if seed % 2 == 0 else 'T'

    # --- Approach aisle BEHIND the robot (x = -2 to 3) ---
    for xi in range(-2, 4):
        create_box([float(xi), -2.0, shelf_height / 2],
                   [0.45, 0.3, shelf_height / 2], shelf_color)
        create_box([float(xi),  2.0, shelf_height / 2],
                   [0.45, 0.3, shelf_height / 2], shelf_color)

    # --- Shelf-end cap panels at junction mouth (left and right) ---
    cap_color = [random.uniform(0.35, 0.5)] * 3 + [1.0]
    for x_cap in [3.5, 4.5]:
        create_box([x_cap, -2.0, shelf_height / 2],
                   [0.08, 0.5, shelf_height / 2], cap_color)
        create_box([x_cap,  2.0, shelf_height / 2],
                   [0.08, 0.5, shelf_height / 2], cap_color)

    # --- Left branch aisle (shelves running in -Y direction) ---
    for yi in range(2, 10):
        y = -float(yi)
        create_box([4.0, y, shelf_height / 2],
                   [0.45, 0.3, shelf_height / 2], shelf_color)
        create_box([8.0, y, shelf_height / 2],
                   [0.45, 0.3, shelf_height / 2], shelf_color)

    # --- Right branch aisle (shelves running in +Y direction) ---
    for yi in range(2, 10):
        y = float(yi)
        create_box([4.0, y, shelf_height / 2],
                   [0.45, 0.3, shelf_height / 2], shelf_color)
        create_box([8.0, y, shelf_height / 2],
                   [0.45, 0.3, shelf_height / 2], shelf_color)

    # --- Forward: continuation aisle or end wall ---
    if junction_type == 'plus':
        for xi in range(5, 16):
            create_box([float(xi), -2.0, shelf_height / 2],
                       [0.45, 0.3, shelf_height / 2], shelf_color)
            create_box([float(xi),  2.0, shelf_height / 2],
                       [0.45, 0.3, shelf_height / 2], shelf_color)
    else:
        create_box([9.0, 0.0, shelf_height / 2],
                   [0.3, 5.0, shelf_height / 2], shelf_color)

    # --- Hazard corner posts at y=±1.8 ---
    YELLOW = [1.0, 0.85, 0.0, 1.0]
    DARK   = [0.15, 0.15, 0.15, 1.0]
    for cx, cy in [(3.5, -1.8), (3.5, 1.8), (4.5, -1.8), (4.5, 1.8)]:
        create_cylinder([cx, cy, 0.5], radius=0.06, height=1.0, color=DARK)
        create_box([cx, cy, 0.75], [0.07, 0.07, 0.12], YELLOW)

    # --- White floor cross at robot position x=4.0 ---
    WHITE = [0.92, 0.92, 0.92, 1.0]
    create_box([4.0, 0.0, 0.012], [2.0, 0.10, 0.012], WHITE)
    create_box([4.0, 0.0, 0.012], [0.10, 2.5, 0.012], WHITE)

    # --- Robot spawns AT junction centre, facing forward (+X) ---
    return _load_robot(start_pos=[4.0, 0.0, 0.5], start_yaw=0.0)


# ===========================================================================
# SCENE REGISTRY
# ===========================================================================

SCENE_CLASSES = {
    0: ('open_aisle',           create_open_aisle),
    1: ('narrow_aisle',         create_narrow_aisle),
    2: ('pick_station',         create_pick_station),
    3: ('blocked_path',         create_blocked_path),
    4: ('cross_aisle_junction', create_cross_aisle_junction),
}

SEEDS = {
    'seed_A': 42,
    'seed_B': 137,
}


def create_environment(class_idx, seed=42):
    """Factory wrapper. Returns robot_id."""
    if class_idx not in SCENE_CLASSES:
        raise ValueError(f"Unknown class index {class_idx}. Must be 0-4.")
    name, fn = SCENE_CLASSES[class_idx]
    return fn(seed=seed)


# ===========================================================================
# QUICK TEST — python src/warehouse_environments.py
# ===========================================================================

if __name__ == '__main__':
    import numpy as np
    import cv2
    import os

    os.makedirs('preview_images', exist_ok=True)
    print("Testing all 5 environments with 2 seeds each...\n")

    for class_idx, (class_name, factory_fn) in SCENE_CLASSES.items():
        for seed_name, seed_val in SEEDS.items():
            client = p.connect(p.DIRECT)

            robot_id = factory_fn(seed=seed_val)
            p.stepSimulation()

            # Hide robot body on ALL classes — the camera represents the
            # robot's POV. The robot body must never appear in training frames
            # or live demo frames. Alpha=0 makes every link fully transparent.
            for link_idx in range(-1, p.getNumJoints(robot_id)):
                p.changeVisualShape(robot_id, link_idx, rgbaColor=[0, 0, 0, 0])

            pos, orient = p.getBasePositionAndOrientation(robot_id)
            rot = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)
            fwd = rot @ np.array([1, 0, 0])

            # Single consistent camera for ALL classes — matches the one
            # physical camera used in the live demo. Camera is raised 0.8
            # units above robot and offset 0.3 back, looking 3 units ahead.
            cam_pos = [pos[0] - 0.3 * fwd[0],
                       pos[1] - 0.3 * fwd[1],
                       pos[2] + 0.8]
            target  = [pos[0] + fwd[0] * 3.0,
                       pos[1] + fwd[1] * 3.0,
                       pos[2] + 0.2]

            view = p.computeViewMatrix(cam_pos, target, [0, 0, 1])
            proj = p.computeProjectionMatrixFOV(fov=70, aspect=1.0,
                                                nearVal=0.1, farVal=100)
            _, _, rgba, _, _ = p.getCameraImage(224, 224, view, proj,
                                                renderer=p.ER_TINY_RENDERER)
            rgb = np.array(rgba, dtype=np.uint8).reshape(224, 224, 4)[:, :, :3]

            fname = (f'preview_images/class{class_idx}'
                     f'_{class_name}_{seed_name}.jpg')
            cv2.imwrite(fname, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(f"  Class {class_idx} ({class_name}) | "
                  f"{seed_name} (seed={seed_val}) -> {fname}")

            p.disconnect()

    print("\nAll 10 preview images saved to preview_images/")
    print("Run scripts/generate_preview_grid.py to rebuild the comparison grid.")
    print("\nClass summary:")
    for idx, (name, _) in SCENE_CLASSES.items():
        print(f"  {idx}: {name}")