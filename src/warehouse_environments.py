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
# Each factory function:
#   - accepts a seed for reproducibility
#   - sets up gravity + ground plane
#   - builds the scene
#   - loads the robot at [0, 0, 0.5]
#   - returns robot_id
#
# Usage:
#   client = p.connect(p.DIRECT)
#   robot_id = create_open_aisle(seed=42)
#   ... collect frames ...
#   p.disconnect()

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
    """Create a static cylinder (used for scanner posts at pick stations)."""
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
    """
    Add a continuous row of shelf units along the x-axis at a fixed y position.
    Each shelf unit is one box. Returns list of body IDs.
    """
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
    """
    Add flat floor tape strips (thin flat boxes lying on the ground).
    Used for yellow tape in open aisles and red tape in blocked paths.
    """
    ids = []
    x = x_start
    while x <= x_end:
        ids.append(create_box(
            [x, y_pos, 0.01],                        # Just above ground
            [0.5, tape_width / 2, 0.01],              # Flat strip
            color
        ))
        x += 1.0
    return ids


# ===========================================================================
# CLASS 0 — OPEN AISLE (2 seeds)
# ===========================================================================

def create_open_aisle(seed=42):
    """
    Wide corridor with grey shelving units on both sides and yellow floor
    tape along both edges. Clear path ahead.

    Visual signature: Yellow tape on floor edges + wide gap (4 units) + depth.
    Robot response: Full speed (0.15 units/step).

    Seed variation: shelf height, shelf color brightness, tape length.
    """
    _setup_base(seed)

    # --- Tunable parameters varied by seed ---
    shelf_height = random.uniform(0.7, 1.0)
    shelf_gray = random.uniform(0.45, 0.65)
    shelf_color = [shelf_gray, shelf_gray, shelf_gray, 1.0]
    aisle_length = 18     # How far the corridor extends

    # --- Two parallel shelf rows, 4 units apart (wide aisle) ---
    _add_shelf_row(
        x_start=-1, x_end=aisle_length,
        y_pos=-2.0,
        shelf_height=shelf_height,
        shelf_color=shelf_color,
        step=1.0
    )
    _add_shelf_row(
        x_start=-1, x_end=aisle_length,
        y_pos=2.0,
        shelf_height=shelf_height,
        shelf_color=shelf_color,
        step=1.0
    )

    # --- Yellow floor tape strips along both edges ---
    # OSHA standard: yellow = aisle marking
    yellow = [1.0, 0.85, 0.0, 1.0]
    _add_floor_tape(x_start=0, x_end=aisle_length, y_pos=-1.6, color=yellow)
    _add_floor_tape(x_start=0, x_end=aisle_length, y_pos=1.6,  color=yellow)

    # --- End wall so camera sees something at depth ---
    create_box(
        [aisle_length + 0.5, 0, shelf_height / 2],
        [0.1, 3.0, shelf_height / 2],
        [0.5, 0.5, 0.5, 1.0]
    )

    robot_id = _load_robot(start_pos=[0, 0, 0.5], start_yaw=0.0)
    return robot_id


# ===========================================================================
# CLASS 1 — NARROW AISLE (2 seeds)
# ===========================================================================

def create_narrow_aisle(seed=42):
    """
    Tight corridor with tall shelving units very close on both sides.
    No floor tape. Robot barely fits through.

    Visual signature: Shelves filling left+right FOV, narrow 1.8-unit gap.
    Robot response: Half speed (0.08), center-seeking.

    Seed variation: shelf height (taller than open aisle), aisle width variation.
    """
    _setup_base(seed)

    # --- Taller shelves than open aisle to fill more vertical FOV ---
    shelf_height = random.uniform(1.1, 1.5)
    shelf_gray = random.uniform(0.35, 0.55)
    shelf_color = [shelf_gray, shelf_gray + 0.05, shelf_gray, 1.0]

    # --- Narrow gap: only 1.8 units between shelf faces ---
    aisle_half_width = random.uniform(0.85, 1.0)   # ~1.7 to 2.0 total width
    aisle_length = 18

    _add_shelf_row(
        x_start=-1, x_end=aisle_length,
        y_pos=-aisle_half_width - 0.3,
        shelf_height=shelf_height,
        shelf_color=shelf_color,
        step=1.0
    )
    _add_shelf_row(
        x_start=-1, x_end=aisle_length,
        y_pos=aisle_half_width + 0.3,
        shelf_height=shelf_height,
        shelf_color=shelf_color,
        step=1.0
    )

    # --- Add some inventory boxes ON the shelves (visual variety) ---
    box_colors = [
        [0.8, 0.3, 0.1, 1.0],   # Orange box
        [0.2, 0.5, 0.8, 1.0],   # Blue box
        [0.7, 0.7, 0.2, 1.0],   # Yellow box
    ]
    for x in range(1, aisle_length, 3):
        color = random.choice(box_colors)
        y_side = random.choice([-aisle_half_width - 0.3, aisle_half_width + 0.3])
        create_box(
            [x, y_side, shelf_height + 0.15],
            [0.2, 0.2, 0.15],
            color
        )

    # --- End wall ---
    create_box(
        [aisle_length + 0.5, 0, shelf_height / 2],
        [0.1, aisle_half_width + 0.8, shelf_height / 2],
        [0.4, 0.4, 0.4, 1.0]
    )

    robot_id = _load_robot(start_pos=[0, 0, 0.5], start_yaw=0.0)
    return robot_id


# ===========================================================================
# CLASS 2 — PICK STATION (2 seeds)
# ===========================================================================

def create_pick_station(seed=42):
    """
    Blue floor marker + vertical scanner post in center-foreground,
    shelf face close ahead. This is where the robot stops to collect items.

    Visual signature: BLUE floor marker (unique color) + vertical post + close shelf.
    Robot response: Slow approach (0.04), full stop at marker.

    Seed variation: post position (left/center/right), shelf color, marker size.
    """
    _setup_base(seed)

    shelf_height = random.uniform(0.9, 1.2)
    shelf_color = [random.uniform(0.4, 0.6)] * 3 + [1.0]

    # --- Shelf face close ahead (the robot is approaching a shelf to pick) ---
    # One wide shelf unit 4 units ahead
    for y in [-1.5, -0.5, 0.5, 1.5]:
        create_box(
            [4.5, y, shelf_height / 2],
            [0.3, 0.45, shelf_height / 2],
            shelf_color
        )

    # Add inventory items on the shelf face
    item_colors = [[0.8, 0.2, 0.2, 1.0], [0.2, 0.7, 0.3, 1.0], [0.9, 0.6, 0.1, 1.0]]
    for i, y in enumerate([-1.2, 0.0, 1.2]):
        create_box(
            [4.2, y, shelf_height * 0.6],
            [0.15, 0.2, 0.2],
            item_colors[i % 3]
        )

    # --- Blue floor marker (unique to this class — no other class uses blue) ---
    # OSHA: blue = informational/work zone marker
    marker_size_x = random.uniform(0.35, 0.5)
    marker_size_y = random.uniform(0.25, 0.35)
    create_box(
        [2.0, 0, 0.01],
        [marker_size_x, marker_size_y, 0.01],
        [0.1, 0.4, 0.9, 1.0]   # Strong blue
    )

    # --- Vertical scanner post (barcode reader stand) ---
    post_y_offset = random.uniform(-0.3, 0.3)   # Slightly left or right
    create_cylinder(
        pos=[2.2, post_y_offset, 0.4],
        radius=0.04,
        height=0.8,
        color=[0.15, 0.15, 0.15, 1.0]   # Dark grey post
    )
    # Post head (the scanner unit at the top)
    create_box(
        [2.2, post_y_offset, 0.85],
        [0.08, 0.08, 0.06],
        [0.2, 0.2, 0.8, 1.0]   # Blue scanner head
    )

    # --- Side walls to frame the station ---
    for x in range(0, 5):
        create_box([x, -2.2, 0.3], [0.05, 0.05, 0.3], [0.5, 0.5, 0.5, 1.0])
        create_box([x,  2.2, 0.3], [0.05, 0.05, 0.3], [0.5, 0.5, 0.5, 1.0])

    robot_id = _load_robot(start_pos=[0, 0, 0.5], start_yaw=0.0)
    return robot_id


# ===========================================================================
# CLASS 3 — BLOCKED PATH (2 seeds)
# ===========================================================================

def create_blocked_path(seed=42):
    """
    Red floor tape strips + boxes/pallets filling 60-80% of aisle width.
    Clearly impassable. Robot must stop and reroute.

    Visual signature: Red tape + dense ground-level clutter + no clear depth.
    Robot response: Full stop (0.0), trigger reroute signal.

    Seed variation: box density, arrangement, box colors/sizes.
    """
    _setup_base(seed)

    shelf_height = random.uniform(0.7, 1.0)
    shelf_color = [0.5, 0.5, 0.5, 1.0]
    aisle_length = 12

    # --- Background shelf rows (so the scene looks like an aisle) ---
    _add_shelf_row(-1, aisle_length, -2.5, shelf_height, shelf_color)
    _add_shelf_row(-1, aisle_length,  2.5, shelf_height, shelf_color)

    # --- Red floor tape (OSHA: red = danger/stop zone) ---
    red = [0.9, 0.1, 0.1, 1.0]
    _add_floor_tape(x_start=1, x_end=8, y_pos=-1.5, color=red)
    _add_floor_tape(x_start=1, x_end=8, y_pos=0.0,  color=red)
    _add_floor_tape(x_start=1, x_end=8, y_pos=1.5,  color=red)

    # --- Boxes and pallets blocking the path ---
    num_boxes = random.randint(8, 14)
    box_colors_pool = [
        [0.6, 0.5, 0.35, 1.0],   # Cardboard brown
        [0.55, 0.5, 0.3, 1.0],   # Tan
        [0.4, 0.4, 0.4, 1.0],    # Grey pallet
        [0.7, 0.6, 0.3, 1.0],    # Light brown
    ]

    placed = 0
    attempts = 0
    while placed < num_boxes and attempts < 100:
        attempts += 1
        x = random.uniform(1.5, 6.5)
        y = random.uniform(-1.8, 1.8)
        w = random.uniform(0.2, 0.5)
        d = random.uniform(0.2, 0.4)
        h = random.uniform(0.2, 0.55)
        color = random.choice(box_colors_pool)
        create_box([x, y, h / 2], [w / 2, d / 2, h / 2], color)
        placed += 1

    # --- Some stacked boxes (height variation makes it visually richer) ---
    for _ in range(random.randint(2, 4)):
        x = random.uniform(2.0, 5.0)
        y = random.uniform(-1.2, 1.2)
        h_bottom = random.uniform(0.3, 0.5)
        h_top = random.uniform(0.2, 0.35)
        color = random.choice(box_colors_pool)
        create_box([x, y, h_bottom / 2], [0.3, 0.25, h_bottom / 2], color)
        create_box([x, y, h_bottom + h_top / 2], [0.25, 0.2, h_top / 2], color)

    robot_id = _load_robot(start_pos=[0, 0, 0.5], start_yaw=0.0)
    return robot_id


# ===========================================================================
# CLASS 4 — CROSS-AISLE JUNCTION (2 seeds)
# ===========================================================================

def create_cross_aisle_junction(seed=42):
    """
    Main aisle opens into a wider space with shelf rows visible branching
    left and/or right. Multiple directions visible simultaneously.

    Visual signature: Lateral open space on both sides + multi-direction shelves.
    Robot response: Reduced speed (0.06), yield logic activated.

    Seed variation: T-junction vs + junction, shelf distances, junction width.
    """
    _setup_base(seed)

    shelf_height = random.uniform(0.75, 1.0)
    shelf_color = [random.uniform(0.45, 0.6)] * 3 + [1.0]

    junction_type = random.choice(['T', 'plus'])   # Seed determines junction shape

    # --- Main approach aisle (robot comes from x=0 heading +x) ---
    # Shelf rows stop before the junction opening
    for x in range(-1, 4):
        create_box([x, -2.0, shelf_height / 2], [0.45, 0.3, shelf_height / 2], shelf_color)
        create_box([x,  2.0, shelf_height / 2], [0.45, 0.3, shelf_height / 2], shelf_color)

    # --- Junction opening: shelves break away left and right ---
    # Left branch aisle
    for y in range(-8, -3):
        create_box([4.0, y, shelf_height / 2], [0.45, 0.3, shelf_height / 2], shelf_color)
        create_box([8.0, y, shelf_height / 2], [0.45, 0.3, shelf_height / 2], shelf_color)

    # Right branch aisle
    for y in range(3, 8):
        create_box([4.0, y, shelf_height / 2], [0.45, 0.3, shelf_height / 2], shelf_color)
        create_box([8.0, y, shelf_height / 2], [0.45, 0.3, shelf_height / 2], shelf_color)

    if junction_type == 'plus':
        # + junction: main aisle CONTINUES ahead beyond the opening
        for x in range(9, 16):
            create_box([x, -2.0, shelf_height / 2], [0.45, 0.3, shelf_height / 2], shelf_color)
            create_box([x,  2.0, shelf_height / 2], [0.45, 0.3, shelf_height / 2], shelf_color)
    else:
        # T junction: main aisle ends at the junction (wall ahead)
        create_box([9.0, 0, shelf_height / 2], [0.3, 4.5, shelf_height / 2], shelf_color)

    # --- Floor: no tape in junction zone (open space feel) ---
    # Small directional arrows on the floor (thin flat boxes) to hint at routes
    arrow_color = [0.8, 0.8, 0.8, 0.8]
    create_box([5.5,  0.0, 0.01], [0.6, 0.08, 0.01], arrow_color)   # Center line
    create_box([5.5, -2.5, 0.01], [0.4, 0.06, 0.01], arrow_color)   # Left hint
    create_box([5.5,  2.5, 0.01], [0.4, 0.06, 0.01], arrow_color)   # Right hint

    robot_id = _load_robot(start_pos=[0, 0, 0.5], start_yaw=0.0)
    return robot_id


# ===========================================================================
# CONVENIENCE: SCENE REGISTRY
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
    """
    Convenience wrapper. Creates the environment for the given class index.
    Returns robot_id.

    Example:
        client = p.connect(p.DIRECT)
        robot_id = create_environment(class_idx=2, seed=42)
    """
    if class_idx not in SCENE_CLASSES:
        raise ValueError(f"Unknown class index {class_idx}. Must be 0-4.")
    name, fn = SCENE_CLASSES[class_idx]
    return fn(seed=seed)


# ===========================================================================
# QUICK TEST — run this file directly to verify all environments load
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

            # Capture one camera frame
            pos, orient = p.getBasePositionAndOrientation(robot_id)
            rot = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)
            cam_pos = [pos[0], pos[1], pos[2] + 0.3]
            fwd = rot @ np.array([1, 0, 0])
            target = [cam_pos[0] + fwd[0], cam_pos[1] + fwd[1], cam_pos[2] + fwd[2]]
            view = p.computeViewMatrix(cam_pos, target, [0, 0, 1])
            proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100)
            _, _, rgba, _, _ = p.getCameraImage(224, 224, view, proj,
                                                renderer=p.ER_TINY_RENDERER)
            rgb = np.array(rgba, dtype=np.uint8).reshape(224, 224, 4)[:, :, :3]

            fname = f'preview_images/class{class_idx}_{class_name}_{seed_name}.jpg'
            cv2.imwrite(fname, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(f"  Class {class_idx} ({class_name}) | {seed_name} (seed={seed_val}) -> {fname}")

            p.disconnect()

    print("\nAll 10 preview images saved to preview_images/")
    print("Open them to verify each class looks visually distinct!")
    print("\nClass summary:")
    for idx, (name, _) in SCENE_CLASSES.items():
        print(f"  {idx}: {name}")