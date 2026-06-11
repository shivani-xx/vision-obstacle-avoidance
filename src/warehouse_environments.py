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
#
# CHANGES FROM v1:
#   - Class 2 (Pick Station): Floor marker changed from blue to ORANGE.
#     Added large vertical orange panel at z=0.45 (camera-visible height).
#     Scanner post made thicker (r=0.07) and taller (h=1.2).
#     The flat floor marker is kept but enlarged so it reads in wide shots.
#     Orange was chosen because it is not used by any other class
#     (yellow=Open Aisle, red=Blocked Path, grey=shelves) — maximum contrast.
#
#   - Class 4 (Cross-Aisle Junction): Junction opening moved from x=4 to x=2
#     so the lateral side aisles fill the camera FOV from the robot's position.
#     White floor cross-lines added at the junction centre for a strong
#     structural cue. Both T-junction and +-junction variants preserved.

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
            [x, y_pos, 0.01],
            [0.5, tape_width / 2, 0.01],
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

    shelf_height = random.uniform(0.7, 1.0)
    shelf_gray = random.uniform(0.45, 0.65)
    shelf_color = [shelf_gray, shelf_gray, shelf_gray, 1.0]
    aisle_length = 18

    # Two parallel shelf rows, 4 units apart (wide aisle)
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

    # Yellow floor tape strips along both edges (OSHA: yellow = aisle marking)
    yellow = [1.0, 0.85, 0.0, 1.0]
    _add_floor_tape(x_start=0, x_end=aisle_length, y_pos=-1.6, color=yellow)
    _add_floor_tape(x_start=0, x_end=aisle_length, y_pos=1.6,  color=yellow)

    # End wall so camera sees something at depth
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

    shelf_height = random.uniform(1.1, 1.5)
    shelf_gray = random.uniform(0.35, 0.55)
    shelf_color = [shelf_gray, shelf_gray + 0.05, shelf_gray, 1.0]

    aisle_half_width = random.uniform(0.85, 1.0)
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

    # Inventory boxes on shelves for visual variety
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

    # End wall
    create_box(
        [aisle_length + 0.5, 0, shelf_height / 2],
        [0.1, aisle_half_width + 0.8, shelf_height / 2],
        [0.4, 0.4, 0.4, 1.0]
    )

    robot_id = _load_robot(start_pos=[0, 0, 0.5], start_yaw=0.0)
    return robot_id


# ===========================================================================
# CLASS 2 — PICK STATION (2 seeds)  [FIXED v2]
# ===========================================================================

def create_pick_station(seed=42):
    """
    ORANGE floor marker + large vertical orange panel (camera-visible) +
    vertical scanner post in center-foreground, shelf face close ahead.
    This is where the robot stops to collect items.

    FIX v2: Marker colour changed from blue to ORANGE.
    Orange is unique — no other class uses it as a primary identifier:
      - Class 0 uses yellow tape
      - Class 3 uses red tape
      - Shelves are grey throughout
    Orange therefore gives the CNN an unambiguous colour signal.

    The flat floor marker alone was invisible from camera height (blind spot
    directly in front of the robot). The fix adds a VERTICAL orange panel
    at z=0.15–0.55, which sits squarely in the camera's forward FOV.
    The floor marker is kept (enlarged) for wide-angle shots.

    Visual signature: ORANGE vertical panel + scanner post + close shelf face.
    Robot response: Slow approach (0.04), full stop at marker.

    Seed variation: post side offset, shelf color, marker panel width.
    """
    _setup_base(seed)

    shelf_height = random.uniform(0.9, 1.2)
    shelf_color = [random.uniform(0.4, 0.6)] * 3 + [1.0]

    # Shelf face close ahead (robot approaching a shelf to pick)
    for y in [-1.5, -0.5, 0.5, 1.5]:
        create_box(
            [4.5, y, shelf_height / 2],
            [0.3, 0.45, shelf_height / 2],
            shelf_color
        )

    # Inventory items on shelf face
    item_colors = [
        [0.8, 0.2, 0.2, 1.0],   # Red item
        [0.2, 0.7, 0.3, 1.0],   # Green item
        [0.9, 0.6, 0.1, 1.0],   # Yellow-orange item
    ]
    for i, y in enumerate([-1.2, 0.0, 1.2]):
        create_box(
            [4.2, y, shelf_height * 0.6],
            [0.15, 0.2, 0.2],
            item_colors[i % 3]
        )

    # -----------------------------------------------------------------------
    # ORANGE station marker — TWO components for maximum visibility:
    #
    # (A) Vertical panel: faces the camera directly, clearly visible from
    #     robot height. This is the primary CNN signal for this class.
    # (B) Floor footprint: enlarged flat marker on the ground for wide shots.
    # -----------------------------------------------------------------------
    ORANGE = [1.0, 0.45, 0.0, 1.0]   # Strong construction orange

    panel_width = random.uniform(0.5, 0.7)   # Seed variation in panel size

    # (A) Vertical orange panel — robot sees this face-on
    create_box(
        [2.0, 0, 0.35],               # z=0.35 centres it in camera FOV
        [0.05, panel_width, 0.25],    # Thin depth, wide, tall slab
        ORANGE
    )

    # (B) Enlarged floor footprint — visible in wide-angle / overhead shots
    create_box(
        [2.0, 0, 0.015],
        [0.55, 0.55, 0.015],
        ORANGE
    )

    # -----------------------------------------------------------------------
    # Scanner post — thicker and taller than v1 so it reads in frame
    # -----------------------------------------------------------------------
    post_y_offset = random.uniform(-0.3, 0.3)
    create_cylinder(
        pos=[2.2, post_y_offset, 0.6],
        radius=0.07,       # Thicker than v1 (was 0.04)
        height=1.2,        # Taller than v1 (was 0.8)
        color=[0.15, 0.15, 0.15, 1.0]
    )
    # Scanner head at top of post
    create_box(
        [2.2, post_y_offset, 1.25],
        [0.09, 0.09, 0.07],
        [0.9, 0.45, 0.0, 1.0]   # Orange scanner head matches station colour
    )

    # Side walls to frame the station
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

    # Background shelf rows
    _add_shelf_row(-1, aisle_length, -2.5, shelf_height, shelf_color)
    _add_shelf_row(-1, aisle_length,  2.5, shelf_height, shelf_color)

    # Red floor tape (OSHA: red = danger/stop zone)
    red = [0.9, 0.1, 0.1, 1.0]
    _add_floor_tape(x_start=1, x_end=8, y_pos=-1.5, color=red)
    _add_floor_tape(x_start=1, x_end=8, y_pos=0.0,  color=red)
    _add_floor_tape(x_start=1, x_end=8, y_pos=1.5,  color=red)

    # Boxes and pallets blocking the path
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

    # Stacked boxes for height variation
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
# CLASS 4 — CROSS-AISLE JUNCTION (2 seeds)  [FIXED v2]
# ===========================================================================

def create_cross_aisle_junction(seed=42):
    """
    Main aisle opens into a wider space with shelf rows visible branching
    left AND right. Multiple directions visible simultaneously from the
    robot's camera.

    FIX v2: Junction opening moved from x=4 to x=2 (much closer to robot).
    At the old distance (x=4+) the lateral openings were at the far edge of
    the camera's FOV and looked almost identical to an open aisle. Moving the
    opening to x=2 means the left and right branch aisles fill a large portion
    of the left and right sides of the frame — making the lateral structure
    immediately obvious to the CNN.

    White floor cross-lines added at the junction centre (x=3.0) as an
    additional structural cue — no other class has this floor marking.

    Visual signature: Lateral open space filling left+right frame edges +
                      white floor cross at junction + multi-direction shelves.
    Robot response: Reduced speed (0.06), yield logic activated.

    Seed variation: T-junction vs +-junction, shelf height, shelf brightness.
    """
    _setup_base(seed)

    shelf_height = random.uniform(0.75, 1.0)
    shelf_color = [random.uniform(0.45, 0.6)] * 3 + [1.0]

    junction_type = random.choice(['T', 'plus'])

    # -----------------------------------------------------------------------
    # Approach aisle: SHORT — only 2 units before the junction opens.
    # This ensures the lateral openings are close enough to fill the FOV.
    # -----------------------------------------------------------------------
    for x in range(-1, 2):
        create_box([x, -2.0, shelf_height / 2], [0.45, 0.3, shelf_height / 2], shelf_color)
        create_box([x,  2.0, shelf_height / 2], [0.45, 0.3, shelf_height / 2], shelf_color)

    # -----------------------------------------------------------------------
    # Junction opening at x=2.
    # Left branch: shelf rows running in the -Y direction from the opening.
    # Right branch: shelf rows running in the +Y direction from the opening.
    # The inner wall of each branch (x=2.5) is what the camera sees to its
    # left and right — clearly different from a straight aisle.
    # -----------------------------------------------------------------------

    # Left branch (negative Y)
    for y_step in range(0, 6):
        y = -(2.5 + y_step * 1.0)
        create_box([2.5, y, shelf_height / 2], [0.45, 0.3, shelf_height / 2], shelf_color)
        create_box([6.5, y, shelf_height / 2], [0.45, 0.3, shelf_height / 2], shelf_color)

    # Right branch (positive Y)
    for y_step in range(0, 6):
        y = (2.5 + y_step * 1.0)
        create_box([2.5, y, shelf_height / 2], [0.45, 0.3, shelf_height / 2], shelf_color)
        create_box([6.5, y, shelf_height / 2], [0.45, 0.3, shelf_height / 2], shelf_color)

    # -----------------------------------------------------------------------
    # Forward continuation (seed-dependent junction type)
    # -----------------------------------------------------------------------
    if junction_type == 'plus':
        # + junction: main aisle continues ahead past the opening
        for x in range(8, 16):
            create_box([x, -2.0, shelf_height / 2], [0.45, 0.3, shelf_height / 2], shelf_color)
            create_box([x,  2.0, shelf_height / 2], [0.45, 0.3, shelf_height / 2], shelf_color)
    else:
        # T junction: wall closes off the forward direction
        create_box([8.5, 0, shelf_height / 2], [0.3, 5.0, shelf_height / 2], shelf_color)

    # -----------------------------------------------------------------------
    # White floor cross-lines at junction centre (x=3.0).
    # Strong structural cue unique to this class — no other class uses white
    # floor markings. The CNN will learn to associate these with junctions.
    # -----------------------------------------------------------------------
    WHITE = [0.92, 0.92, 0.92, 1.0]

    # Longitudinal centre line (forward direction)
    create_box([3.5, 0.0, 0.012], [1.4, 0.08, 0.012], WHITE)

    # Transverse line (left-right direction) — the cross-bar
    create_box([3.0, 0.0, 0.012], [0.08, 2.2, 0.012], WHITE)

    # Short tick marks pointing into each branch
    create_box([3.0, -1.8, 0.012], [0.08, 0.5, 0.012], WHITE)
    create_box([3.0,  1.8, 0.012], [0.08, 0.5, 0.012], WHITE)

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

            # Capture one camera frame from robot's forward-facing camera
            pos, orient = p.getBasePositionAndOrientation(robot_id)
            rot = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)
            cam_pos = [pos[0], pos[1], pos[2] + 0.3]
            fwd = rot @ np.array([1, 0, 0])
            target = [cam_pos[0] + fwd[0], cam_pos[1] + fwd[1], cam_pos[2] + fwd[2]]
            view = p.computeViewMatrix(cam_pos, target, [0, 0, 1])
            proj = p.computeProjectionMatrixFOV(fov=90, aspect=1.0, nearVal=0.1, farVal=100)
            _, _, rgba, _, _ = p.getCameraImage(224, 224, view, proj,
                                                renderer=p.ER_TINY_RENDERER)
            rgb = np.array(rgba, dtype=np.uint8).reshape(224, 224, 4)[:, :, :3]

            fname = f'preview_images/class{class_idx}_{class_name}_{seed_name}.jpg'
            cv2.imwrite(fname, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(f"  Class {class_idx} ({class_name}) | {seed_name} (seed={seed_val}) -> {fname}")

            p.disconnect()

    print("\nAll 10 preview images saved to preview_images/")
    print("Run scripts/generate_preview_grid.py to rebuild the comparison grid.")
    print("\nClass summary:")
    for idx, (name, _) in SCENE_CLASSES.items():
        print(f"  {idx}: {name}")