# src/warehouse_environments.py

import pybullet as p
import pybullet_data
import random
import math
import numpy as np


def _setup_base(seed):
   
    random.seed(seed)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    plane_id = p.loadURDF('plane.urdf')
    return plane_id


def create_box(pos, half_extents, color, orientation=None):
    
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
    
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos
    )


def _load_robot(start_pos=None, start_yaw=0.0):
    
    if start_pos is None:
        start_pos = [0, 0, 0.5]
    orn = p.getQuaternionFromEuler([0, 0, start_yaw])
    return p.loadURDF('r2d2.urdf', start_pos, orn)


def _add_shelf_row(x_start, x_end, y_pos, shelf_height, shelf_color, step=1.0):
    
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



def create_open_aisle(seed=42):

    _setup_base(seed)
    random.seed(seed)

    aisle_length = 18

    # ---------- Geometry variation ----------
    left_width = random.uniform(1.6, 3.5)
    right_width = random.uniform(1.6, 3.5)

    shelf_height_left = random.uniform(0.7, 1.4)
    shelf_height_right = random.uniform(0.7, 1.4)

    shelf_palettes = [
        [0.55, 0.55, 0.55, 1.0],  # gray
        [0.35, 0.50, 0.35, 1.0],  # green
        [0.40, 0.40, 0.60, 1.0],  # blue
        [0.55, 0.45, 0.30, 1.0],  # brown
    ]

    shelf_color = random.choice(shelf_palettes)

    # ---------- Left shelves ----------
    gap_probability = random.uniform(0.10, 0.35)

    for x in np.arange(-1, aisle_length, 1.0):

        if random.random() < gap_probability:
            continue

        create_box(
            [x, -left_width, shelf_height_left / 2],
            [0.45, 0.3, shelf_height_left / 2],
            shelf_color
        )

    # ---------- Right shelves ----------
    for x in np.arange(-1, aisle_length, 1.0):

        if random.random() < gap_probability:
            continue

        create_box(
            [x, right_width, shelf_height_right / 2],
            [0.45, 0.3, shelf_height_right / 2],
            shelf_color
        )

    # ---------- Yellow safety tape ----------
    yellow = [1.0, 0.85, 0.0, 1.0]

    _add_floor_tape(
        0,
        aisle_length,
        -(left_width - 0.35),
        yellow
    )

    _add_floor_tape(
        0,
        aisle_length,
        (right_width - 0.35),
        yellow
    )

    # ---------- Pallet colors ----------
    pallet_colors = [
        [0.7, 0.5, 0.2, 1.0],
        [0.8, 0.2, 0.2, 1.0],
        [0.2, 0.5, 0.8, 1.0],
    ]

    # ---------- Storage zones outside aisle ----------
    for _ in range(random.randint(3, 8)):

        create_box(
            [
                random.uniform(1, aisle_length - 1),
                random.choice([
                    -(left_width + random.uniform(0.8, 2.0)),
                    right_width + random.uniform(0.8, 2.0)
                ]),
                random.uniform(0.15, 0.4)
            ],
            [
                random.uniform(0.2, 0.6),
                random.uniform(0.2, 0.6),
                random.uniform(0.15, 0.5)
            ],
            random.choice(pallet_colors)
        )

    # ---------- Random side pallets ----------
    for _ in range(random.randint(4, 12)):

        side_y = random.choice([
            random.uniform(
                -(max(left_width, right_width) + 2.5),
                -(max(left_width, right_width) + 1.0)
            ),
            random.uniform(
                max(left_width, right_width) + 1.0,
                max(left_width, right_width) + 2.5
            )
        ])

        create_box(
            [
                random.uniform(0, aisle_length),
                side_y,
                0.2
            ],
            [0.25, 0.25, 0.2],
            random.choice(pallet_colors)
        )

    # ---------- End wall ----------
    wall_width = random.uniform(
        max(left_width, right_width) + 0.5,
        max(left_width, right_width) + 2.5
    )

    wall_color = random.choice([
        [0.4, 0.4, 0.4, 1.0],
        [0.3, 0.3, 0.5, 1.0],
        [0.5, 0.4, 0.3, 1.0]
    ])

    create_box(
        [
            aisle_length + 0.5,
            0,
            max(shelf_height_left, shelf_height_right) / 2
        ],
        [
            0.1,
            wall_width,
            max(shelf_height_left, shelf_height_right) / 2
        ],
        wall_color
    )

    return _load_robot(
        start_pos=[0, 0, 0.5],
        start_yaw=0.0
    )




def create_narrow_aisle(seed=42):

    _setup_base(seed)
    random.seed(seed)

    aisle_length = 18

    # ---------- Narrow geometry ----------
    aisle_half_width = random.uniform(0.65, 0.90)

    # ---------- Shelf variation ----------
    left_height = random.uniform(1.4, 2.2)
    right_height = random.uniform(1.4, 2.2)

    shelf_palettes = [
        [0.25, 0.35, 0.25, 1.0],   # dark green
        [0.40, 0.40, 0.40, 1.0],   # gray
        [0.35, 0.35, 0.55, 1.0],   # blue gray
    ]

    shelf_color = random.choice(shelf_palettes)

    # ---------- Shelf rows ----------
    for x in np.arange(-1, aisle_length, 1.0):

        left_depth = random.uniform(0.30, 0.55)
        right_depth = random.uniform(0.30, 0.55)

        create_box(
            [x, -(aisle_half_width + left_depth), left_height / 2],
            [0.45, left_depth, left_height / 2],
            shelf_color
        )

        create_box(
            [x, aisle_half_width + right_depth, right_height / 2],
            [0.45, right_depth, right_height / 2],
            shelf_color
        )

    # ---------- Shelf cargo ----------
    cargo_colors = [
        [0.8, 0.3, 0.1, 1.0],
        [0.2, 0.5, 0.8, 1.0],
        [0.7, 0.7, 0.2, 1.0],
        [0.6, 0.3, 0.7, 1.0],
    ]

    for _ in range(random.randint(20, 45)):

        x = random.uniform(0, aisle_length)

        side = random.choice([-1, 1])

        y = side * (
            aisle_half_width +
            random.uniform(0.15, 0.45)
        )

        create_box(
            [
                x,
                y,
                random.uniform(0.6, max(left_height, right_height))
            ],
            [
                random.uniform(0.10, 0.25),
                random.uniform(0.10, 0.25),
                random.uniform(0.10, 0.25)
            ],
            random.choice(cargo_colors)
        )

    # ---------- Shelf protrusions ----------
    for _ in range(random.randint(8, 15)):

        x = random.uniform(1, aisle_length - 1)

        side = random.choice([-1, 1])

        y = side * aisle_half_width

        create_box(
            [
                x,
                y,
                random.uniform(0.5, min(left_height, right_height) - 0.2)
            ],
            [0.12, 0.12, 0.12],
            random.choice(cargo_colors)
        )

    # ---------- End wall ----------
    create_box(
        [aisle_length + 0.5, 0, max(left_height, right_height) / 2],
        [0.1, aisle_half_width + 0.8, max(left_height, right_height) / 2],
        [0.35, 0.35, 0.35, 1.0]
    )

    return _load_robot(
        start_pos=[0, 0, 0.5],
        start_yaw=0.0
    )



def create_pick_station(seed=42):

    random.seed(seed)

    _setup_base(seed)

    # --------------------------------------------------
    # Central pick station (ALWAYS visible)
    # --------------------------------------------------

    station_x = 2.5
    station_y = 0.0

    create_box(
        [station_x, station_y, 0.4],
        [0.8, 0.8, 0.4],
        [1.0, 0.5, 0.0, 1]
    )

    # --------------------------------------------------
    # Back wall panels
    # --------------------------------------------------

    for panel_y in [-1.5, 0, 1.5]:

        create_box(
            [4.5, panel_y, 0.8],
            [0.15, 0.8, 0.8],
            [0.45, 0.45, 0.45, 1]
        )

    # --------------------------------------------------
    # Random inventory boxes
    # --------------------------------------------------

    box_count = random.randint(8, 20)

    for _ in range(box_count):

        x = random.uniform(0.5, 5.5)
        y = random.uniform(-2.5, 2.5)

        # keep area around station clear
        if abs(x - station_x) < 1.2 and abs(y) < 1.0:
            continue

        size = random.uniform(0.15, 0.45)

        color = [
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0),
            1
        ]

        create_box(
            [x, y, size / 2],
            [size, size, size],
            color
        )

    # --------------------------------------------------
    # Random shelving
    # --------------------------------------------------

    shelf_count = random.randint(4, 10)

    for _ in range(shelf_count):

        side = random.choice([-1, 1])

        x = random.uniform(0.5, 5.5)
        y = side * random.uniform(1.5, 2.5)

        height = random.uniform(0.8, 1.8)

        create_box(
            [x, y, height / 2],
            [0.3, 0.8, height],
            [0.35, 0.4, 0.35, 1]
        )

    # --------------------------------------------------
    # Floor markers
    # --------------------------------------------------

    marker_count = random.randint(2, 8)

    for _ in range(marker_count):

        marker_x = random.uniform(0.5, 5.5)
        marker_y = random.uniform(-2.5, 2.5)

        create_box(
            [marker_x, marker_y, 0.01],
            [0.25, 0.25, 0.02],
            [
                random.uniform(0.5, 1.0),
                random.uniform(0.5, 1.0),
                0,
                1
            ]
        )

    return _load_robot(
        start_pos=[0, 0, 0.5],
        start_yaw=0.0
    )


def create_blocked_path(seed=42):

    _setup_base(seed)
    random.seed(seed)

    aisle_length = 14

    # ---------- Shelf variation ----------
    shelf_height = random.uniform(0.8, 1.4)

    shelf_palettes = [
        [0.45, 0.45, 0.45, 1.0],
        [0.35, 0.45, 0.35, 1.0],
        [0.45, 0.40, 0.30, 1.0],
    ]

    shelf_color = random.choice(shelf_palettes)

    left_width = random.uniform(2.0, 2.8)
    right_width = random.uniform(2.0, 2.8)

    _add_shelf_row(
        -1,
        aisle_length,
        -left_width,
        shelf_height,
        shelf_color
    )

    _add_shelf_row(
        -1,
        aisle_length,
        right_width,
        shelf_height,
        shelf_color
    )

    # ---------- Hazard tape ----------
    red = [0.9, 0.1, 0.1, 1.0]

    _add_floor_tape(
        0,
        aisle_length,
        -0.8,
        red
    )

    _add_floor_tape(
        0,
        aisle_length,
        0.8,
        red
    )

    # ---------- Obstacle colors ----------
    obstacle_colors = [
        [0.60, 0.50, 0.35, 1.0],
        [0.55, 0.50, 0.30, 1.0],
        [0.45, 0.45, 0.45, 1.0],
        [0.70, 0.60, 0.30, 1.0],
    ]

    # ---------- Different blockage patterns ----------
    pattern = random.randint(0, 4)

    # ==================================================
    # Pattern 0 : CENTER BLOCKAGE
    # ==================================================
    if pattern == 0:

        for _ in range(random.randint(5, 10)):

            create_box(
                [
                    random.uniform(4, 10),
                    random.uniform(-0.35, 0.35),
                    random.uniform(0.25, 0.7)
                ],
                [
                    random.uniform(0.18, 0.35),
                    random.uniform(0.18, 0.35),
                    random.uniform(0.18, 0.35)
                ],
                random.choice(obstacle_colors)
            )

    # ==================================================
    # Pattern 1 : LEFT SIDE BLOCKED
    # ==================================================
    elif pattern == 1:

        for _ in range(random.randint(8, 14)):

            create_box(
                [
                    random.uniform(3, 11),
                    random.uniform(-1.3, -0.2),
                    random.uniform(0.25, 0.8)
                ],
                [
                    random.uniform(0.18, 0.40),
                    random.uniform(0.18, 0.40),
                    random.uniform(0.18, 0.40)
                ],
                random.choice(obstacle_colors)
            )

    # ==================================================
    # Pattern 2 : RIGHT SIDE BLOCKED
    # ==================================================
    elif pattern == 2:

        for _ in range(random.randint(8, 14)):

            create_box(
                [
                    random.uniform(3, 11),
                    random.uniform(0.2, 1.3),
                    random.uniform(0.25, 0.8)
                ],
                [
                    random.uniform(0.18, 0.40),
                    random.uniform(0.18, 0.40),
                    random.uniform(0.18, 0.40)
                ],
                random.choice(obstacle_colors)
            )

    # ==================================================
    # Pattern 3 : FULL BLOCKAGE WALL
    # ==================================================
    elif pattern == 3:

        for _ in range(random.randint(14, 22)):

            create_box(
                [
                    random.uniform(5, 9),
                    random.uniform(-1.2, 1.2),
                    random.uniform(0.25, 0.9)
                ],
                [
                    random.uniform(0.20, 0.45),
                    random.uniform(0.20, 0.45),
                    random.uniform(0.20, 0.45)
                ],
                random.choice(obstacle_colors)
            )

    # ==================================================
    # Pattern 4 : SCATTERED CARGO FIELD
    # ==================================================
    else:

        for _ in range(random.randint(12, 24)):

            create_box(
                [
                    random.uniform(3, 12),
                    random.uniform(-1.5, 1.5),
                    random.uniform(0.2, 0.8)
                ],
                [
                    random.uniform(0.15, 0.50),
                    random.uniform(0.15, 0.50),
                    random.uniform(0.15, 0.50)
                ],
                random.choice(obstacle_colors)
            )

    # ---------- Random stacked pallets ----------
    for _ in range(random.randint(2, 6)):

        x = random.uniform(4, 10)
        y = random.uniform(-1.2, 1.2)

        base_h = random.uniform(0.25, 0.5)
        top_h = random.uniform(0.15, 0.35)

        color = random.choice(obstacle_colors)

        create_box(
            [x, y, base_h / 2],
            [0.30, 0.25, base_h / 2],
            color
        )

        create_box(
            [x, y, base_h + top_h / 2],
            [0.24, 0.20, top_h / 2],
            color
        )

    # ---------- End wall ----------
    create_box(
        [aisle_length + 0.5, 0, shelf_height / 2],
        [0.1, 3.0, shelf_height / 2],
        [0.35, 0.35, 0.35, 1.0]
    )

    return _load_robot(
        start_pos=[0, 0, 0.5],
        start_yaw=0.0
    )



def create_cross_aisle_junction(seed=42):

    _setup_base(seed)
    random.seed(seed)

    shelf_height = random.uniform(0.8, 1.4)

    shelf_palettes = [
        [0.45, 0.45, 0.45, 1.0],
        [0.35, 0.50, 0.35, 1.0],
        [0.45, 0.40, 0.30, 1.0],
        [0.35, 0.35, 0.55, 1.0]
    ]

    shelf_color = random.choice(shelf_palettes)

    aisle_half_width = random.uniform(1.8, 2.8)

    junction_x = random.uniform(4.5, 8.5)

    junction_type = random.choice([
        "plus",
        "plus",
        "plus",
        "T",
        "wide"
    ])

    aisle_length = 18

    # ==================================================
    # Main aisle before intersection
    # ==================================================

    x = -1

    while x < junction_x:

        create_box(
            [x, -aisle_half_width, shelf_height / 2],
            [0.45, 0.3, shelf_height / 2],
            shelf_color
        )

        create_box(
            [x, aisle_half_width, shelf_height / 2],
            [0.45, 0.3, shelf_height / 2],
            shelf_color
        )

        x += 1.0

    # ==================================================
    # Cross aisle
    # ==================================================

    cross_width = random.uniform(2.5, 5.0)

    y = -8

    while y <= 8:

        if abs(y) > cross_width / 2:

            create_box(
                [junction_x, y, shelf_height / 2],
                [0.35, 0.45, shelf_height / 2],
                shelf_color
            )

            create_box(
                [junction_x + 4, y, shelf_height / 2],
                [0.35, 0.45, shelf_height / 2],
                shelf_color
            )

        y += 1.0

    # ==================================================
    # Continue aisle after intersection
    # ==================================================

    if junction_type == "plus":

        x = junction_x + 4

        while x < aisle_length:

            create_box(
                [x, -aisle_half_width, shelf_height / 2],
                [0.45, 0.3, shelf_height / 2],
                shelf_color
            )

            create_box(
                [x, aisle_half_width, shelf_height / 2],
                [0.45, 0.3, shelf_height / 2],
                shelf_color
            )

            x += 1.0

    elif junction_type == "T":

        create_box(
            [junction_x + 4.5, 0,
             shelf_height / 2],
            [0.15, 5.0, shelf_height / 2],
            shelf_color
        )

    elif junction_type == "wide":

        x = junction_x + 5

        while x < aisle_length:

            create_box(
                [x, -aisle_half_width, shelf_height / 2],
                [0.45, 0.3, shelf_height / 2],
                shelf_color
            )

            create_box(
                [x, aisle_half_width, shelf_height / 2],
                [0.45, 0.3, shelf_height / 2],
                shelf_color
            )

            x += 1.0

    # ==================================================
    # Safety posts
    # ==================================================

    post_color = [0.15, 0.15, 0.15, 1.0]
    cap_color = [1.0, 0.85, 0.0, 1.0]

    for px, py in [
        (junction_x - 0.5, -aisle_half_width),
        (junction_x - 0.5, aisle_half_width),
        (junction_x + 0.5, -aisle_half_width),
        (junction_x + 0.5, aisle_half_width)
    ]:

        create_cylinder(
            [px, py, 0.5],
            radius=0.06,
            height=1.0,
            color=post_color
        )

        create_box(
            [px, py, 0.75],
            [0.07, 0.07, 0.12],
            cap_color
        )

    # ==================================================
    # Floor markings
    # ==================================================

    white = [0.95, 0.95, 0.95, 1.0]

    create_box(
        [junction_x, 0, 0.012],
        [2.5, 0.08, 0.012],
        white
    )

    create_box(
        [junction_x, 0, 0.012],
        [0.08, cross_width / 2, 0.012],
        white
    )

    # ==================================================
    # Cargo near intersection
    # ==================================================

    cargo_colors = [
        [0.7, 0.5, 0.2, 1.0],
        [0.8, 0.2, 0.2, 1.0],
        [0.2, 0.5, 0.8, 1.0]
    ]

    for _ in range(random.randint(4, 10)):

        create_box(
            [
                random.uniform(junction_x - 2,
                               junction_x + 2),
                random.choice([
                    random.uniform(
                        aisle_half_width + 0.5,
                        aisle_half_width + 2
                    ),
                    random.uniform(
                        -(aisle_half_width + 2),
                        -(aisle_half_width + 0.5)
                    )
                ]),
                0.2
            ],
            [0.25, 0.25, 0.2],
            random.choice(cargo_colors)
        )

    return _load_robot(
        start_pos=[0, 0, 0.5],
        start_yaw=0.0
    )



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
    'seed_C': 256,
    'seed_D': 7,
    'seed_E': 1001,
    'seed_F': 64,
    'seed_G': 333,
    'seed_H': 89,
    'seed_I': 512,
    'seed_J': 2024,
}


def create_environment(class_idx, seed=42):
    
    if class_idx not in SCENE_CLASSES:
        raise ValueError(f"Unknown class index {class_idx}. Must be 0-4.")
    name, fn = SCENE_CLASSES[class_idx]
    return fn(seed=seed)



if __name__ == '__main__':
    import numpy as np
    import cv2
    import os

    os.makedirs('preview_images', exist_ok=True)
    print(f"Testing all 5 environments with {len(SEEDS)} seeds each...\n")

    for class_idx, (class_name, factory_fn) in SCENE_CLASSES.items():
        for seed_name, seed_val in SEEDS.items():
            client = p.connect(p.DIRECT)

            robot_id = factory_fn(seed=seed_val)
            p.stepSimulation()

            for link_idx in range(-1, p.getNumJoints(robot_id)):
                p.changeVisualShape(robot_id, link_idx, rgbaColor=[0, 0, 0, 0])

            pos, orient = p.getBasePositionAndOrientation(robot_id)
            rot = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)
            fwd = rot @ np.array([1, 0, 0])

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

    total_images = len(SCENE_CLASSES) * len(SEEDS)
    print(f"\nAll {total_images} preview images saved to preview_images/")
    print("Run scripts/generate_preview_grid.py to rebuild the comparison grid.")
    print("\nClass summary:")
    for idx, (name, _) in SCENE_CLASSES.items():
        print(f"  {idx}: {name}")