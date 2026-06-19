# src/warehouse_environments.py

import pybullet as p
import pybullet_data
import random
import math


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




def create_narrow_aisle(seed=42):
   
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



def create_pick_station(seed=42):
   
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



def create_blocked_path(seed=42):
    
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



def create_cross_aisle_junction(seed=42):
    
    _setup_base(seed)

    shelf_height  = random.uniform(0.75, 1.0)
    shelf_color   = [random.uniform(0.45, 0.6)] * 3 + [1.0]
    junction_type = 'plus' if seed % 2 == 0 else 'T'

    for xi in range(-2, 4):
        create_box([float(xi), -2.0, shelf_height / 2],
                   [0.45, 0.3, shelf_height / 2], shelf_color)
        create_box([float(xi),  2.0, shelf_height / 2],
                   [0.45, 0.3, shelf_height / 2], shelf_color)

    cap_color = [random.uniform(0.35, 0.5)] * 3 + [1.0]
    for x_cap in [3.5, 4.5]:
        create_box([x_cap, -2.0, shelf_height / 2],
                   [0.08, 0.5, shelf_height / 2], cap_color)
        create_box([x_cap,  2.0, shelf_height / 2],
                   [0.08, 0.5, shelf_height / 2], cap_color)

    for yi in range(2, 10):
        y = -float(yi)
        create_box([4.0, y, shelf_height / 2],
                   [0.45, 0.3, shelf_height / 2], shelf_color)
        create_box([8.0, y, shelf_height / 2],
                   [0.45, 0.3, shelf_height / 2], shelf_color)

    for yi in range(2, 10):
        y = float(yi)
        create_box([4.0, y, shelf_height / 2],
                   [0.45, 0.3, shelf_height / 2], shelf_color)
        create_box([8.0, y, shelf_height / 2],
                   [0.45, 0.3, shelf_height / 2], shelf_color)

    if junction_type == 'plus':
        for xi in range(5, 16):
            create_box([float(xi), -2.0, shelf_height / 2],
                       [0.45, 0.3, shelf_height / 2], shelf_color)
            create_box([float(xi),  2.0, shelf_height / 2],
                       [0.45, 0.3, shelf_height / 2], shelf_color)
    else:
        create_box([9.0, 0.0, shelf_height / 2],
                   [0.3, 5.0, shelf_height / 2], shelf_color)

    YELLOW = [1.0, 0.85, 0.0, 1.0]
    DARK   = [0.15, 0.15, 0.15, 1.0]
    for cx, cy in [(3.5, -1.8), (3.5, 1.8), (4.5, -1.8), (4.5, 1.8)]:
        create_cylinder([cx, cy, 0.5], radius=0.06, height=1.0, color=DARK)
        create_box([cx, cy, 0.75], [0.07, 0.07, 0.12], YELLOW)

    WHITE = [0.92, 0.92, 0.92, 1.0]
    create_box([4.0, 0.0, 0.012], [2.0, 0.10, 0.012], WHITE)
    create_box([4.0, 0.0, 0.012], [0.10, 2.5, 0.012], WHITE)

    return _load_robot(start_pos=[4.0, 0.0, 0.5], start_yaw=0.0)



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