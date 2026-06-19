# scripts/collect_warehouse_data.py

import pybullet as p
import pybullet_data
import numpy as np
import cv2
import os
import pandas as pd
import time
import math
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from warehouse_environments import SCENE_CLASSES, SEEDS, create_environment


OUTPUT_DIR      = os.path.join('data', 'warehouse_dataset')
IMG_DIR         = os.path.join(OUTPUT_DIR, 'images')
LABELS_CSV      = os.path.join(OUTPUT_DIR, 'labels.csv')

FRAMES_PER_SEED = 850

IMG_WIDTH  = 224
IMG_HEIGHT = 224
CAM_FOV    = 70

BASE_SPEED    = 0.12     
TURN_RATE     = 0.12    
RANDOM_TURN_P = 0.08     
MAX_RANDOM_YAW = 0.05    


def capture_frame(robot_id, width=IMG_WIDTH, height=IMG_HEIGHT):
    
    pos, orient = p.getBasePositionAndOrientation(robot_id)
    rot = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)
    fwd = rot @ np.array([1, 0, 0])

    cam_pos = [
        pos[0] - 0.3 * fwd[0],
        pos[1] - 0.3 * fwd[1],
        pos[2] + 0.8
    ]
    target = [
        pos[0] + fwd[0] * 3.0,
        pos[1] + fwd[1] * 3.0,
        pos[2] + 0.2
    ]

    view = p.computeViewMatrix(cam_pos, target, [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(
        fov=CAM_FOV, aspect=width / height,
        nearVal=0.1, farVal=100
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width, height, view, proj,
        renderer=p.ER_TINY_RENDERER
    )
    rgb = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    return rgb


def hide_robot(robot_id):
    
    for link_idx in range(-1, p.getNumJoints(robot_id)):
        p.changeVisualShape(robot_id, link_idx, rgbaColor=[0, 0, 0, 0])


def get_label(class_idx):
    return class_idx


def get_start_position(class_idx, seed):
    
    random.seed(seed * 100 + class_idx) 
    if class_idx == 4:
        
        return [4.0, random.uniform(-0.3, 0.3), 0.5]
    else:
        return [0.0, random.uniform(-0.2, 0.2), 0.5]


def step_robot(x, y, angle, class_idx):
    
    if random.random() < RANDOM_TURN_P:
        angle += random.uniform(-MAX_RANDOM_YAW, MAX_RANDOM_YAW)

    if class_idx == 4:
        angle = math.sin(x * 0.3) * 0.15   
        x += 0.05                            
        if x > 5.5:
            x = 3.5                        
    else:
        
        x += BASE_SPEED * math.cos(angle)
        y += BASE_SPEED * math.sin(angle)

        
        if class_idx == 1:  
            y = max(-0.6, min(0.6, y))
        else:
            y = max(-1.2, min(1.2, y))

        if x > 14.0:
            x = 0.0
            y = random.uniform(-0.3, 0.3)
            angle = random.uniform(-0.1, 0.1)

        if class_idx == 2:
            x = max(0.0, min(3.0, x))

    return x, y, angle


def collect_from_class(class_idx, class_name, seed, frame_counter_start,
                        img_dir, target_frames):
    
    client = p.connect(p.DIRECT)

    robot_id = create_environment(class_idx, seed=seed)
    p.stepSimulation()
    hide_robot(robot_id)

    x, y, angle = get_start_position(class_idx, seed)

    records = []
    frame_counter = frame_counter_start

    for i in range(target_frames):
        
        orient = p.getQuaternionFromEuler([0, 0, angle])
        p.resetBasePositionAndOrientation(robot_id, [x, y, 0.5], orient)
        p.stepSimulation()

        rgb = capture_frame(robot_id)

        fname = f'cls{class_idx}_{class_name}_{frame_counter:06d}.jpg'
        fpath = os.path.join(img_dir, fname)
        cv2.imwrite(fpath, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        records.append({
            'filename':   fname,
            'class_idx':  get_label(class_idx),
            'class_name': class_name,
            'seed':       seed,
            'env_run':    f'{class_name}_seed{seed}',
        })

        x, y, angle = step_robot(x, y, angle, class_idx)
        frame_counter += 1

        
        if (i + 1) % 100 == 0:
            print(f'      {i+1}/{target_frames} frames', end='\r')

    p.disconnect()
    return records


def collect_full_dataset():
   
    os.makedirs(IMG_DIR, exist_ok=True)

    all_records = []
    global_frame_counter = 0
    start_time = time.time()

    print('=' * 60)
    print('Warehouse Scene Dataset Collection')
    print(f'Target: {len(SCENE_CLASSES)} classes × 2 seeds × {FRAMES_PER_SEED} frames')
    print(f'Total target: {len(SCENE_CLASSES) * 2 * FRAMES_PER_SEED} frames')
    print(f'Output: {OUTPUT_DIR}')
    print('=' * 60)

    for class_idx, (class_name, _) in SCENE_CLASSES.items():
        class_start = time.time()
        class_records = []

        for seed_name, seed_val in SEEDS.items():
            print(f'\n  Class {class_idx} ({class_name}) | {seed_name} (seed={seed_val})')
            records = collect_from_class(
                class_idx=class_idx,
                class_name=class_name,
                seed=seed_val,
                frame_counter_start=global_frame_counter,
                img_dir=IMG_DIR,
                target_frames=FRAMES_PER_SEED,
            )
            class_records.extend(records)
            all_records.extend(records)
            global_frame_counter += len(records)
            print(f'      Done: {len(records)} frames collected')

        class_elapsed = time.time() - class_start
        print(f'  Class {class_idx} total: {len(class_records)} frames '
              f'in {class_elapsed:.1f}s')

    df = pd.DataFrame(all_records)
    df.to_csv(LABELS_CSV, index=False)

    total_elapsed = time.time() - start_time

    print('\n' + '=' * 60)
    print('COLLECTION COMPLETE')
    print(f'Total frames:  {len(df)}')
    print(f'Total time:    {total_elapsed / 60:.1f} minutes')
    print(f'Labels CSV:    {LABELS_CSV}')
    print(f'Images dir:    {IMG_DIR}')
    print()
    print('Class distribution:')
    for class_idx, (class_name, _) in SCENE_CLASSES.items():
        count = (df['class_idx'] == class_idx).sum()
        pct = count / len(df) * 100
        bar = '█' * int(pct / 2)
        print(f'  [{class_idx}] {class_name:<25} {count:>5} frames  '
              f'({pct:.1f}%)  {bar}')

    print()
    expected_per_class = len(df) / len(SCENE_CLASSES)
    lower = expected_per_class * 0.85
    upper = expected_per_class * 1.15
    all_balanced = True
    for class_idx in SCENE_CLASSES:
        count = (df['class_idx'] == class_idx).sum()
        ok = lower <= count <= upper
        if not ok:
            all_balanced = False
        status = '✓' if ok else '✗ IMBALANCED'
        print(f'  Class {class_idx}: {count} frames — {status}')

    if all_balanced:
        print('\n All classes within ±15% balance requirement')
    else:
        print('\n WARNING: Some classes outside ±15% — re-check FRAMES_PER_SEED')

    print('=' * 60)
    print('\nNext step: python scripts/validate_dataset.py')
    return df


if __name__ == '__main__':
    df = collect_full_dataset()