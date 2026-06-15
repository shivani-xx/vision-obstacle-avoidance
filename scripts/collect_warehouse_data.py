# scripts/collect_warehouse_data.py
# D5: Data Collection Script — Autonomous Inventory Robot Scene Classification
# Capstone Day 9 | Intern: Shivani | Purple AI Labs Ltd
#
# Collects 8,500 labelled frames across 5 scene classes × 2 seeds each.
#
# Expert Labeller Design (D4):
#   Ground truth label comes from the ENVIRONMENT TYPE, not raycasting.
#   The environment knows its own class — every frame captured inside
#   create_open_aisle() gets label 0, create_narrow_aisle() gets label 1, etc.
#   This is valid because each environment simulates exactly one scene type.
#   Transition zones are avoided by keeping the robot within the environment bounds.
#
# Usage:
#   conda activate research
#   cd C:\Users\shiva\Documents\vision-obstacle-avoidance
#   python scripts/collect_warehouse_data.py
#
# Output:
#   data/warehouse_dataset/images/   <- JPEG frames
#   data/warehouse_dataset/labels.csv  <- filename, class_idx, class_name, seed, env_run

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

# Add src/ to path so we can import warehouse_environments
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from warehouse_environments import SCENE_CLASSES, SEEDS, create_environment

# ===========================================================================
# CONFIGURATION
# ===========================================================================

OUTPUT_DIR      = os.path.join('data', 'warehouse_dataset')
IMG_DIR         = os.path.join(OUTPUT_DIR, 'images')
LABELS_CSV      = os.path.join(OUTPUT_DIR, 'labels.csv')

# Total target: 8,500 frames, ~1,700 per class
# 2 seeds per class → ~850 frames per seed
FRAMES_PER_SEED = 850

# Camera settings — MUST match warehouse_environments.py exactly
# These are the settings used in live deployment too (one physical camera)
IMG_WIDTH  = 224
IMG_HEIGHT = 224
CAM_FOV    = 70

# Robot movement parameters — varied to maximise visual diversity
BASE_SPEED    = 0.12     # Forward movement per step
TURN_RATE     = 0.12     # Yaw change per step when turning
RANDOM_TURN_P = 0.08     # Probability of random yaw nudge each step (adds diversity)
MAX_RANDOM_YAW = 0.05    # Max random yaw change


# ===========================================================================
# CAMERA — single consistent function (same as warehouse_environments.py)
# ===========================================================================

def capture_frame(robot_id, width=IMG_WIDTH, height=IMG_HEIGHT):
    """
    Capture one RGB frame from the robot's front-facing camera.
    Camera position and FOV MUST match the live deployment camera exactly.
    Robot body is hidden (alpha=0) so it never appears in training frames.
    """
    pos, orient = p.getBasePositionAndOrientation(robot_id)
    rot = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)
    fwd = rot @ np.array([1, 0, 0])

    # Camera raised above robot, slightly behind, looking ahead and slightly down
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
    """Make robot body transparent — must never appear in training frames."""
    for link_idx in range(-1, p.getNumJoints(robot_id)):
        p.changeVisualShape(robot_id, link_idx, rgbaColor=[0, 0, 0, 0])


# ===========================================================================
# EXPERT LABELLER (D4)
# ===========================================================================

def get_label(class_idx):
    """
    Expert labeller: the label IS the environment class index.

    Design decision:
      We know the ground truth label because WE built the environment.
      Every frame captured inside create_open_aisle() is definitionally
      an open aisle frame — no ambiguity, no raycasting needed.

    Edge case handling:
      - Robot is always reset to a valid position within the environment bounds.
      - If the robot drifts too close to a wall (position check), it is reset
        before a frame is captured — no ambiguous boundary frames are saved.
      - Transition frames don't exist here because each environment is a pure
        single-class scene (not a multi-zone map).
    """
    return class_idx


# ===========================================================================
# ROBOT MOVEMENT — varied positions/orientations for visual diversity
# ===========================================================================

def get_start_position(class_idx, seed):
    """
    Return a valid start position for the robot in this environment.
    Varied per seed to maximise visual diversity within the class.

    For junction (class 4): robot always starts at [4.0, 0, 0.5]
    because that's the junction centre (defined in the environment).
    For all others: start at [0, 0, 0.5] with small y jitter.
    """
    random.seed(seed * 100 + class_idx)  # Deterministic but varied
    if class_idx == 4:
        # Junction — must start inside the junction centre
        return [4.0, random.uniform(-0.3, 0.3), 0.5]
    else:
        return [0.0, random.uniform(-0.2, 0.2), 0.5]


def step_robot(x, y, angle, class_idx):
    """
    Move the robot one step. Movement strategy varies by class to produce
    visual diversity — different angles, positions, and distances.

    Returns updated (x, y, angle) and whether this position is valid.
    """
    # Random yaw nudge — adds natural camera angle variation
    if random.random() < RANDOM_TURN_P:
        angle += random.uniform(-MAX_RANDOM_YAW, MAX_RANDOM_YAW)

    # For junction — oscillate slightly left/right to see different branch angles
    if class_idx == 4:
        angle = math.sin(x * 0.3) * 0.15   # Gentle oscillation
        x += 0.05                            # Very slow forward drift
        if x > 5.5:
            x = 3.5                          # Reset to just before junction
    else:
        # For corridor classes — move forward with slight weaving
        x += BASE_SPEED * math.cos(angle)
        y += BASE_SPEED * math.sin(angle)

        # Clamp y to stay within aisle (avoid wall collision views)
        if class_idx == 1:   # Narrow aisle — tighter y bound
            y = max(-0.6, min(0.6, y))
        else:
            y = max(-1.2, min(1.2, y))

        # Reset if robot has moved too far forward
        if x > 14.0:
            x = 0.0
            y = random.uniform(-0.3, 0.3)
            angle = random.uniform(-0.1, 0.1)

        # For pick station — keep robot close to the station
        if class_idx == 2:
            x = max(0.0, min(3.0, x))

    return x, y, angle


# ===========================================================================
# COLLECTION LOOP
# ===========================================================================

def collect_from_class(class_idx, class_name, seed, frame_counter_start,
                        img_dir, target_frames):
    """
    Collect target_frames frames from one environment (class × seed).
    Returns list of record dicts.
    """
    client = p.connect(p.DIRECT)

    # Create the environment
    robot_id = create_environment(class_idx, seed=seed)
    p.stepSimulation()
    hide_robot(robot_id)

    # Start position
    x, y, angle = get_start_position(class_idx, seed)

    records = []
    frame_counter = frame_counter_start

    for i in range(target_frames):
        # Set robot position
        orient = p.getQuaternionFromEuler([0, 0, angle])
        p.resetBasePositionAndOrientation(robot_id, [x, y, 0.5], orient)
        p.stepSimulation()

        # Capture frame
        rgb = capture_frame(robot_id)

        # Save frame
        fname = f'cls{class_idx}_{class_name}_{frame_counter:06d}.jpg'
        fpath = os.path.join(img_dir, fname)
        cv2.imwrite(fpath, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Record label (D4 expert labeller)
        records.append({
            'filename':   fname,
            'class_idx':  get_label(class_idx),
            'class_name': class_name,
            'seed':       seed,
            'env_run':    f'{class_name}_seed{seed}',
        })

        # Move robot for next frame
        x, y, angle = step_robot(x, y, angle, class_idx)
        frame_counter += 1

        # Progress print every 100 frames
        if (i + 1) % 100 == 0:
            print(f'      {i+1}/{target_frames} frames', end='\r')

    p.disconnect()
    return records


def collect_full_dataset():
    """
    Collect complete warehouse dataset across all 5 classes × 2 seeds.
    Total target: 8,500 frames (~1,700 per class, ~850 per seed).
    """
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

    # Save labels CSV
    df = pd.DataFrame(all_records)
    df.to_csv(LABELS_CSV, index=False)

    total_elapsed = time.time() - start_time

    # Print final summary
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

    # Balance check (capstone requirement: ±15% per class)
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
        print('\n✓ All classes within ±15% balance requirement')
    else:
        print('\n✗ WARNING: Some classes outside ±15% — re-check FRAMES_PER_SEED')

    print('=' * 60)
    print('\nNext step: python scripts/validate_dataset.py')
    return df


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == '__main__':
    df = collect_full_dataset()