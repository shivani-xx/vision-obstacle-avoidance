import pybullet as p
import numpy as np
import cv2
import os
import pandas as pd
import time
import sys
import math

sys.path.insert(0, 'src')

from terrain_environments import (
    create_terrain,
    TERRAIN_TYPES
)

from terrain_labeller import (
    get_terrain_label
)

def capture_robot_view(robot_id, width=224, height=224):

    pos, quat = p.getBasePositionAndOrientation(robot_id)

    rot = np.array(
        p.getMatrixFromQuaternion(quat)
    ).reshape(3, 3)

    cam = [
        pos[0],
        pos[1],
        pos[2] + 0.3
    ]

    fwd = rot @ np.array([1, 0, 0])

    tgt = [
        cam[0] + fwd[0],
        cam[1] + fwd[1],
        cam[2] + fwd[2]
    ]

    view = p.computeViewMatrix(
        cam,
        tgt,
        [0, 0, 1]
    )

    proj = p.computeProjectionMatrixFOV(
        60,
        width / height,
        0.1,
        100
    )

    _, _, rgba, _, _ = p.getCameraImage(
        width,
        height,
        view,
        proj,
        renderer=p.ER_TINY_RENDERER
    )

    return np.array(
        rgba,
        dtype=np.uint8
    ).reshape(height, width, 4)[:, :, :3]

def collect_for_terrain(
    terrain_type,
    seeds,
    frames_per_seed,
    output_dir
):
    """Collect frames for a single terrain type across multiple seeds."""

    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    records = []

    for seed in seeds:

        client = p.connect(p.DIRECT)

        robot_id = create_terrain(
            terrain_type,
            seed=seed
        )

        x, y, angle = 0.0, 0.0, 0.0

        speed = 0.12

        for i in range(frames_per_seed):

            # Add slight random variation to y and angle
            y_offset = math.sin(i * 0.1) * 0.3
            angle_offset = math.sin(i * 0.07) * 0.1

            orient = p.getQuaternionFromEuler(
                [0, 0, angle + angle_offset]
            )

            p.resetBasePositionAndOrientation(
                robot_id,
                [x, y + y_offset, 0.5],
                orient
            )

            p.stepSimulation()

            # Capture frame
            img = capture_robot_view(robot_id)

            # Get label
            label = get_terrain_label(
                terrain_type,
                x,
                terrain_start_x=2.0
            )

            # Save image
            fname = f'{terrain_type}_s{seed}_{i:04d}.jpg'

            cv2.imwrite(
                os.path.join(img_dir, fname),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )

            records.append({
                'filename': fname,
                'terrain': label,
                'terrain_env': terrain_type,
                'seed': seed,
                'x': round(x, 2),
            })

            # Move forward
            x += speed

            # Reset if too far
            if x > 12:
                x = 0.0

        p.disconnect()

        print(
            f'    {terrain_type} seed={seed}: {frames_per_seed} frames'
        )

    return records

def collect_full_dataset():
    """Collect complete terrain dataset."""

    output_dir = 'data\\terrain_dataset'

    start = time.time()

    all_records = []

    # ~1,300 frames per terrain type × 4 types = ~5,200 total
    # 3 seeds per terrain for variety

    terrain_config = {
        'flat_ground': {
            'seeds': [42, 100, 200],
            'per_seed': 430
        },

        'uphill_slope': {
            'seeds': [42, 100, 200],
            'per_seed': 430
        },

        'rough_terrain': {
            'seeds': [42, 100, 200],
            'per_seed': 430
        },

        'hazard': {
            'seeds': [42, 100, 200],
            'per_seed': 430
        },
    }

    for terrain, cfg in terrain_config.items():

        print(f'  Collecting {terrain}...')

        records = collect_for_terrain(
            terrain,
            cfg['seeds'],
            cfg['per_seed'],
            output_dir
        )

        all_records.extend(records)

    # Save labels CSV

    df = pd.DataFrame(all_records)

    df.to_csv(
        os.path.join(output_dir, 'labels.csv'),
        index=False
    )

    elapsed = time.time() - start

    print(
        f'\nDataset collected: {len(df)} frames in {elapsed/60:.1f} minutes'
    )

    print('\nLabel distribution:')

    print(df['terrain'].value_counts())

    return df


if __name__ == '__main__':
    collect_full_dataset()