import pybullet as p
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, 'src')

from terrain_environments import (
    create_terrain,
    TERRAIN_TYPES
)

def capture_robot_view(robot_id, width=224, height=224):
    """Capture camera image from robot's perspective."""

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

# Capture previews for all 4 terrain types

os.makedirs('results\\terrain_previews', exist_ok=True)

images = {}

for terrain in TERRAIN_TYPES:

    client = p.connect(p.DIRECT)

    robot_id = create_terrain(
        terrain,
        seed=42
    )

    img = capture_robot_view(robot_id)

    images[terrain] = img

    cv2.imwrite(
        f'results\\terrain_previews\\{terrain}.jpg',
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    )

    p.disconnect()

    print(f'  Captured {terrain}')

# Create 2x2 comparison grid

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 10)
)

for ax, (name, img) in zip(
    axes.flat,
    images.items()
):

    ax.imshow(img)

    ax.set_title(
        name.replace('_', ' ').title(),
        fontsize=14,
        fontweight='bold'
    )

    ax.axis('off')

plt.suptitle(
    'Capstone: 4 Terrain Types (Robot Camera View)',
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    'results\\terrain_previews\\comparison_grid.png',
    dpi=150
)

print(
    '\nPreview grid saved to results\\terrain_previews\\comparison_grid.png'
)