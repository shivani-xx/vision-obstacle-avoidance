# scripts/environments.py

import pybullet as p
import pybullet_data
import random
import math


def create_box(pos, size=[0.5, 0.5, 0.5], color=[0.8, 0.2, 0.2, 1]):

    col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=size
    )

    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=size,
        rgbaColor=color
    )

    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos
    )


def create_cylinder(pos, radius=0.3, height=0.8, color=[0.2, 0.6, 0.8, 1]):

    col = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=height
    )

    vis = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color
    )

    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos
    )


def setup_easy_environment(seed=42):

    """
    Wide corridor with few obstacles.
    Robot can mostly go straight.
    """

    random.seed(seed)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, -10)

    p.loadURDF('plane.urdf')

    # Wide corridor walls
    for i in range(25):

        create_box(
            [i - 5, -3, 0.5],
            [0.1, 0.1, 0.5],
            [0.6, 0.6, 0.6, 1]
        )

        create_box(
            [i - 5, 3, 0.5],
            [0.1, 0.1, 0.5],
            [0.6, 0.6, 0.6, 1]
        )

    # Few scattered obstacles
    for _ in range(random.randint(5, 8)):

        x = random.uniform(-3, 18)
        y = random.uniform(-2, 2)
        size = random.uniform(0.2, 0.4)

        color = [
            random.random(),
            random.random(),
            random.random(),
            1
        ]

        create_box(
            [x, y, size],
            [size, size, size],
            color
        )

    return 'easy'


def setup_medium_environment(seed=42):

    """
    Narrower passages with more obstacles.
    Robot must turn and navigate carefully.
    """

    random.seed(seed)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, -10)

    p.loadURDF('plane.urdf')

    # Narrower corridor walls
    for i in range(25):

        create_box(
            [i - 5, -2, 0.5],
            [0.1, 0.1, 0.5],
            [0.5, 0.5, 0.5, 1]
        )

        create_box(
            [i - 5, 2, 0.5],
            [0.1, 0.1, 0.5],
            [0.5, 0.5, 0.5, 1]
        )

    # Internal walls
    for i in range(3):

        wall_x = 3 + i * 5

        wall_y = random.choice([-0.8, 0.8])

        for j in range(4):

            create_box(
                [wall_x,
                 wall_y + j * 0.3 * (1 if wall_y > 0 else -1),
                 0.4],
                [0.15, 0.15, 0.4],
                [0.4, 0.4, 0.7, 1]
            )

    # Moderate obstacles
    for _ in range(random.randint(12, 18)):

        x = random.uniform(-3, 18)
        y = random.uniform(-1.5, 1.5)
        size = random.uniform(0.15, 0.35)

        color = [
            random.random(),
            random.random(),
            random.random(),
            1
        ]

        if random.random() > 0.5:

            create_box(
                [x, y, size],
                [size, size, size],
                color
            )

        else:

            create_cylinder(
                [x, y, size],
                radius=size,
                height=size * 2,
                color=color
            )

    return 'medium'


def setup_hard_environment(seed=42):

    """
    Dense clutter and maze-like layout.
    Very difficult navigation environment.
    """

    random.seed(seed)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, -10)

    p.loadURDF('plane.urdf')

    # Tight corridor walls
    for i in range(25):

        create_box(
            [i - 5, -1.5, 0.5],
            [0.1, 0.1, 0.5],
            [0.4, 0.4, 0.4, 1]
        )

        create_box(
            [i - 5, 1.5, 0.5],
            [0.1, 0.1, 0.5],
            [0.4, 0.4, 0.4, 1]
        )

    # Maze-like internal walls
    wall_positions = [
        (2, -0.5),
        (5, 0.5),
        (8, -0.3),
        (11, 0.7),
        (14, -0.6)
    ]

    for wx, wy in wall_positions:

        for j in range(5):

            create_box(
                [wx, wy + j * 0.25, 0.4],
                [0.12, 0.12, 0.4],
                [0.3, 0.3, 0.6, 1]
            )

    # Dense obstacles
    for _ in range(random.randint(25, 35)):

        x = random.uniform(-3, 18)
        y = random.uniform(-1.2, 1.2)
        size = random.uniform(0.1, 0.3)

        color = [
            random.uniform(0.2, 1),
            random.uniform(0.2, 1),
            random.uniform(0.2, 1),
            1
        ]

        shape = random.choice(['box', 'cylinder'])

        if shape == 'box':

            create_box(
                [x, y, size],
                [size, size * random.uniform(0.5, 1.5), size],
                color
            )

        else:

            create_cylinder(
                [x, y, size],
                radius=size,
                height=size * 2,
                color=color
            )

    return 'hard'

def create_environment(difficulty='medium', seed=42):

    """
    Factory function used by Day 5 deployment.
    Creates environment based on difficulty.
    """

    if difficulty == 'easy':
        return setup_easy_environment(seed)

    elif difficulty == 'medium':
        return setup_medium_environment(seed)

    elif difficulty == 'hard':
        return setup_hard_environment(seed)

    else:
        raise ValueError(
            f"Unknown difficulty: {difficulty}"
        )