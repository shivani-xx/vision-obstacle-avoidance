# src/terrain_environments.py

import pybullet as p
import pybullet_data
import numpy as np
import random
import math


def create_box(pos, half_extents, color, orientation=None):
    """Create a static box in the simulation."""
    col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=half_extents
    )

    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color
    )

    orn = orientation if orientation else [0, 0, 0, 1]

    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        baseOrientation=orn
    )


def create_sphere(pos, radius, color):
    """Create a static sphere (for rough terrain debris)."""
    col = p.createCollisionShape(
        p.GEOM_SPHERE,
        radius=radius
    )

    vis = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color
    )

    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos
    )


def create_cylinder(pos, radius, height, color):
    """Create a static cylinder (for rough terrain debris)."""
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

def create_flat_ground(seed=42):
    """Clean, smooth ground with minimal decoration."""

    random.seed(seed)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, -10)

    # Ground plane
    p.loadURDF('plane.urdf')

    # Change ground colour to green (flat grass look)
    p.changeVisualShape(
        0,
        -1,
        rgbaColor=[0.35, 0.55, 0.30, 1]
    )

    # Add distant boundary walls so camera sees something
    for i in range(20):
        create_box(
            [i - 5, -5, 0.5],
            [0.1, 0.1, 0.5],
            [0.7, 0.7, 0.7, 1]
        )

        create_box(
            [i - 5, 5, 0.5],
            [0.1, 0.1, 0.5],
            [0.7, 0.7, 0.7, 1]
        )

    # Load robot
    robot_id = p.loadURDF(
        'r2d2.urdf',
        [0, 0, 0.5]
    )

    return robot_id

def create_uphill_slope(seed=42):
    """Ramp/incline ahead of the robot."""
    random.seed(seed)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.loadURDF('plane.urdf')
    
    # === KEY TECHNIQUE: Rotated box creates a ramp ===
    # Rotate around the Y-axis to tilt the box forward/backward
    # Pitch angle in radians: 0.2 rad = ~11 degrees, 0.3 rad = ~17 degrees
    pitch_angle = 0.25  # radians (~14 degree slope)
    ramp_orientation = p.getQuaternionFromEuler([0, pitch_angle, 0])
    
    # Create the ramp: a long, wide, thin box, tilted
    # Position it so the bottom edge touches the ground plane
    ramp_length = 6.0   # How long the ramp is
    ramp_width = 4.0    # How wide
    ramp_thickness = 0.1
    ramp_x = 6.0  # Place it ahead of the robot
    ramp_z = ramp_length * math.sin(pitch_angle) / 2 + ramp_thickness
    
    create_box(
        [ramp_x, 0, ramp_z],
        [ramp_length/2, ramp_width/2, ramp_thickness],
        [0.6, 0.5, 0.3, 1],  # Brown/tan colour for earth
        ramp_orientation
    )
    
    # Add visual cues: lighter colour strip at the top of the ramp
    top_x = ramp_x + ramp_length * math.cos(pitch_angle) / 2
    top_z = ramp_z + ramp_length * math.sin(pitch_angle) / 2
    create_box([top_x, 0, top_z], [0.3, ramp_width/2, 0.02], [0.8, 0.7, 0.5, 1])
    
    # Side walls to frame the slope
    for i in range(10):
        x = i * 1.0
        create_box([x, -2.5, 0.3], [0.05, 0.05, 0.3], [0.5, 0.5, 0.5, 1])
        create_box([x,  2.5, 0.3], [0.05, 0.05, 0.3], [0.5, 0.5, 0.5, 1])
    
    robot_id = p.loadURDF('r2d2.urdf', [0, 0, 0.5])
    return robot_id

def create_rough_terrain(seed=42):
    """Scattered debris and small objects on the ground."""

    random.seed(seed)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)

    p.loadURDF('plane.urdf')

    # Change ground to a darker/dirtier colour
    p.changeVisualShape(
        0,
        -1,
        rgbaColor=[0.4, 0.35, 0.25, 1]
    )

    # Scatter 40–60 small debris objects
    debris_count = random.randint(40, 60)

    earth_colours = [
        [0.5, 0.4, 0.3, 1],      # Brown
        [0.45, 0.45, 0.40, 1],   # Grey-brown
        [0.55, 0.50, 0.35, 1],   # Tan
        [0.35, 0.40, 0.30, 1],   # Dark green-brown
        [0.50, 0.50, 0.50, 1],   # Grey rock
    ]

    for _ in range(debris_count):

        x = random.uniform(1, 15)
        y = random.uniform(-3, 3)

        colour = random.choice(earth_colours)

        # Randomly choose shape: small cube, sphere, or cylinder
        shape_type = random.choice(
            ['box', 'sphere', 'cylinder']
        )

        size = random.uniform(0.05, 0.2)

        if shape_type == 'box':
            create_box(
                [x, y, size],
                [size, size, size],
                colour
            )

        elif shape_type == 'sphere':
            create_sphere(
                [x, y, size],
                size,
                colour
            )

        else:
            create_cylinder(
                [x, y, size],
                size * 0.7,
                size * 2,
                colour
            )

    # Add a few larger rocks for visual variety
    for _ in range(5):

        x = random.uniform(2, 12)
        y = random.uniform(-2, 2)

        size = random.uniform(0.2, 0.4)

        create_box(
            [x, y, size],
            [size, size * 0.8, size * 0.6],
            [0.45, 0.42, 0.40, 1]
        )

    robot_id = p.loadURDF(
        'r2d2.urdf',
        [0, 0, 0.5]
    )

    return robot_id

def create_hazard(seed=42):
    """Visible pit/gap/danger zone ahead of the robot."""

    random.seed(seed)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)

    p.loadURDF('plane.urdf')

    # Place a dark-coloured flat rectangle on the ground = visual "pit"
    # This simulates a hole or gap that the robot camera can see

    pit_x = random.uniform(3, 6)
    pit_width = random.uniform(2.0, 4.0)
    pit_depth_visual = random.uniform(1.5, 3.0)

    # The "pit" is a very flat dark box sitting on the ground

    create_box(
        [pit_x, 0, 0.005],
        [pit_depth_visual / 2, pit_width / 2, 0.005],
        [0.08, 0.05, 0.05, 1],
    )

    # Add warning edges: red/yellow strips around the pit

    create_box(
        [pit_x - pit_depth_visual / 2 - 0.1, 0, 0.01],
        [0.1, pit_width / 2, 0.01],
        [0.9, 0.2, 0.1, 1]
    )

    create_box(
        [pit_x + pit_depth_visual / 2 + 0.1, 0, 0.01],
        [0.1, pit_width / 2, 0.01],
        [0.9, 0.8, 0.1, 1]
    )

    # Add some normal ground features before the pit

    for i in range(5):

        create_box(
            [i * 0.5, -2, 0.2],
            [0.05, 0.05, 0.2],
            [0.6, 0.6, 0.6, 1]
        )

        create_box(
            [i * 0.5, 2, 0.2],
            [0.05, 0.05, 0.2],
            [0.6, 0.6, 0.6, 1]
        )

    robot_id = p.loadURDF(
        'r2d2.urdf',
        [0, 0, 0.5]
    )

    return robot_id

TERRAIN_TYPES = [
    'flat_ground',
    'uphill_slope',
    'rough_terrain',
    'hazard'
]


def create_terrain(terrain_type, seed=42):
    """Factory function to create any terrain type."""

    creators = {
        'flat_ground': create_flat_ground,
        'uphill_slope': create_uphill_slope,
        'rough_terrain': create_rough_terrain,
        'hazard': create_hazard,
    }

    if terrain_type not in creators:
        raise ValueError(
            f'Unknown terrain: {terrain_type}. Use: {TERRAIN_TYPES}'
        )

    return creators[terrain_type](seed)