# src/scene_labeller.py
# Scene Labeller — Ground Truth Labels for Warehouse Scene Classification
# Capstone Days 8-12 | Intern: Shivani | Purple AI Labs Ltd
#
# Since environments are fully controlled (we know which factory function
# built the scene), the ground truth label is simply the class index passed
# to create_environment(). The only edge case handled is transition frames
# where the robot has not yet moved into a representative view of the scene.

import math


# ===========================================================================
# CLASS REGISTRY
# ===========================================================================

CLASS_NAMES = {
    0: 'open_aisle',
    1: 'narrow_aisle',
    2: 'pick_station',
    3: 'blocked_path',
    4: 'cross_aisle_junction',
}

# Robot spawn positions per class — must match warehouse_environments.py
SPAWN_POSITIONS = {
    0: [0.0, 0.0, 0.5],   # open_aisle
    1: [0.0, 0.0, 0.5],   # narrow_aisle
    2: [0.0, 0.0, 0.5],   # pick_station
    3: [0.0, 0.0, 0.5],   # blocked_path
    4: [4.0, 0.0, 0.5],   # cross_aisle_junction — spawns inside junction
}

# Frames collected within this XY distance from spawn are discarded.
# The robot needs a few steps before the camera shows a representative
# view of the scene. Discarding beats mislabelling.
TRANSITION_THRESHOLD = 0.5  # world units


# ===========================================================================
# LABELLER
# ===========================================================================

def get_label(class_idx, robot_pos):
    """
    Return the ground truth integer label for a frame, or -1 if the frame
    is a transition frame and should be discarded.

    Args:
        class_idx (int)        : Scene class (0-4). This IS the label —
                                 known at construction time because we
                                 control which factory function built the
                                 environment.
        robot_pos (list/tuple) : Current robot [x, y, z] in world coords.

    Returns:
        int: class_idx (0-4) for valid frames, -1 for transition frames.
    """
    if class_idx not in CLASS_NAMES:
        raise ValueError(f"Unknown class_idx {class_idx}. Must be 0-4.")

    spawn = SPAWN_POSITIONS[class_idx]

    # Distance in XY plane only — robot does not change height
    dist = math.sqrt(
        (robot_pos[0] - spawn[0]) ** 2 +
        (robot_pos[1] - spawn[1]) ** 2
    )

    if dist < TRANSITION_THRESHOLD:
        return -1

    return class_idx


def is_valid_frame(label):
    """Return True if label is a valid class (0-4), False if transition (-1)."""
    return label >= 0


def get_class_name(class_idx):
    """Return human-readable name for a class index."""
    if class_idx not in CLASS_NAMES:
        raise ValueError(f"Unknown class_idx {class_idx}. Must be 0-4.")
    return CLASS_NAMES[class_idx]