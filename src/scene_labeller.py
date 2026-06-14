# src/scene_labeller.py

CLASS_NAMES = {
    0: 'open_aisle',
    1: 'narrow_aisle',
    2: 'pick_station',
    3: 'blocked_path',
    4: 'cross_aisle_junction',
}


def get_scene_label(class_idx):
    """
    Ground truth label for a warehouse frame.

    Since environments are fully controlled, the environment class
    is the ground truth label.
    """
    if class_idx not in CLASS_NAMES:
        raise ValueError(
            f"Unknown class_idx {class_idx}. Must be 0-4."
        )

    return class_idx

# src/scene_labeller.py

CLASS_NAMES = {
    0: 'open_aisle',
    1: 'narrow_aisle',
    2: 'pick_station',
    3: 'blocked_path',
    4: 'cross_aisle_junction',
}


def get_scene_label(class_idx):
    """
    Ground truth label for a warehouse frame.

    Since environments are fully controlled, the environment class
    is the ground truth label.
    """
    if class_idx not in CLASS_NAMES:
        raise ValueError(
            f"Unknown class_idx {class_idx}. Must be 0-4."
        )

    return class_idx


def get_class_name(class_idx):
    if class_idx not in CLASS_NAMES:
        raise ValueError(
            f"Unknown class_idx {class_idx}. Must be 0-4."
        )

    return CLASS_NAMES[class_idx]
def get_class_name(class_idx):
    if class_idx not in CLASS_NAMES:
        raise ValueError(
            f"Unknown class_idx {class_idx}. Must be 0-4."
        )

    return CLASS_NAMES[class_idx]