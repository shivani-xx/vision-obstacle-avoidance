# src/terrain_labeller.py

def get_terrain_label(
    terrain_type,
    robot_x,
    terrain_start_x=2.0
):
    """
    Returns the terrain label based on environment type and robot position.

    If the robot hasn't reached the terrain feature yet (still on the
    flat approach section), returns 'flat_ground' regardless of environment.

    Once the robot passes terrain_start_x, returns the actual terrain type.

    Args:
        terrain_type: The environment type
        robot_x: Current x-position of the robot
        terrain_start_x: x-position where terrain feature begins

    Returns:
        Label string
    """

    # If robot is still in the approach zone,
    # it's seeing flat ground

    if (
        robot_x < terrain_start_x
        and terrain_type != 'flat_ground'
    ):
        return 'flat_ground'

    return terrain_type