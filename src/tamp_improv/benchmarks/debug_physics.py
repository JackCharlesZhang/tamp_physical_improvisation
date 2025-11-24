"""Debug utilities for physics collision and state tracking."""

import numpy as np
from numpy.typing import NDArray

# Debug flags - set these to enable different types of logging
DEBUG_COLLISIONS = True  # Log when objects collide with walls
DEBUG_HELD_OBJECTS = True  # Log when objects transition DYNAMIC <-> KINEMATIC
DEBUG_BOUNDS = True  # Log when objects are near or outside bounds
DEBUG_SKILL_PHASES = True  # Log skill execution phases (verbose)


def log_collision(collision_type: str, obj_name: str, wall_name: str, position: tuple[float, float]) -> None:
    """Log collision events."""
    if DEBUG_COLLISIONS:
        print(f"[COLLISION] {collision_type}: {obj_name} hit {wall_name} at ({position[0]:.3f}, {position[1]:.3f})")


def log_held_transition(obj_name: str, old_state: str, new_state: str, collision_type: str) -> None:
    """Log when object transitions between DYNAMIC and KINEMATIC."""
    if DEBUG_HELD_OBJECTS:
        print(f"[HELD_STATE] {obj_name}: {old_state} -> {new_state}, collision_type={collision_type}")


def log_bounds_check(
    obj_name: str,
    position: tuple[float, float],
    world_bounds: tuple[float, float, float, float],
    margin: float = 0.1
) -> None:
    """Log when objects are near or outside world bounds.

    Args:
        obj_name: Name of the object
        position: (x, y) position
        world_bounds: (min_x, max_x, min_y, max_y)
        margin: Distance from boundary to trigger warning
    """
    if not DEBUG_BOUNDS:
        return

    min_x, max_x, min_y, max_y = world_bounds
    x, y = position

    # Check if outside bounds
    if x < min_x or x > max_x or y < min_y or y > max_y:
        print(f"[BOUNDS ERROR] {obj_name} OUT OF BOUNDS at ({x:.3f}, {y:.3f})")
        print(f"  World bounds: x=[{min_x:.3f}, {max_x:.3f}], y=[{min_y:.3f}, {max_y:.3f}]")
        print(f"  Violations: x_min={x < min_x}, x_max={x > max_x}, y_min={y < min_y}, y_max={y > max_y}")

    # Check if near bounds
    elif (abs(x - min_x) < margin or abs(x - max_x) < margin or
          abs(y - min_y) < margin or abs(y - max_y) < margin):
        distances = {
            'left': x - min_x,
            'right': max_x - x,
            'bottom': y - min_y,
            'top': max_y - y
        }
        closest = min(distances.items(), key=lambda item: item[1])
        print(f"[BOUNDS WARNING] {obj_name} near {closest[0]} wall: distance={closest[1]:.3f} at ({x:.3f}, {y:.3f})")


def log_robot_revert(reason: str, robot_pos: tuple[float, float], held_objects: list[str]) -> None:
    """Log when robot reverts to last state due to collision."""
    if DEBUG_COLLISIONS:
        # print(f"[ROBOT_REVERT] {reason} at ({robot_pos[0]:.3f}, {robot_pos[1]:.3f})")
        if held_objects:
            # print(f"  Held objects also reverted: {held_objects}")
            pass


def log_skill_action(skill_name: str, phase: str, action: NDArray[np.float64], details: dict = None) -> None:
    """Log skill actions (optional, can be verbose)."""
    if not DEBUG_SKILL_PHASES:
        return

    dx, dy, dtheta, darm, dgripper = action
    action_str = f"dx={dx:.3f}, dy={dy:.3f}, dtheta={dtheta:.3f}, darm={darm:.3f}, dgripper={dgripper:.3f}"

    if details:
        details_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in details.items())
        print(f"[{skill_name}] Phase {phase}: {action_str} | {details_str}")
    else:
        print(f"[{skill_name}] Phase {phase}: {action_str}")
