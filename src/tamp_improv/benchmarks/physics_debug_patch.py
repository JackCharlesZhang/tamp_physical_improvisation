"""Monkey-patch prbench to add debug logging without modifying their code."""

import types
import pymunk
from pymunk.vec2d import Vec2d
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prbench.envs.dynamic2d.utils import KinRobot

from tamp_improv.benchmarks.debug_physics import (
    log_collision,
    log_robot_revert,
    log_bounds_check,
    log_held_transition,
)

# Store original functions
_original_on_collision_w_static = None
_original_revert_to_last_state = None


def patched_on_collision_w_static(
    arbiter: pymunk.Arbiter, space: pymunk.Space, robot: "KinRobot"
) -> None:
    """Patched collision callback that logs before reverting."""
    # Get collision info
    shapes = arbiter.shapes
    if len(shapes) >= 2:
        robot_shape, static_shape = shapes[0], shapes[1]
        contact_point = arbiter.contact_point_set.points[0] if arbiter.contact_point_set.count > 0 else None

        if contact_point:
            pos = contact_point.point_a
            log_collision(
                collision_type="ROBOT_STATIC",
                obj_name="robot",
                wall_name="wall",
                position=(pos.x, pos.y)
            )

    # Call original function
    robot.revert_to_last_state()


def patched_revert_to_last_state(self: "KinRobot") -> None:
    """Patched revert function that logs held objects."""
    held_obj_names = [f"body_{obj[0].id}" for obj, _, _ in self.held_objects]

    log_robot_revert(
        reason="wall_collision",
        robot_pos=(self._base_body.position.x, self._base_body.position.y),
        held_objects=held_obj_names
    )

    # Call original implementation
    _original_revert_to_last_state(self)


def add_bounds_checking_to_env(env, world_bounds: tuple[float, float, float, float]) -> None:
    """Add bounds checking to environment step function.

    Args:
        env: The prbench environment instance
        world_bounds: (min_x, max_x, min_y, max_y)
    """
    # Handle different wrapper types
    if hasattr(env, '_object_centric_env'):
        # ConstantObjectPRBenchEnv wrapper
        target_env = env._object_centric_env
    elif hasattr(env, '_env'):
        # Other wrapper type
        target_env = env._env
    else:
        target_env = env

    # Get the UNBOUND method from the class to avoid capturing instance reference
    original_step = type(target_env).step

    def patched_step(self, action):
        # Call original step
        result = original_step(self, action)

        # Check bounds for all objects (skip walls and static objects)
        if hasattr(self, '_state_obj_to_pymunk_body'):
            for obj, body in self._state_obj_to_pymunk_body.items():
                # Skip walls, table, and target_surface (static objects)
                if 'wall' in obj.name.lower() or obj.name in ['table', 'target_surface']:
                    continue

                if hasattr(self, 'pymunk_space') and body in self.pymunk_space.bodies:
                    log_bounds_check(
                        obj_name=obj.name,
                        position=(body.position.x, body.position.y),
                        world_bounds=world_bounds,
                        margin=0.15  # Warn if within 0.15 units of boundary
                    )

        return result

    # Bind the patched function as a method to the instance
    target_env.step = types.MethodType(patched_step, target_env)


def add_held_state_logging_to_env(env) -> None:
    """Add logging when objects transition between DYNAMIC and KINEMATIC."""
    # Handle different wrapper types
    if hasattr(env, '_object_centric_env'):
        # ConstantObjectPRBenchEnv wrapper
        target_env = env._object_centric_env
    elif hasattr(env, '_env'):
        # Other wrapper type
        target_env = env._env
    else:
        target_env = env

    if not hasattr(target_env, '_add_state_to_space'):
        raise AttributeError(f"Environment {type(target_env).__name__} does not have _add_state_to_space")

    # Get the UNBOUND method from the class, not the bound method from the instance
    # This is critical for deepcopy to work correctly - we don't want to capture
    # a reference to a specific environment instance
    original_add_state = type(target_env)._add_state_to_space

    # Track previous states
    _obj_states = {}

    def patched_add_state(self, state):
        """Log state transitions before adding to space."""
        nonlocal _obj_states

        # Track which objects are held in this state
        for obj in state:
            if obj.name in ['robot', 'table'] or 'wall' in obj.name:
                continue

            try:
                held = state.get(obj, 'held')
                prev_state = _obj_states.get(obj.name, None)

                if prev_state is not None and prev_state != held:
                    log_held_transition(
                        obj_name=obj.name,
                        old_state="KINEMATIC" if prev_state else "DYNAMIC",
                        new_state="KINEMATIC" if held else "DYNAMIC",
                        collision_type="ROBOT" if held else "DYNAMIC"
                    )

                _obj_states[obj.name] = held
            except Exception:
                pass  # Skip objects without 'held' attribute

        # Call original with self explicitly to use the correct instance
        original_add_state(self, state)

    # Bind the patched function as a method to the instance
    target_env._add_state_to_space = types.MethodType(patched_add_state, target_env)


def patch_prbench_for_debugging(env, world_bounds: tuple[float, float, float, float]) -> None:
    """Apply all debug patches to the environment.

    Args:
        env: The prbench environment instance (ObjectCentricDynObstruction2DEnv)
        world_bounds: (min_x, max_x, min_y, max_y) for bounds checking
    """
    global _original_on_collision_w_static, _original_revert_to_last_state

    print("[DEBUG_PATCH] Applying debug patches to prbench environment...")

    # Patch collision handler
    try:
        from prbench.envs.dynamic2d import utils as prbench_utils
        _original_on_collision_w_static = prbench_utils.on_collision_w_static
        prbench_utils.on_collision_w_static = patched_on_collision_w_static
        print("[DEBUG_PATCH]   ✓ Collision handler patched")
    except Exception as e:
        print(f"[DEBUG_PATCH]   ✗ Failed to patch collision handler: {e}")

    # Patch robot revert function
    try:
        from prbench.envs.dynamic2d.utils import KinRobot
        _original_revert_to_last_state = KinRobot.revert_to_last_state
        KinRobot.revert_to_last_state = patched_revert_to_last_state
        print("[DEBUG_PATCH]   ✓ Robot revert function patched")
    except Exception as e:
        print(f"[DEBUG_PATCH]   ✗ Failed to patch revert function: {e}")

    # Add bounds checking
    try:
        add_bounds_checking_to_env(env, world_bounds)
        print(f"[DEBUG_PATCH]   ✓ Bounds checking added (bounds: {world_bounds})")
    except Exception as e:
        print(f"[DEBUG_PATCH]   ✗ Failed to add bounds checking: {e}")

    # Add held state logging
    try:
        add_held_state_logging_to_env(env)
        print("[DEBUG_PATCH]   ✓ Held state transitions logging added")
    except Exception as e:
        print(f"[DEBUG_PATCH]   ✗ Failed to add held state logging: {e}")

    print("[DEBUG_PATCH] Debug patching complete!\n")
