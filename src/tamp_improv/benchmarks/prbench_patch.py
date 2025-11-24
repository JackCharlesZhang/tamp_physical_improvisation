"""Monkey-patch prbench environments to add reset_from_state method.

This patch adds the reset_from_state method to prbench environments
without modifying the prbench source code.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from relational_structs import ObjectCentricState


def patch_prbench_environments() -> None:
    """Add reset_from_state methods to prbench environment classes."""
    try:
        from prbench.core import ConstantObjectPRBenchEnv
        from prbench.envs.dynamic2d.base_env import ObjectCentricDynamic2DRobotEnv
        from prbench.envs.dynamic2d.dyn_obstruction2d import (
            ObjectCentricDynObstruction2DEnv,
        )

        # Patch _add_state_to_space to debug what's being added
        original_add_state = ObjectCentricDynObstruction2DEnv._add_state_to_space

        def debug_add_state_to_space(self, state):
            import traceback
            # print(f"\n[DEBUG] ===== _add_state_to_space called =====")
            # print(f"[DEBUG] _add_state_to_space called with objects: {[obj.name for obj in state]}")
            # print(f"[DEBUG] _current_state has: {[obj.name for obj in self._current_state] if self._current_state else 'None'}")
            # print(f"[DEBUG] _initial_constant_state has: {[obj.name for obj in self._initial_constant_state] if self._initial_constant_state else 'None'}")
            # print(f"[DEBUG] Cache dictionary ID: {id(self._state_obj_to_pymunk_body)}")
            # print(f"[DEBUG] self (environment) ID: {id(self)}")
            # print(f"[DEBUG] Traceback (last 5 frames):")
            # for line in traceback.format_stack()[-6:-1]:
            #     print(line.strip())

            # Check object IDs before adding
            # if self._current_state:
            #     obstruction_objs = [obj for obj in self._current_state if obj.name == 'obstruction0']
            #     if obstruction_objs:
            #         print(f"[DEBUG] obstruction0 object ID in _current_state: {id(obstruction_objs[0])}")

            result = original_add_state(self, state)  # Need to pass self explicitly

            # print(f"[DEBUG] After _add_state_to_space, cache has: {[obj.name for obj in self._state_obj_to_pymunk_body.keys()]}")
            # print(f"[DEBUG] Cache dictionary ID after: {id(self._state_obj_to_pymunk_body)}")
            # Check if obstruction0 is in cache with its ID
            # cache_obstruction = [obj for obj in self._state_obj_to_pymunk_body.keys() if obj.name == 'obstruction0']
            # if cache_obstruction:
            #     print(f"[DEBUG] obstruction0 object ID in cache: {id(cache_obstruction[0])}")
            # print(f"[DEBUG] ===== _add_state_to_space finished =====\n")

            return result

        ObjectCentricDynObstruction2DEnv._add_state_to_space = debug_add_state_to_space
        # print("[PRBENCH_PATCH] Added debug wrapper to _add_state_to_space")

        # Patch full_state property to debug what's being accessed
        original_full_state = ObjectCentricDynamic2DRobotEnv.full_state.fget

        def debug_full_state(self):
            # print(f"\n[DEBUG] ===== full_state property accessed =====")
            # print(f"[DEBUG] self (environment) ID: {id(self)}")
            # print(f"[DEBUG] _current_state: {[obj.name for obj in self._current_state] if self._current_state else 'None'}")
            # print(f"[DEBUG] _initial_constant_state: {[obj.name for obj in self._initial_constant_state] if self._initial_constant_state else 'None'}")
            result = original_full_state(self)
            # print(f"[DEBUG] full_state result: {[obj.name for obj in result]}")
            # print(f"[DEBUG] ===== full_state finished =====\n")
            return result

        ObjectCentricDynamic2DRobotEnv.full_state = property(debug_full_state)
        # print("[PRBENCH_PATCH] Added debug wrapper to full_state property")

        # Patch _read_state_from_space to debug right before the assertion
        original_read_state = ObjectCentricDynObstruction2DEnv._read_state_from_space

        def debug_read_state_from_space(self):
            # print(f"[DEBUG] _read_state_from_space called")
            # print(f"[DEBUG] _current_state has objects: {[obj.name for obj in self._current_state]}")
            # print(f"[DEBUG] cache has objects: {[obj.name for obj in self._state_obj_to_pymunk_body.keys()]}")
            # Check obstruction0 IDs
            # current_obstruction = [obj for obj in self._current_state if obj.name == 'obstruction0']
            # cache_obstruction = [obj for obj in self._state_obj_to_pymunk_body.keys() if obj.name == 'obstruction0']
            # if current_obstruction and cache_obstruction:
            #     print(f"[DEBUG] obstruction0 ID in _current_state: {id(current_obstruction[0])}")
            #     print(f"[DEBUG] obstruction0 ID in cache: {id(cache_obstruction[0])}")
            #     print(f"[DEBUG] Are they the same object? {current_obstruction[0] is cache_obstruction[0]}")
            #     print(f"[DEBUG] Is current_obstruction[0] in cache keys? {current_obstruction[0] in self._state_obj_to_pymunk_body}")
            return original_read_state(self)

        ObjectCentricDynObstruction2DEnv._read_state_from_space = debug_read_state_from_space
        # print("[PRBENCH_PATCH] Added debug wrapper to _read_state_from_space")

        # Patch reset to debug when cache is cleared
        original_reset = ObjectCentricDynamic2DRobotEnv.reset

        def debug_reset(self, *, seed=None, options=None):
            import traceback
            # print(f"\n[DEBUG] ===== reset() called =====")
            # print(f"[DEBUG] self (environment) ID in reset: {id(self)}")
            # print(f"[DEBUG] Cache dict ID before reset: {id(self._state_obj_to_pymunk_body) if hasattr(self, '_state_obj_to_pymunk_body') else 'No cache'}")
            # print(f"[DEBUG] Traceback:")
            # for line in traceback.format_stack()[:-1]:
            #     print(line.strip())
            # print(f"[DEBUG] options: {options}")
            # print(f"[DEBUG] Cache before reset: {[obj.name for obj in self._state_obj_to_pymunk_body.keys()] if hasattr(self, '_state_obj_to_pymunk_body') and self._state_obj_to_pymunk_body else 'Empty or uninitialized'}")
            result = original_reset(self, seed=seed, options=options)
            # print(f"[DEBUG] self (environment) ID after reset: {id(self)}")
            # print(f"[DEBUG] Cache dict ID after reset: {id(self._state_obj_to_pymunk_body)}")
            # print(f"[DEBUG] Cache after reset: {[obj.name for obj in self._state_obj_to_pymunk_body.keys()]}")
            # print(f"[DEBUG] ===== reset() finished =====\n")
            return result

        ObjectCentricDynamic2DRobotEnv.reset = debug_reset
        # print("[PRBENCH_PATCH] Added debug wrapper to reset")

        # Patch _get_obs to see when observations are retrieved
        original_get_obs = ObjectCentricDynamic2DRobotEnv._get_obs

        def debug_get_obs(self):
            import traceback
            # print(f"\n[DEBUG] ===== _get_obs() called =====")
            # print(f"[DEBUG] self (environment) ID: {id(self)}")
            # print(f"[DEBUG] Cache dictionary ID: {id(self._state_obj_to_pymunk_body)}")
            # print(f"[DEBUG] Traceback (last 5 frames):")
            # for line in traceback.format_stack()[-6:-1]:
            #     print(line.strip())
            # print(f"[DEBUG] Cache before _read_state_from_space: {[obj.name for obj in self._state_obj_to_pymunk_body.keys()] if self._state_obj_to_pymunk_body else 'Empty'}")
            result = original_get_obs(self)
            # print(f"[DEBUG] ===== _get_obs() finished =====\n")
            return result

        ObjectCentricDynamic2DRobotEnv._get_obs = debug_get_obs
        # print("[PRBENCH_PATCH] Added debug wrapper to _get_obs")

        # Add clone() method to handle deepcopy issue with object identity in cache
        def clone_env(self):
            """Clone environment, properly handling the pymunk body cache.

            The issue: deepcopy creates new object instances for _current_state but
            the _state_obj_to_pymunk_body cache still has old object instances as keys.
            Since the cache uses object identity (id(obj)), lookups fail after deepcopy.

            Solution: Clear the cache after deepcopy. It will be rebuilt on next reset.
            """
            import copy
            # print("[DEBUG] clone() called - performing deepcopy")
            # print(f"[DEBUG] self type: {type(self).__name__}")
            # print(f"[DEBUG] self ID: {id(self)}")

            # Check if this is the wrapper or the inner env
            # if hasattr(self, '_object_centric_env'):
            #     inner_env = self._object_centric_env
            #     print(f"[DEBUG] Found inner _object_centric_env, ID: {id(inner_env)}")
            #     if hasattr(inner_env, '_state_obj_to_pymunk_body'):
            #         print(f"[DEBUG] Inner env cache before deepcopy: {[obj.name for obj in inner_env._state_obj_to_pymunk_body.keys()] if inner_env._state_obj_to_pymunk_body else 'Empty'}")
            # elif hasattr(self, '_state_obj_to_pymunk_body'):
            #     print(f"[DEBUG] Direct env cache before deepcopy: {[obj.name for obj in self._state_obj_to_pymunk_body.keys()] if self._state_obj_to_pymunk_body else 'Empty'}")

            cloned = copy.deepcopy(self)

            # BUG FIX: After deepcopy, the cloned wrapper's _object_centric_env might still
            # point to the original environment due to some reference issue in deepcopy.
            # We need to find the actual cloned inner environment and fix the reference.

            # Clear the pymunk cache in the cloned environment
            # Need to handle both the wrapper case (ConstantObjectPRBenchEnv) and direct env case
            if hasattr(cloned, '_object_centric_env'):
                # This is a wrapper (ConstantObjectPRBenchEnv)
                inner_cloned = cloned._object_centric_env
                # print(f"[DEBUG] Cloned wrapper, clearing inner env caches")
                # print(f"[DEBUG] Original inner env ID: {id(self._object_centric_env)}")
                # print(f"[DEBUG] Cloned inner env ID: {id(inner_cloned)}")
                # print(f"[DEBUG] Are they the same? {self._object_centric_env is inner_cloned}")

                # CRITICAL FIX: If deepcopy failed to copy _object_centric_env properly,
                # we need to find the actual cloned environment in the memo dict or
                # do a manual copy
                if self._object_centric_env is inner_cloned:
                    # print("[DEBUG] WARNING: Deepcopy failed to clone _object_centric_env!")
                    # print("[DEBUG] Manually deep copying the inner environment...")
                    cloned._object_centric_env = copy.deepcopy(self._object_centric_env)
                    inner_cloned = cloned._object_centric_env
                    # print(f"[DEBUG] New cloned inner env ID: {id(inner_cloned)}")

                if hasattr(inner_cloned, '_state_obj_to_pymunk_body'):
                    original_cache_id = id(self._object_centric_env._state_obj_to_pymunk_body)
                    cloned_cache_id = id(inner_cloned._state_obj_to_pymunk_body)
                    # print(f"[DEBUG] Original cache ID: {original_cache_id}")
                    # print(f"[DEBUG] Cloned cache ID before fix: {cloned_cache_id}")
                    # print(f"[DEBUG] Are they the same dict? {original_cache_id == cloned_cache_id}")
                    # print(f"[DEBUG] Clearing inner _state_obj_to_pymunk_body (had {len(inner_cloned._state_obj_to_pymunk_body)} entries)")
                    # # CRITICAL: Create a NEW dict object, don't just clear the existing one
                    # If deepcopy failed to create separate dicts, this ensures independence
                    inner_cloned._state_obj_to_pymunk_body = dict()
                    # print(f"[DEBUG] Cloned cache ID after creating new dict: {id(inner_cloned._state_obj_to_pymunk_body)}")
                if hasattr(inner_cloned, '_static_object_body_cache'):
                    # print(f"[DEBUG] Clearing inner _static_object_body_cache")
                    inner_cloned._static_object_body_cache = dict()
            else:
                # This is the direct ObjectCentricDynamic2DRobotEnv
                # print(f"[DEBUG] Cloned direct env, clearing caches")
                if hasattr(cloned, '_state_obj_to_pymunk_body'):
                    original_cache_id = id(self._state_obj_to_pymunk_body)
                    cloned_cache_id = id(cloned._state_obj_to_pymunk_body)
                    # print(f"[DEBUG] Original cache ID: {original_cache_id}")
                    # print(f"[DEBUG] Cloned cache ID before fix: {cloned_cache_id}")
                    # print(f"[DEBUG] Are they the same dict? {original_cache_id == cloned_cache_id}")
                    # print(f"[DEBUG] Clearing _state_obj_to_pymunk_body (had {len(cloned._state_obj_to_pymunk_body)} entries)")
                    # CRITICAL: Create a NEW dict object to ensure independence
                    cloned._state_obj_to_pymunk_body = dict()
                    # print(f"[DEBUG] Cloned cache ID after creating new dict: {id(cloned._state_obj_to_pymunk_body)}")
                if hasattr(cloned, '_static_object_body_cache'):
                    # print(f"[DEBUG] Clearing _static_object_body_cache")
                    cloned._static_object_body_cache = dict()

            # print("[DEBUG] clone() complete - caches cleared in clone")
            return cloned

        # Add to both ConstantObjectPRBenchEnv and ObjectCentricDynamic2DRobotEnv
        ConstantObjectPRBenchEnv.clone = clone_env
        ObjectCentricDynamic2DRobotEnv.clone = clone_env
        # print("[PRBENCH_PATCH] Added clone() method to handle deepcopy cache issue")

        # Patch ConstantObjectPRBenchEnv
        if not hasattr(ConstantObjectPRBenchEnv, "reset_from_state"):

            def reset_from_state_wrapper(
                self, state: NDArray[Any] | ObjectCentricState, *, seed: int | None = None
            ) -> tuple[NDArray[Any], dict]:
                """Reset environment to a specific state.

                Args:
                    state: Either a vectorized state (NDArray) or an ObjectCentricState
                    seed: Optional random seed

                Returns:
                    Tuple of (observation, info)
                """
                # Debug: check what kind of state we're getting
                # print(f"[DEBUG] ConstantObjectPRBenchEnv.reset_from_state called")
                # print(f"[DEBUG] self (wrapper) ID: {id(self)}")
                # print(f"[DEBUG] self._object_centric_env ID: {id(self._object_centric_env)}")
                # if isinstance(state, np.ndarray):
                #     print(f"[DEBUG] reset_from_state: got numpy array of shape {state.shape}")
                #     print(f"[DEBUG] _constant_objects: {[obj.name for obj in self._constant_objects]}")
                #     # Devectorize to see what objects will be in the state
                #     from relational_structs.spaces import ObjectCentricBoxSpace
                #     if isinstance(self.observation_space, ObjectCentricBoxSpace):
                #         devectorized = self.observation_space.devectorize(state)
                #         print(f"[DEBUG] Devectorized state has objects: {[obj.name for obj in devectorized]}")
                # else:
                #     print(f"[DEBUG] reset_from_state: got ObjectCentricState with objects: {[obj.name for obj in state]}")
                return self.reset(seed=seed, options={"init_state": state})

            ConstantObjectPRBenchEnv.reset_from_state = reset_from_state_wrapper
            # print("[PRBENCH_PATCH] Added reset_from_state to ConstantObjectPRBenchEnv")

        # Patch ObjectCentricDynamic2DRobotEnv
        if not hasattr(ObjectCentricDynamic2DRobotEnv, "reset_from_state"):

            def reset_from_state_base(
                self, state: ObjectCentricState | NDArray[Any], *, seed: int | None = None
            ) -> tuple[ObjectCentricState, dict]:
                """Reset environment to a specific state.

                Args:
                    state: ObjectCentricState or numpy array to reset to
                    seed: Optional random seed

                Returns:
                    Tuple of (observation, info)
                """
                # Debug: check what we got
                if isinstance(state, np.ndarray):
                    # print(f"[DEBUG] ObjectCentric reset_from_state: got numpy array of shape {state.shape}")
                    # print("[DEBUG] ERROR: ObjectCentricDynamic2DRobotEnv.reset_from_state expects ObjectCentricState, not numpy array!")
                    # print("[DEBUG] This means unwrapping went too deep - should stop at ConstantObjectPRBenchEnv wrapper")
                    # The ConstantObjectPRBenchEnv wrapper handles numpy array -> ObjectCentricState conversion
                    # We shouldn't be calling reset_from_state on the object-centric env directly with a numpy array
                    raise TypeError(
                        "ObjectCentricDynamic2DRobotEnv.reset_from_state expects ObjectCentricState, "
                        f"but got {type(state)}. The unwrapping logic went too deep."
                    )
                else:
                    print(f"\n[PHYSICS_STATE] ===== BEFORE reset_from_state =====")
                    print(f"[PHYSICS_STATE] Input state robot: x={state.get('robot', 'x'):.4f}, y={state.get('robot', 'y'):.4f}, theta={state.get('robot', 'theta'):.4f}")

                    # Capture physics state BEFORE reset
                    if hasattr(self, '_state_obj_to_pymunk_body') and 'robot' in [obj.name for obj in self._state_obj_to_pymunk_body.keys()]:
                        robot_obj = next(obj for obj in self._state_obj_to_pymunk_body.keys() if obj.name == 'robot')
                        robot_body = self._state_obj_to_pymunk_body[robot_obj]
                        print(f"[PHYSICS_STATE] BEFORE PyMunk robot body: pos=({robot_body.position.x:.4f}, {robot_body.position.y:.4f}), angle={robot_body.angle:.4f}")
                        print(f"[PHYSICS_STATE] BEFORE PyMunk robot velocity: ({robot_body.velocity.x:.4f}, {robot_body.velocity.y:.4f}), angular_vel={robot_body.angular_velocity:.4f}")
                        print(f"[PHYSICS_STATE] BEFORE PyMunk robot body_type: {robot_body.body_type}")

                    result = self.reset(seed=seed, options={"init_state": state})

                    print(f"\n[PHYSICS_STATE] ===== AFTER reset_from_state =====")
                    obs_state = result[0]
                    print(f"[PHYSICS_STATE] Output obs robot: x={obs_state.get('robot', 'x'):.4f}, y={obs_state.get('robot', 'y'):.4f}, theta={obs_state.get('robot', 'theta'):.4f}")

                    # Capture physics state AFTER reset
                    if hasattr(self, '_state_obj_to_pymunk_body') and 'robot' in [obj.name for obj in self._state_obj_to_pymunk_body.keys()]:
                        robot_obj = next(obj for obj in self._state_obj_to_pymunk_body.keys() if obj.name == 'robot')
                        robot_body = self._state_obj_to_pymunk_body[robot_obj]
                        print(f"[PHYSICS_STATE] AFTER PyMunk robot body: pos=({robot_body.position.x:.4f}, {robot_body.position.y:.4f}), angle={robot_body.angle:.4f}")
                        print(f"[PHYSICS_STATE] AFTER PyMunk robot velocity: ({robot_body.velocity.x:.4f}, {robot_body.velocity.y:.4f}), angular_vel={robot_body.angular_velocity:.4f}")
                        print(f"[PHYSICS_STATE] AFTER PyMunk robot body_type: {robot_body.body_type}")

                        # Check for discrepancies
                        pos_diff_x = abs(robot_body.position.x - obs_state.get('robot', 'x'))
                        pos_diff_y = abs(robot_body.position.y - obs_state.get('robot', 'y'))
                        angle_diff = abs(robot_body.angle - obs_state.get('robot', 'theta'))
                        if pos_diff_x > 0.001 or pos_diff_y > 0.001 or angle_diff > 0.001:
                            print(f"[PHYSICS_STATE] ⚠️  MISMATCH: PyMunk body != ObjectCentricState!")
                            print(f"[PHYSICS_STATE]   Position diff: ({pos_diff_x:.6f}, {pos_diff_y:.6f}), angle diff: {angle_diff:.6f}")
                    print(f"[PHYSICS_STATE] =====================================\n")

                    return result

            ObjectCentricDynamic2DRobotEnv.reset_from_state = reset_from_state_base
            # print(
            #     "[PRBENCH_PATCH] Added reset_from_state to ObjectCentricDynamic2DRobotEnv"
            # )

    except ImportError as e:
        # print(f"[PRBENCH_PATCH] Warning: Could not patch prbench environments: {e}")
        pass
