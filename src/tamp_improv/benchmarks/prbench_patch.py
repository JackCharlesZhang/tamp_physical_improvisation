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
            print(f"[DEBUG] _add_state_to_space called with objects: {[obj.name for obj in state]}")
            print(f"[DEBUG] _current_state has: {[obj.name for obj in self._current_state] if self._current_state else 'None'}")
            print(f"[DEBUG] _initial_constant_state has: {[obj.name for obj in self._initial_constant_state] if self._initial_constant_state else 'None'}")
            result = original_add_state(state)
            print(f"[DEBUG] After _add_state_to_space, cache has: {[obj.name for obj in self._state_obj_to_pymunk_body.keys()]}")
            return result

        ObjectCentricDynObstruction2DEnv._add_state_to_space = debug_add_state_to_space
        print("[PRBENCH_PATCH] Added debug wrapper to _add_state_to_space")

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
                if isinstance(state, np.ndarray):
                    print(f"[DEBUG] reset_from_state: got numpy array of shape {state.shape}")
                    print(f"[DEBUG] _constant_objects: {[obj.name for obj in self._constant_objects]}")
                    # Devectorize to see what objects will be in the state
                    from relational_structs.spaces import ObjectCentricBoxSpace
                    if isinstance(self.observation_space, ObjectCentricBoxSpace):
                        devectorized = self.observation_space.devectorize(state)
                        print(f"[DEBUG] Devectorized state has objects: {[obj.name for obj in devectorized]}")
                else:
                    print(f"[DEBUG] reset_from_state: got ObjectCentricState with objects: {[obj.name for obj in state]}")
                return self.reset(seed=seed, options={"init_state": state})

            ConstantObjectPRBenchEnv.reset_from_state = reset_from_state_wrapper
            print("[PRBENCH_PATCH] Added reset_from_state to ConstantObjectPRBenchEnv")

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
                    print(f"[DEBUG] ObjectCentric reset_from_state: got numpy array of shape {state.shape}")
                    print("[DEBUG] ERROR: ObjectCentricDynamic2DRobotEnv.reset_from_state expects ObjectCentricState, not numpy array!")
                    print("[DEBUG] This means unwrapping went too deep - should stop at ConstantObjectPRBenchEnv wrapper")
                    # The ConstantObjectPRBenchEnv wrapper handles numpy array -> ObjectCentricState conversion
                    # We shouldn't be calling reset_from_state on the object-centric env directly with a numpy array
                    raise TypeError(
                        "ObjectCentricDynamic2DRobotEnv.reset_from_state expects ObjectCentricState, "
                        f"but got {type(state)}. The unwrapping logic went too deep."
                    )
                else:
                    print(f"[DEBUG] ObjectCentric reset_from_state: got state with objects: {[obj.name for obj in state]}")
                    print(f"[DEBUG] _initial_constant_state has: {[obj.name for obj in self._initial_constant_state] if self._initial_constant_state else 'None'}")
                    print(f"[DEBUG] _state_obj_to_pymunk_body has: {list(self._state_obj_to_pymunk_body.keys()) if hasattr(self, '_state_obj_to_pymunk_body') else 'No cache yet'}")

                    result = self.reset(seed=seed, options={"init_state": state})

                    print(f"[DEBUG] After reset, _state_obj_to_pymunk_body has: {[obj.name for obj in self._state_obj_to_pymunk_body.keys()]}")
                    print(f"[DEBUG] After reset, _current_state has: {[obj.name for obj in self._current_state]}")

                    return result

            ObjectCentricDynamic2DRobotEnv.reset_from_state = reset_from_state_base
            print(
                "[PRBENCH_PATCH] Added reset_from_state to ObjectCentricDynamic2DRobotEnv"
            )

    except ImportError as e:
        print(f"[PRBENCH_PATCH] Warning: Could not patch prbench environments: {e}")
