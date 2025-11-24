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
                else:
                    print(f"[DEBUG] reset_from_state: got ObjectCentricState with objects: {[obj.name for obj in state]}")
                return self.reset(seed=seed, options={"init_state": state})

            ConstantObjectPRBenchEnv.reset_from_state = reset_from_state_wrapper
            print("[PRBENCH_PATCH] Added reset_from_state to ConstantObjectPRBenchEnv")

        # Patch ObjectCentricDynamic2DRobotEnv
        if not hasattr(ObjectCentricDynamic2DRobotEnv, "reset_from_state"):

            def reset_from_state_base(
                self, state: ObjectCentricState, *, seed: int | None = None
            ) -> tuple[ObjectCentricState, dict]:
                """Reset environment to a specific state.

                Args:
                    state: ObjectCentricState to reset to
                    seed: Optional random seed

                Returns:
                    Tuple of (observation, info)
                """
                # Debug: check objects in state
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
