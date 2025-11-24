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
                return self.reset(seed=seed, options={"init_state": state})

            ObjectCentricDynamic2DRobotEnv.reset_from_state = reset_from_state_base
            print(
                "[PRBENCH_PATCH] Added reset_from_state to ObjectCentricDynamic2DRobotEnv"
            )

    except ImportError as e:
        print(f"[PRBENCH_PATCH] Warning: Could not patch prbench environments: {e}")
