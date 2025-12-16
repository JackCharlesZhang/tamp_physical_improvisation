"""Monkey-patch prbench environments to add reset_from_state method.

This patch adds the reset_from_state method to prbench environments
without modifying the prbench source code.
"""

from typing import Any
import copy

import numpy as np
from numpy.typing import NDArray
from relational_structs import ObjectCentricState

from prbench.core import ConstantObjectPRBenchEnv
from prbench.envs.dynamic2d.base_env import ObjectCentricDynamic2DRobotEnv
from prbench.envs.dynamic2d.dyn_obstruction2d import (
    ObjectCentricDynObstruction2DEnv,
)

def patch_prbench_environments() -> None:
    """Add reset_from_state methods to prbench environment classes."""
    
    # Add clone() method to handle deepcopy issue with object identity in cache
    def clone_env(self):
        """Clone environment, properly handling the pymunk body cache.

        The issue: deepcopy creates new object instances for _current_state but
        the _state_obj_to_pymunk_body cache still has old object instances as keys.
        Since the cache uses object identity (id(obj)), lookups fail after deepcopy.

        Solution: Clear the cache after deepcopy. It will be rebuilt on next reset.
        """
        cloned = copy.deepcopy(self)

        # BUG FIX: After deepcopy, the cloned wrapper's _object_centric_env might still
        # point to the original environment due to some reference issue in deepcopy.
        # We need to find the actual cloned inner environment and fix the reference.

        # Clear the pymunk cache in the cloned environment
        # Need to handle both the wrapper case (ConstantObjectPRBenchEnv) and direct env case
        if hasattr(cloned, '_object_centric_env'):
            # This is a wrapper (ConstantObjectPRBenchEnv)
            inner_cloned = cloned._object_centric_env

            # CRITICAL FIX: If deepcopy failed to copy _object_centric_env properly,
            # we need to find the actual cloned environment in the memo dict or
            # do a manual copy
            if self._object_centric_env is inner_cloned:
                cloned._object_centric_env = copy.deepcopy(self._object_centric_env)
                inner_cloned = cloned._object_centric_env

            if hasattr(inner_cloned, '_state_obj_to_pymunk_body'):
                # CRITICAL: Create a NEW dict object, don't just clear the existing one
                # If deepcopy failed to create separate dicts, this ensures independence
                inner_cloned._state_obj_to_pymunk_body = dict()
            if hasattr(inner_cloned, '_static_object_body_cache'):
                inner_cloned._static_object_body_cache = dict()
        else:
            # This is the direct ObjectCentricDynamic2DRobotEnv
            if hasattr(cloned, '_state_obj_to_pymunk_body'):
                # CRITICAL: Create a NEW dict object to ensure independence
                cloned._state_obj_to_pymunk_body = dict()
            if hasattr(cloned, '_static_object_body_cache'):
                cloned._static_object_body_cache = dict()

        return cloned

    # Add to both ConstantObjectPRBenchEnv and ObjectCentricDynamic2DRobotEnv
    ConstantObjectPRBenchEnv.clone = clone_env
    ObjectCentricDynamic2DRobotEnv.clone = clone_env

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
            if isinstance(state, np.ndarray):
                raise TypeError(
                    "ObjectCentricDynamic2DRobotEnv.reset_from_state expects ObjectCentricState, "
                    f"but got {type(state)}. The unwrapping logic went too deep."
                )
            else:
                result = self.reset(seed=seed, options={"init_state": state})
                return result

        ObjectCentricDynamic2DRobotEnv.reset_from_state = reset_from_state_base


