"""Wrapper for ObjectCentricClutteredStorage2DEnv to add reset_from_state."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from prbench.envs.geom2d.clutteredstorage2d import ObjectCentricClutteredStorage2DEnv
from relational_structs import ObjectCentricState


class ClutteredStorage2DEnvWrapper(gym.Wrapper):
    """Wrapper that adds reset_from_state capability to ObjectCentricClutteredStorage2DEnv."""

    def __init__(self, env: ObjectCentricClutteredStorage2DEnv):
        """Initialize wrapper.

        Args:
            env: The base ObjectCentricClutteredStorage2DEnv to wrap
        """
        super().__init__(env)
        # Store reference to unwrapped env for type safety
        self.unwrapped_env = env

    def reset_from_state(
        self,
        state: ObjectCentricState,
        *,
        seed: int | None = None,
    ) -> tuple[ObjectCentricState, dict[str, Any]]:
        """Reset environment to specific state.

        Args:
            state: The ObjectCentricState to reset to
            seed: Optional random seed

        Returns:
            Tuple of (observation, info)
        """
        # Call parent reset to initialize properly
        if seed is not None:
            super().reset(seed=seed)

        # Set the state directly
        self.unwrapped_env._state = state  # pylint: disable=protected-access

        return self._get_obs(), self._get_info()

    def _get_obs(self) -> ObjectCentricState:
        """Get current observation."""
        return self.unwrapped_env._state  # pylint: disable=protected-access

    def _get_info(self) -> dict[str, Any]:
        """Get info dict."""
        return {}

    def clone(self) -> ClutteredStorage2DEnvWrapper:
        """Create a deep copy of the environment for planning simulations."""
        import copy
        cloned_base_env = copy.deepcopy(self.unwrapped_env)
        return ClutteredStorage2DEnvWrapper(cloned_base_env)
