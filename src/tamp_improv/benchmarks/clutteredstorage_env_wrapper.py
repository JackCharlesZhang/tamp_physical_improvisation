"""Wrapper for ObjectCentricClutteredStorage2DEnv to add reset_from_state."""

from __future__ import annotations

from typing import Any
import gymnasium as gym
from prbench.envs.geom2d.clutteredstorage2d import ObjectCentricClutteredStorage2DEnv
from relational_structs import ObjectCentricState, ObjectCentricStateSpace


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

        types = env.type_features

        # Reset to get the actual observation structure
        temp_obs, _ = env.reset()
        # The observation objects are the ones actually in the observation state
        self.observation_objects = list(temp_obs)

        self.observation_state_space = ObjectCentricStateSpace(types)

        self.observation_space = self.observation_state_space.to_box(
            constant_objects=self.observation_objects,
            type_features=env.type_features
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        observation, info = super().reset(seed=seed, options=options)
        # Convert ObjectCentricState to vector using observation objects
        return observation.vec(objects=self.observation_objects), info

    def step(self, action):
        """Step the environment and return vectorized observation."""
        observation, reward, terminated, truncated, info = super().step(action)
        # Convert ObjectCentricState to vector using observation objects
        return observation.vec(objects=self.observation_objects), reward, terminated, truncated, info

    def reset_from_state(
        self,
        state: NDArray[np.float32],
        *,
        seed: int | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
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

    def _get_obs(self):
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
