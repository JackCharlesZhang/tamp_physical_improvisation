"""Wrapper for ObjectCentricClutteredStorage2DEnv to add reset_from_state."""

from __future__ import annotations

from typing import Any
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from prbench.envs.geom2d.clutteredstorage2d import (
    ObjectCentricClutteredStorage2DEnv,
    ShelfType,
    TargetBlockType,
)
from prbench.envs.geom2d.object_types import CRVRobotType
from prbench.envs.geom2d.structs import SE2Pose
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
        # IMPORTANT: Create a COPY of the list to avoid mutation
        self.observation_objects = [obj for obj in temp_obs]

        print(f"[INIT] observation_objects: {self.observation_objects}")
        print(f"[INIT] observation_objects id: {id(self.observation_objects)}")
        print(f"[INIT] observation_objects length: {len(self.observation_objects)}")

        self.observation_state_space = ObjectCentricStateSpace(types)

        self.observation_space = self.observation_state_space.to_box(
            constant_objects=self.observation_objects,
            type_features=env.type_features
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        print(f"\n[RESET] Called with seed={seed}")
        print(f"[RESET] observation_objects BEFORE: {self.observation_objects}")
        print(f"[RESET] observation_objects length BEFORE: {len(self.observation_objects)}")
        observation, info = super().reset(seed=seed, options=options)
        print(f"[RESET] observation_objects AFTER: {self.observation_objects}")
        print(f"[RESET] observation_objects length AFTER: {len(self.observation_objects)}")
        print(f"[RESET] observation type: {type(observation)}")
        print(f"[RESET] observation objects in returned state: {list(observation)}")
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
            state: Vectorized observation to reset to
            seed: Optional random seed

        Returns:
            Tuple of (observation, info)
        """
        print(f"\n[RESET_FROM_STATE] Called")
        print(f"[RESET_FROM_STATE] observation_objects BEFORE: {self.observation_objects}")
        print(f"[RESET_FROM_STATE] observation_objects id BEFORE: {id(self.observation_objects)}")
        print(f"[RESET_FROM_STATE] observation_objects length BEFORE: {len(self.observation_objects)}")
        print(f"[RESET_FROM_STATE] state vector shape: {state.shape}")

        # Call parent reset to initialize properly if seed is provided
        if seed is not None:
            print(f"[RESET_FROM_STATE] Calling super().reset(seed={seed})")
            super().reset(seed=seed)
            print(f"[RESET_FROM_STATE] observation_objects AFTER super().reset(): {self.observation_objects}")
            print(f"[RESET_FROM_STATE] observation_objects id AFTER super().reset(): {id(self.observation_objects)}")
            print(f"[RESET_FROM_STATE] observation_objects length AFTER super().reset(): {len(self.observation_objects)}")

        # Convert vectorized state to ObjectCentricState
        print(f"[RESET_FROM_STATE] Converting vector to ObjectCentricState")
        state_obj = ObjectCentricState.from_vec(
            state,
            constant_objects=self.observation_objects,
            type_features=self.unwrapped_env.type_features
        )

        # Set the state directly - this preserves all properties exactly
        # Using _create_initial_state would reset dimensions and colors to defaults
        self.unwrapped_env._current_state = state_obj  # pylint: disable=protected-access

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        """Get current observation."""
        state = self.unwrapped_env._current_state  # pylint: disable=protected-access
        return state.vec(objects=self.observation_objects)

    def _get_info(self) -> dict[str, Any]:
        """Get info dict."""
        return {}

    def clone(self) -> ClutteredStorage2DEnvWrapper:
        """Create a deep copy of the environment for planning simulations."""
        import copy
        cloned_base_env = copy.deepcopy(self.unwrapped_env)
        return ClutteredStorage2DEnvWrapper(cloned_base_env)
