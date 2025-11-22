"""Goal-conditioned distance heuristic for pruning shortcuts.

This module implements a learned distance function f(s, z') that estimates
the number of steps required to reach abstract goal state z' from state s.
The heuristic is trained using goal-conditioned RL with a reward of -1 per step,
which causes the value function to approximate V(s, z') = -distance(s, z').

Note: z' is an abstract state (set of atoms/predicates), not a low-level state.
This matches what policies actually see during execution.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC, DQN
from stable_baselines3.common.callbacks import BaseCallback

from tamp_improv.benchmarks.goal_wrapper import GoalConditionedWrapper

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class DistanceHeuristicConfig:
    """Configuration for distance heuristic training."""

    # Algorithm: "SAC" for continuous, "DQN" for discrete actions
    algorithm: str = "DQN"

    # Learning parameters
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 100000

    # Training parameters
    max_episode_steps: int = 100
    learning_starts: int = 1000

    # Device settings
    device: str = "cuda"


class DistanceHeuristicCallback(BaseCallback):
    """Callback to track distance heuristic training progress."""

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_lengths: list[int] = []
        self.episode_rewards: list[float] = []
        self.current_length = 0
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        """Called at each step of training."""
        self.current_length += 1
        self.current_reward += self.locals["rewards"][0]
        dones = self.locals["dones"]

        if dones[0]:
            self.episode_lengths.append(self.current_length)
            self.episode_rewards.append(self.current_reward)
            self.current_length = 0
            self.current_reward = 0.0

        if self.num_timesteps % self.check_freq == 0:
            print("\nDistance Heuristic Training Progress:")
            print(f"Timesteps: {self.num_timesteps}")

            if self.episode_lengths:
                n_recent = min(100, len(self.episode_lengths))
                recent_lengths = self.episode_lengths[-n_recent:]
                recent_rewards = self.episode_rewards[-n_recent:]

                print(f"Episodes completed: {len(self.episode_lengths)}")
                print(
                    f"Recent Avg Episode Length: {np.mean(recent_lengths):.2f}"
                )
                print(
                    f"Recent Avg Reward: {np.mean(recent_rewards):.2f}"
                )

            buffer = self.locals.get("replay_buffer")
            if buffer is not None:
                print(f"Replay buffer size: {buffer.size()}/{buffer.buffer_size}")

        return True


class DistanceHeuristicWrapper(gym.Wrapper):
    """Wrapper that provides -1 reward per step for distance learning.

    This wrapper modifies the reward structure to be -1 per step, regardless
    of whether the goal is reached. This causes the learned value function
    V(s, z') to approximate -distance(s, z'), where s is the low-level state
    and z' is the abstract goal state (atoms).
    """

    def __init__(
        self,
        env: gym.Env,
        state_pairs: list[tuple[ObsType, ObsType]],
        perceiver: Any,
        max_episode_steps: int = 100,
        max_atom_size: int = 50,
    ):
        """Initialize wrapper with state pairs to sample from.

        Args:
            env: Base environment (should support reset_from_state)
            state_pairs: List of (source_state, goal_state) tuples
            perceiver: Perceiver to convert observations to atoms
            max_episode_steps: Maximum steps per episode
            max_atom_size: Maximum number of unique atoms (for fixed-size vector)
        """
        super().__init__(env)
        self.state_pairs = state_pairs
        self.perceiver = perceiver
        self.max_episode_steps = max_episode_steps
        self.max_atom_size = max_atom_size
        self.steps = 0
        self.current_goal_atoms: set | None = None

        # Dynamic atom indexing (like GoalConditionedWrapper)
        self.atom_to_index: dict[str, int] = {}
        self._next_index = 0

        # Determine observation space structure
        if len(state_pairs) > 0:
            sample_state = state_pairs[0][0]

            # Determine state observation space
            if hasattr(sample_state, "nodes"):
                # Graph observation
                flattened_size = sample_state.nodes.flatten().shape[0]
                obs_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(flattened_size,), dtype=np.float32
                )
            else:
                # Array observation
                flattened = np.array(sample_state).flatten()
                obs_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=flattened.shape, dtype=np.float32
                )

            # Goal space is fixed-size atom vector (multi-hot encoding)
            goal_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(max_atom_size,), dtype=np.float32
            )

            # Goal-conditioned observation space
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": obs_space,
                    "achieved_goal": goal_space,
                    "desired_goal": goal_space,
                }
            )

    def flatten_obs(self, obs: ObsType) -> np.ndarray:
        """Flatten observation to array."""
        if hasattr(obs, "nodes"):
            return obs.nodes.flatten().astype(np.float32)
        return np.array(obs).flatten().astype(np.float32)

    def _get_atom_index(self, atom_str: str) -> int:
        """Get a unique index for this atom (dynamic indexing)."""
        if atom_str in self.atom_to_index:
            return self.atom_to_index[atom_str]
        assert (
            self._next_index < self.max_atom_size
        ), f"No more space for new atom at index {self._next_index}. Increase max_atom_size (currently {self.max_atom_size})."
        idx = self._next_index
        self.atom_to_index[atom_str] = idx
        self._next_index += 1
        return idx

    def create_atom_vector(self, atoms: set) -> np.ndarray:
        """Create a multi-hot vector representation of the set of atoms."""
        vector = np.zeros(self.max_atom_size, dtype=np.float32)
        for atom in atoms:
            idx = self._get_atom_index(str(atom))
            vector[idx] = 1.0
        return vector

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment by sampling a random state pair."""
        self.steps = 0

        # Sample a random state pair
        pair_idx = np.random.randint(len(self.state_pairs))
        source_state, goal_state = self.state_pairs[pair_idx]

        # Convert goal state to atoms (abstract representation)
        self.current_goal_atoms = self.perceiver.step(goal_state)

        # Reset from source state
        # Need to access unwrapped env to get reset_from_state method
        unwrapped_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        if hasattr(unwrapped_env, "reset_from_state"):
            # Try with seed first, fall back to without seed if not supported
            try:
                obs, info = unwrapped_env.reset_from_state(source_state, seed=seed)
            except TypeError:
                # Some environments don't accept seed parameter
                obs, info = unwrapped_env.reset_from_state(source_state)
        else:
            raise AttributeError(
                f"Environment must have reset_from_state method. "
                f"Environment type: {type(unwrapped_env)}"
            )

        # Create goal-conditioned observation
        flat_obs = self.flatten_obs(obs)
        goal_vector = self.create_atom_vector(self.current_goal_atoms)

        # Current achieved atoms
        current_atoms = self.perceiver.step(obs)
        achieved_goal_vector = self.create_atom_vector(current_atoms)

        dict_obs = {
            "observation": flat_obs,
            "achieved_goal": achieved_goal_vector,
            "desired_goal": goal_vector,
        }

        return dict_obs, info

    def step(
        self, action: ActType
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Step environment with -1 reward per step."""
        next_obs, _, terminated, truncated, info = self.env.step(action)
        self.steps += 1

        flat_obs = self.flatten_obs(next_obs)
        goal_vector = self.create_atom_vector(self.current_goal_atoms)

        # Get current achieved atoms
        current_atoms = self.perceiver.step(next_obs)
        achieved_goal_vector = self.create_atom_vector(current_atoms)

        dict_obs = {
            "observation": flat_obs,
            "achieved_goal": achieved_goal_vector,
            "desired_goal": goal_vector,
        }

        # Key: Always return -1 reward (this makes V(s, z') = -distance)
        reward = -1.0

        # Check if goal is reached (atoms match)
        goal_reached = current_atoms == self.current_goal_atoms
        info["is_success"] = bool(goal_reached)

        # Terminate if goal reached or max steps
        terminated = terminated or goal_reached
        truncated = truncated or (self.steps >= self.max_episode_steps)

        return dict_obs, reward, terminated, truncated, info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: list[dict[str, Any]] | dict[str, Any],
        _indices: list[int] | None = None,
    ) -> np.ndarray:
        """Compute reward for HER (always -1)."""
        if achieved_goal.ndim == 1:
            # Single goal
            return np.array([-1.0], dtype=np.float32)
        else:
            # Batch of goals
            return np.full(achieved_goal.shape[0], -1.0, dtype=np.float32)


class GoalConditionedDistanceHeuristic:
    """Learns distance function f(s, s') via goal-conditioned RL.

    Trains a policy with reward=-1 per step, causing the value function to
    learn V(s, g) ≈ -distance(s, g). The heuristic can then estimate distances
    by querying the critic/value network.
    """

    def __init__(self, config: DistanceHeuristicConfig | None = None, seed: int = 42):
        """Initialize distance heuristic.

        Args:
            config: Configuration for training
            seed: Random seed
        """
        self.config = config or DistanceHeuristicConfig()
        self.seed = seed

        if self.config.device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        self.model: SAC | DQN | None = None
        self.algorithm: str | None = None
        self.perceiver: Any | None = None
        self.wrapper: DistanceHeuristicWrapper | None = None
        self._obs_mean: np.ndarray | None = None
        self._obs_std: np.ndarray | None = None

    def train(
        self,
        env: gym.Env,
        state_pairs: list[tuple[ObsType, ObsType]],
        perceiver: Any,
        max_training_steps: int,
    ) -> None:
        """Train distance heuristic on sampled state pairs.

        Args:
            env: Base environment
            state_pairs: List of (source, target) state pairs
            perceiver: Perceiver to convert states to atoms
            max_training_steps: Total training timesteps
        """
        print(f"\nTraining distance heuristic on {len(state_pairs)} state pairs...")
        print(f"Device: {self.device}")

        # Store perceiver for use in estimate_distance
        self.perceiver = perceiver

        # Wrap environment for distance learning (converts goal states to atoms internally)
        wrapped_env = DistanceHeuristicWrapper(
            env,
            state_pairs,
            perceiver,
            max_episode_steps=self.config.max_episode_steps,
        )

        # Store wrapper so we can use its atom_to_index mapping
        self.wrapper = wrapped_env

        # Compute observation statistics for normalization
        self._compute_obs_statistics(state_pairs)

        # Detect action space type and choose appropriate algorithm
        action_space = wrapped_env.action_space
        if isinstance(action_space, gym.spaces.Box):
            # Continuous action space - use SAC
            algorithm = "SAC"
            print(f"Detected continuous action space (Box), using SAC")
        elif isinstance(action_space, gym.spaces.Discrete):
            # Discrete action space - use DQN
            algorithm = "DQN"
            print(f"Detected discrete action space (Discrete), using DQN")
        else:
            raise ValueError(
                f"Unsupported action space type: {type(action_space)}. "
                f"Only Box and Discrete are supported."
            )

        # Create model based on action space
        if algorithm == "SAC":
            self.model = SAC(
                "MultiInputPolicy",
                wrapped_env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                buffer_size=self.config.buffer_size,
                learning_starts=self.config.learning_starts,
                device=self.device,
                seed=self.seed,
                verbose=1,
            )
        elif algorithm == "DQN":
            self.model = DQN(
                "MultiInputPolicy",
                wrapped_env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                buffer_size=self.config.buffer_size,
                learning_starts=self.config.learning_starts,
                device=self.device,
                seed=self.seed,
                verbose=1,
            )
        
        # Store algorithm type for later use
        self.algorithm = algorithm

        # Train with callback
        callback = DistanceHeuristicCallback(check_freq=1000)
        self.model.learn(total_timesteps=max_training_steps, callback=callback)

        print("\nDistance heuristic training complete!")

    def estimate_distance(self, source_state: ObsType, target_state: ObsType) -> float:
        """Estimate distance from source to target state.

        Args:
            source_state: Starting state
            target_state: Goal state (will be converted to atoms)

        Returns:
            Estimated distance (number of steps)
        """
        assert self.model is not None, "Model must be trained before estimation"
        assert self.perceiver is not None, "Perceiver must be set before estimation"
        assert self.wrapper is not None, "Wrapper must be set before estimation"

        # Flatten source observation
        source_flat = self._flatten_and_normalize(source_state)

        # Convert target state to atoms, then to goal vector
        target_atoms = self.perceiver.step(target_state)
        target_goal_vector = self.wrapper.create_atom_vector(target_atoms)

        # Convert source state to atoms for achieved goal
        source_atoms = self.perceiver.step(source_state)
        achieved_goal_vector = self.wrapper.create_atom_vector(source_atoms)

        # Create goal-conditioned observation
        obs = {
            "observation": source_flat,
            "achieved_goal": achieved_goal_vector,
            "desired_goal": target_goal_vector,
        }

        # Get value estimate from critic (V(s, g) ≈ -distance)
        with torch.no_grad():
            obs_tensor = {
                k: torch.as_tensor(v).unsqueeze(0).to(self.device)
                for k, v in obs.items()
            }

            if self.algorithm == "SAC":
                # SAC has two Q-functions, we'll use the minimum
                # Get action from policy
                action, _ = self.model.predict(obs, deterministic=True)
                action_tensor = torch.as_tensor(action).unsqueeze(0).to(self.device)

                # Get Q-values from both critics
                q1_value = self.model.critic(obs_tensor, action_tensor)[0]
                q2_value = self.model.critic(obs_tensor, action_tensor)[1]

                # Use minimum Q-value (standard in SAC)
                q_value = torch.min(q1_value, q2_value).item()
            
            elif self.algorithm == "DQN":
                # DQN: Get Q-values for all actions, take max
                q_values = self.model.q_net(obs_tensor)  # Shape: (batch, n_actions)
                
                # Take the max Q-value (best action)
                q_value = q_values.max(dim=1)[0].item()

        # Since reward = -1 per step, Q(s,a,g) ≈ -distance
        # So distance ≈ -Q(s,a,g)
        estimated_distance = -q_value

        # Clamp to reasonable range
        estimated_distance = max(0.0, estimated_distance)

        return float(estimated_distance)

    def batch_estimate_distances(
        self,
        state_pairs: list[tuple[ObsType, ObsType]],
    ) -> np.ndarray:
        """Estimate distances for multiple state pairs efficiently.

        Args:
            state_pairs: List of (source, target) state pairs

        Returns:
            Array of estimated distances
        """
        distances = np.zeros(len(state_pairs))

        for i, (source, target) in enumerate(state_pairs):
            distances[i] = self.estimate_distance(source, target)

        return distances

    def save(self, path: str) -> None:
        """Save the distance heuristic.

        Args:
            path: Directory path to save model

        Note:
            Perceiver is not saved and must be provided when loading.
        """
        assert self.model is not None, "Model must be trained before saving"
        assert self.wrapper is not None, "Wrapper must be available before saving"

        os.makedirs(path, exist_ok=True)

        # Save model (SAC or QR-DQN)
        self.model.save(f"{path}/distance_model")
        
        # Save algorithm type
        with open(f"{path}/algorithm.pkl", "wb") as f:
            pickle.dump({"algorithm": self.algorithm}, f)

        # Save observation statistics
        with open(f"{path}/obs_stats.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_mean": self._obs_mean,
                    "obs_std": self._obs_std,
                },
                f,
            )

        # Save atom indexing from wrapper
        with open(f"{path}/atom_to_index.pkl", "wb") as f:
            pickle.dump(
                {
                    "atom_to_index": self.wrapper.atom_to_index,
                    "_next_index": self.wrapper._next_index,
                    "max_atom_size": self.wrapper.max_atom_size,
                },
                f,
            )

        # Save observation/action spaces
        with open(f"{path}/observation_space.pkl", "wb") as f:
            pickle.dump(self.model.observation_space, f)
        with open(f"{path}/action_space.pkl", "wb") as f:
            pickle.dump(self.model.action_space, f)

        print(f"Distance heuristic saved to {path}")

    def load(self, path: str, perceiver: Any) -> None:
        """Load a pre-trained distance heuristic.

        Args:
            path: Directory path containing saved model
            perceiver: Perceiver to convert states to atoms
        """
        # Store perceiver
        self.perceiver = perceiver
        
        # Load algorithm type
        with open(f"{path}/algorithm.pkl", "rb") as f:
            algo_data = pickle.load(f)
            self.algorithm = algo_data["algorithm"]
        
        print(f"Loading {self.algorithm} model...")

        # Load observation statistics
        with open(f"{path}/obs_stats.pkl", "rb") as f:
            stats = pickle.load(f)
            self._obs_mean = stats["obs_mean"]
            self._obs_std = stats["obs_std"]

        # Load atom indexing
        with open(f"{path}/atom_to_index.pkl", "rb") as f:
            atom_data = pickle.load(f)

        # Create a dummy wrapper to hold the atom indexing
        # We need a minimal environment for this
        class MinimalEnv(gym.Env):
            def __init__(self, obs_space, act_space):
                self.observation_space = obs_space
                self.action_space = act_space

        with open(f"{path}/observation_space.pkl", "rb") as f:
            observation_space = pickle.load(f)
        with open(f"{path}/action_space.pkl", "rb") as f:
            action_space = pickle.load(f)

        minimal_env = MinimalEnv(observation_space["observation"], action_space)

        # Create wrapper with empty state pairs (we just need it for atom indexing)
        self.wrapper = DistanceHeuristicWrapper(
            minimal_env,
            [],  # Empty state pairs
            perceiver,
            max_atom_size=atom_data["max_atom_size"],
        )

        # Restore atom indexing
        self.wrapper.atom_to_index = atom_data["atom_to_index"]
        self.wrapper._next_index = atom_data["_next_index"]

        # Load spaces again for model loading
        with open(f"{path}/observation_space.pkl", "rb") as f:
            observation_space = pickle.load(f)
        with open(f"{path}/action_space.pkl", "rb") as f:
            action_space = pickle.load(f)

        # Create dummy env for loading
        class DummyEnv(gym.Env):  # pylint: disable=abstract-method
            """Dummy environment for loading model."""

            def __init__(self, obs_space, act_space):
                self.observation_space = obs_space
                self.action_space = act_space

            def compute_reward(self, achieved_goal, _desired_goal, _info, _indices=None):
                if isinstance(achieved_goal, np.ndarray):
                    return np.full(achieved_goal.shape[0], -1.0, dtype=np.float32)
                return -1.0

            def reset(self, **_kwargs):
                obs = {}
                for key, space in self.observation_space.spaces.items():
                    obs[key] = np.zeros(space.shape, dtype=space.dtype)
                return obs, {}

            def step(self, action):
                obs, _ = self.reset()
                return obs, -1.0, False, False, {}

        dummy_env = DummyEnv(observation_space, action_space)  # type: ignore

        # Load model based on algorithm type
        if self.algorithm == "SAC":
            self.model = SAC.load(f"{path}/distance_model", env=dummy_env, device=self.device)
        elif self.algorithm == "DQN":
            self.model = DQN.load(f"{path}/distance_model", env=dummy_env, device=self.device)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        print(f"Distance heuristic ({self.algorithm}) loaded from {path}")

    def _flatten_and_normalize(self, obs: ObsType) -> np.ndarray:
        """Flatten and normalize observation."""
        if hasattr(obs, "nodes"):
            flat = obs.nodes.flatten().astype(np.float32)
        else:
            flat = np.array(obs).flatten().astype(np.float32)

        # Normalize if statistics are available
        if self._obs_mean is not None and self._obs_std is not None:
            flat = (flat - self._obs_mean) / (self._obs_std + 1e-8)

        return flat

    def _compute_obs_statistics(self, state_pairs: list[tuple[ObsType, ObsType]]) -> None:
        """Compute mean and std of observations for normalization."""
        all_obs = []

        for source, target in state_pairs:
            if hasattr(source, "nodes"):
                all_obs.append(source.nodes.flatten())
                all_obs.append(target.nodes.flatten())
            else:
                all_obs.append(np.array(source).flatten())
                all_obs.append(np.array(target).flatten())

        all_obs_array = np.array(all_obs, dtype=np.float32)
        self._obs_mean = np.mean(all_obs_array, axis=0)
        self._obs_std = np.std(all_obs_array, axis=0)

        print(f"Computed observation statistics: mean={self._obs_mean.mean():.3f}, std={self._obs_std.mean():.3f}")
