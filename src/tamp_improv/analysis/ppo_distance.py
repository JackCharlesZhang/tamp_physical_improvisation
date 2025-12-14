"""Distance-based PPO performance analysis for gridworld environment.

This module implements an experiment to analyze PPO's success rate as a function
of the Manhattan distance between initial state and goal cell in the gridworld.

The experiment:
1. For each distance, samples multiple (initial_cell, goal_cell) pairs
2. Trains a SEPARATE PPO policy for EACH individual cell pair
3. Evaluates each policy on the cell pair it was trained on
4. Saves results showing the relationship between distance and performance

Note: This trains one policy per cell pair (not one policy per distance),
allowing analysis of how distance affects learning on specific configurations.

Usage:
    python -m src.tamp_improv.analysis.ppo_distance --num_cells 4 --seed 42
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import GraphInstance
from numpy.typing import NDArray

from tamp_improv.approaches.improvisational.policies.rl import (
    RLConfig,
    RLPolicy,
    TrainingProgressCallback,
)
from tamp_improv.benchmarks.gridworld_fixed import GridworldFixedEnv


# ============================================================================
# Utility Functions
# ============================================================================


def calculate_manhattan_distance(
    cell1: tuple[int, int], cell2: tuple[int, int]
) -> int:
    """Calculate Manhattan distance between two cells.

    Args:
        cell1: First cell coordinates (row, col)
        cell2: Second cell coordinates (row, col)

    Returns:
        Manhattan distance between cells
    """
    return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])


def generate_all_cell_pairs_at_distance(
    num_cells: int, distance: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Generate all valid cell pairs at a specific Manhattan distance.

    Args:
        num_cells: Grid dimension (C×C grid)
        distance: Target Manhattan distance

    Returns:
        List of (initial_cell, goal_cell) tuples at the specified distance
    """
    pairs = []
    for start_row in range(num_cells):
        for start_col in range(num_cells):
            for goal_row in range(num_cells):
                for goal_col in range(num_cells):
                    start_cell = (start_row, start_col)
                    goal_cell = (goal_row, goal_col)
                    if calculate_manhattan_distance(start_cell, goal_cell) == distance:
                        pairs.append((start_cell, goal_cell))
    return pairs


def sample_cell_pairs_for_training(
    all_pairs: list[tuple[tuple[int, int], tuple[int, int]]], num_samples: int, seed: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Sample a subset of cell pairs for training.

    Args:
        all_pairs: All valid cell pairs at a distance
        num_samples: Number of pairs to sample
        seed: Random seed for reproducibility

    Returns:
        Sampled list of cell pairs
    """
    rng = np.random.default_rng(seed)
    if len(all_pairs) <= num_samples:
        return all_pairs
    indices = rng.choice(len(all_pairs), size=num_samples, replace=False)
    return [all_pairs[i] for i in indices]


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class DistanceExperimentConfig:
    """Configuration for distance-based PPO experiment.

    Gridworld parameters:
        num_cells: Number of cells in each dimension (C×C grid)
        num_states_per_cell: Number of low-level states per cell
        num_teleporters: Number of teleporter pairs
        max_episode_steps: Maximum steps per episode

    Training parameters:
        total_timesteps: Total timesteps for PPO training per distance
        learning_rate: PPO learning rate
        batch_size: PPO batch size
        n_epochs: PPO epochs per update
        gamma: Discount factor
        ent_coef: Entropy coefficient

    Experiment parameters:
        distance_range: List of distances to test (e.g., [1, 2, 4, 6])
        num_training_pairs_per_distance: Number of cell pairs to sample per distance
            (each pair gets its own trained PPO policy)
        num_eval_episodes: Number of evaluation episodes per cell pair
        seed: Random seed for reproducibility
        output_dir: Directory to save results
        checkpoint_dir: Directory to save model checkpoints
    """

    # Gridworld parameters
    num_cells: int = 4
    num_states_per_cell: int = 5
    num_teleporters: int = 0
    max_episode_steps: int = 200

    # Training parameters
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    ent_coef: float = 0.03

    # Experiment parameters
    distance_range: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    num_training_pairs_per_distance: int = 40
    num_eval_episodes: int = 10
    seed: int = 42
    output_dir: str = "src/tamp_improv/results/ppo_distance"
    checkpoint_dir: str = "src/tamp_improv/checkpoints/ppo_distance"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "num_cells": self.num_cells,
            "num_states_per_cell": self.num_states_per_cell,
            "num_teleporters": self.num_teleporters,
            "max_episode_steps": self.max_episode_steps,
            "total_timesteps": self.total_timesteps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "ent_coef": self.ent_coef,
            "distance_range": self.distance_range,
            "num_training_pairs_per_distance": self.num_training_pairs_per_distance,
            "num_eval_episodes": self.num_eval_episodes,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "checkpoint_dir": self.checkpoint_dir,
        }


# ============================================================================
# Custom Environment Wrapper
# ============================================================================


class FixedDistanceGridworldWrapper(gym.Wrapper):
    """Wrapper for gridworld with fixed initial position and goal cell.

    This wrapper allows setting specific initial positions and goal cells
    for training PPO on specific distance configurations.
    """

    def __init__(
        self,
        env: GridworldFixedEnv,
        initial_cell: tuple[int, int],
        goal_cell: tuple[int, int],
        num_states_per_cell: int,
    ):
        """Initialize wrapper.

        Args:
            env: Base gridworld environment
            initial_cell: Initial cell (row, col)
            goal_cell: Goal cell (row, col)
            num_states_per_cell: Number of states per cell (S)
        """
        super().__init__(env)
        self.initial_cell = initial_cell
        self.goal_cell = goal_cell
        self.num_states_per_cell = num_states_per_cell

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[GraphInstance, dict[str, Any]]:
        """Reset environment with fixed initial cell and goal cell.

        Initial position is sampled randomly within the initial cell.
        """
        # Call parent reset to initialize RNG
        super().reset(seed=seed)

        # Set robot to random position within initial cell
        cell_row, cell_col = self.initial_cell
        offset_row = self.np_random.integers(0, self.num_states_per_cell)
        offset_col = self.np_random.integers(0, self.num_states_per_cell)

        robot_x = cell_col * self.num_states_per_cell + offset_col
        robot_y = cell_row * self.num_states_per_cell + offset_row

        self.env.robot_pos = np.array([robot_x, robot_y], dtype=np.int32)
        self.env.goal_cell = self.goal_cell
        self.env.step_count = 0

        obs = self.env._get_obs()
        info = self.env._get_info()

        return obs, info


# ============================================================================
# Training and Evaluation Functions
# ============================================================================


def train_ppo_on_cell_pair(
    cell_pair: tuple[tuple[int, int], tuple[int, int]],
    distance: int,
    pair_index: int,
    config: DistanceExperimentConfig,
) -> tuple[RLPolicy, dict[str, Any]]:
    """Train a PPO policy on a specific cell pair.

    Args:
        cell_pair: (initial_cell, goal_cell) to train on
        distance: Manhattan distance (for logging/checkpointing)
        pair_index: Index of this pair (for logging/checkpointing)
        config: Experiment configuration

    Returns:
        Tuple of (trained_policy, training_metrics)
    """
    initial_cell, goal_cell = cell_pair

    print(f"\n{'-'*60}")
    print(f"Training PPO on pair {pair_index}")
    print(f"  Initial: {initial_cell} -> Goal: {goal_cell} (distance={distance})")
    print(f"{'-'*60}")

    # Create base environment
    base_env = GridworldFixedEnv(
        num_cells=config.num_cells,
        num_states_per_cell=config.num_states_per_cell,
        num_teleporters=config.num_teleporters,
        max_episode_steps=config.max_episode_steps,
        seed=config.seed,
    )

    # Create wrapper for this specific cell pair
    wrapped_env = FixedDistanceGridworldWrapper(
        base_env, initial_cell, goal_cell, config.num_states_per_cell
    )

    # Apply FlattenGraphObsWrapper to convert Graph observations to flat arrays
    from tamp_improv.approaches.improvisational.policies.rl import FlattenGraphObsWrapper
    flatten_env = FlattenGraphObsWrapper(wrapped_env)

    # Create RLConfig
    rl_config = RLConfig(
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        ent_coef=config.ent_coef,
        device="cuda",
        deterministic=False,
    )

    # Initialize policy
    policy = RLPolicy(seed=config.seed, config=rl_config)

    # Create callback for tracking
    checkpoint_path = Path(config.checkpoint_dir) / f"distance_{distance}" / f"pair_{pair_index}"
    callback = TrainingProgressCallback(
        check_freq=100,
        verbose=0,  # Less verbose for individual pairs
        early_stopping=False,
        policy_key=f"d{distance}_p{pair_index}",
        save_checkpoints=True,
        checkpoint_dir=str(checkpoint_path),
    )

    # Create PPO model with flattened environment
    from stable_baselines3 import PPO
    model = PPO(
        "MlpPolicy",
        flatten_env,
        learning_rate=rl_config.learning_rate,
        n_steps=config.max_episode_steps,
        batch_size=rl_config.batch_size,
        n_epochs=rl_config.n_epochs,
        gamma=rl_config.gamma,
        ent_coef=rl_config.ent_coef,
        device=rl_config.device,
        seed=config.seed,
        verbose=0,  # Less verbose for individual pairs
    )

    # Train the model
    model.learn(total_timesteps=config.total_timesteps, callback=callback)
    policy.model = model

    # Extract training metrics
    training_metrics = {
        "distance": distance,
        "total_timesteps": config.total_timesteps,
        "final_success_rate": callback._get_success_rate,
        "final_avg_episode_length": callback._get_avg_episode_length,
        "final_avg_reward": callback._get_avg_reward,
    }

    print(f"  Training complete - Success rate: {training_metrics['final_success_rate']:.2%}")

    return policy, training_metrics


def evaluate_policy(
    policy: RLPolicy,
    cell_pair: tuple[tuple[int, int], tuple[int, int]],
    config: DistanceExperimentConfig,
    num_episodes: int,
) -> dict[str, Any]:
    """Evaluate a trained policy on a specific cell pair.

    Args:
        policy: Trained PPO policy
        cell_pair: (initial_cell, goal_cell) to evaluate on
        config: Experiment configuration
        num_episodes: Number of episodes to evaluate

    Returns:
        Dictionary of evaluation metrics
    """
    initial_cell, goal_cell = cell_pair

    # Create environment for this specific cell pair
    base_env = GridworldFixedEnv(
        num_cells=config.num_cells,
        num_states_per_cell=config.num_states_per_cell,
        num_teleporters=config.num_teleporters,
        max_episode_steps=config.max_episode_steps,
        seed=config.seed + 1000,  # Different seed for evaluation
    )

    wrapped_env = FixedDistanceGridworldWrapper(
        base_env, initial_cell, goal_cell, config.num_states_per_cell
    )

    successes = []
    episode_lengths = []
    episode_rewards = []

    for ep in range(num_episodes):
        obs, info = wrapped_env.reset(seed=config.seed + 1000 + ep)
        done = False
        truncated = False
        step_count = 0
        total_reward = 0.0

        while not (done or truncated):
            action = policy.get_action(obs)
            obs, reward, done, truncated, info = wrapped_env.step(action)
            total_reward += reward
            step_count += 1

        success = done and not truncated
        successes.append(success)
        episode_lengths.append(step_count)
        episode_rewards.append(total_reward)

    return {
        "initial_cell": initial_cell,
        "goal_cell": goal_cell,
        "num_episodes": num_episodes,
        "success_rate": np.mean(successes),
        "avg_episode_length": np.mean(episode_lengths),
        "std_episode_length": np.std(episode_lengths),
        "avg_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
    }


def evaluate_random_policy(
    cell_pair: tuple[tuple[int, int], tuple[int, int]],
    config: DistanceExperimentConfig,
    num_episodes: int,
) -> dict[str, Any]:
    """Evaluate a random action policy on a specific cell pair.

    Args:
        cell_pair: (initial_cell, goal_cell) to evaluate on
        config: Experiment configuration
        num_episodes: Number of episodes to evaluate

    Returns:
        Dictionary of evaluation metrics with 'random_' prefix
    """
    initial_cell, goal_cell = cell_pair

    # Create environment for this specific cell pair
    base_env = GridworldFixedEnv(
        num_cells=config.num_cells,
        num_states_per_cell=config.num_states_per_cell,
        num_teleporters=config.num_teleporters,
        max_episode_steps=config.max_episode_steps,
        seed=config.seed + 2000,  # Different seed for random evaluation
    )

    wrapped_env = FixedDistanceGridworldWrapper(
        base_env, initial_cell, goal_cell, config.num_states_per_cell
    )

    successes = []
    episode_lengths = []
    episode_rewards = []

    # Use numpy RNG for consistent random action sampling
    rng = np.random.default_rng(config.seed + 3000)

    for ep in range(num_episodes):
        obs, info = wrapped_env.reset(seed=config.seed + 2000 + ep)
        done = False
        truncated = False
        step_count = 0
        total_reward = 0.0

        while not (done or truncated):
            # Sample random action (0=up, 1=down, 2=left, 3=right, 4=teleport)
            action = rng.integers(0, wrapped_env.action_space.n)
            obs, reward, done, truncated, info = wrapped_env.step(action)
            total_reward += reward
            step_count += 1

        success = done and not truncated
        successes.append(success)
        episode_lengths.append(step_count)
        episode_rewards.append(total_reward)

    # Return with 'random_' prefix to distinguish from PPO results
    return {
        "random_success_rate": np.mean(successes),
        "random_avg_episode_length": np.mean(episode_lengths),
        "random_std_episode_length": np.std(episode_lengths),
        "random_avg_reward": np.mean(episode_rewards),
        "random_std_reward": np.std(episode_rewards),
    }


# ============================================================================
# Main Experiment Loop
# ============================================================================


def run_distance_experiment(config: DistanceExperimentConfig) -> pd.DataFrame:
    """Run the complete distance-based PPO experiment.

    Args:
        config: Experiment configuration

    Returns:
        DataFrame with results for each distance and cell pair
    """
    print(f"\n{'='*80}")
    print("Starting Distance-Based PPO Experiment")
    print("Training ONE policy per cell pair")
    print(f"{'='*80}\n")
    print(f"Configuration:")
    print(f"  Grid: {config.num_cells}×{config.num_cells} cells")
    print(f"  States per cell: {config.num_states_per_cell}")
    print(f"  Distance range: {config.distance_range}")
    print(f"  Training pairs per distance: {config.num_training_pairs_per_distance}")
    print(f"  Evaluation episodes per pair: {config.num_eval_episodes}")
    print(f"  Total timesteps per policy: {config.total_timesteps}")
    print(f"  Seed: {config.seed}\n")

    total_policies = sum(
        len(sample_cell_pairs_for_training(
            generate_all_cell_pairs_at_distance(config.num_cells, d),
            config.num_training_pairs_per_distance,
            config.seed
        ))
        for d in config.distance_range
    )
    print(f"  Will train {total_policies} total PPO policies\n")

    # Create output directories
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_file = output_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Saved configuration to {config_file}\n")

    # Store all results
    all_results = []

    # Iterate over distances
    for distance in config.distance_range:
        print(f"\n{'='*80}")
        print(f"Processing distance: {distance}")
        print(f"{'='*80}\n")

        # Generate all cell pairs at this distance
        all_pairs = generate_all_cell_pairs_at_distance(config.num_cells, distance)
        print(f"Found {len(all_pairs)} total cell pairs at distance {distance}")

        if len(all_pairs) == 0:
            print(f"No valid cell pairs at distance {distance}, skipping...")
            continue

        # Sample training pairs
        training_pairs = sample_cell_pairs_for_training(
            all_pairs, config.num_training_pairs_per_distance, config.seed
        )
        print(f"Selected {len(training_pairs)} pairs for training\n")

        # Train a separate PPO policy for EACH cell pair
        for i, cell_pair in enumerate(training_pairs):
            print(f"[Distance {distance}] Training on pair {i+1}/{len(training_pairs)}")

            # Train policy on this specific cell pair
            policy, training_metrics = train_ppo_on_cell_pair(
                cell_pair, distance, i, config
            )

            # Evaluate the PPO policy on the SAME cell pair it was trained on
            eval_metrics = evaluate_policy(
                policy, cell_pair, config, config.num_eval_episodes
            )

            # Evaluate random policy on the SAME cell pair for comparison
            random_metrics = evaluate_random_policy(
                cell_pair, config, config.num_eval_episodes
            )

            # Combine results (PPO + random baseline)
            result = {
                "distance": distance,
                "pair_index": i,
                "initial_cell_row": cell_pair[0][0],
                "initial_cell_col": cell_pair[0][1],
                "goal_cell_row": cell_pair[1][0],
                "goal_cell_col": cell_pair[1][1],
                **eval_metrics,
                **random_metrics,
                **training_metrics,
            }
            all_results.append(result)

            # Print comparison
            print(f"  PPO success: {eval_metrics['success_rate']:.2%} | "
                  f"Random success: {random_metrics['random_success_rate']:.2%}")

        print(f"\nCompleted distance {distance} - Trained {len(training_pairs)} policies")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n{'='*80}")
    print(f"Saved detailed results to {results_file}")

    # Generate summary statistics (PPO + Random)
    summary = results_df.groupby("distance").agg({
        "success_rate": ["mean", "std", "min", "max"],
        "avg_episode_length": ["mean", "std"],
        "avg_reward": ["mean", "std"],
        "random_success_rate": ["mean", "std", "min", "max"],
        "random_avg_episode_length": ["mean", "std"],
        "random_avg_reward": ["mean", "std"],
    }).round(4)

    summary_file = output_path / f"summary_{timestamp}.csv"
    summary.to_csv(summary_file)
    print(f"Saved summary to {summary_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")
    print(summary)
    print(f"\n{'='*80}")

    return results_df


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point for the experiment."""
    parser = argparse.ArgumentParser(
        description="PPO Distance Experiment for Gridworld"
    )

    # Gridworld parameters
    parser.add_argument(
        "--num_cells", type=int, default=4, help="Number of cells per dimension"
    )
    parser.add_argument(
        "--num_states_per_cell", type=int, default=5, help="States per cell"
    )
    parser.add_argument(
        "--num_teleporters", type=int, default=0, help="Number of teleporter pairs"
    )
    parser.add_argument(
        "--max_episode_steps", type=int, default=200, help="Max steps per episode"
    )

    # Training parameters
    parser.add_argument(
        "--total_timesteps", type=int, default=200_000, help="Total training timesteps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="PPO learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="PPO batch size"
    )

    # Experiment parameters
    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        help="Distances to test (e.g., --distances 1 2 4 6)",
    )
    parser.add_argument(
        "--num_training_pairs",
        type=int,
        default=40,
        help="Number of cell pairs to sample per distance (each gets its own PPO policy)",
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes per pair",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/ppo_distance", help="Output directory"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/ppo_distance",
        help="Checkpoint directory",
    )

    args = parser.parse_args()

    # Create configuration
    config = DistanceExperimentConfig(
        num_cells=args.num_cells,
        num_states_per_cell=args.num_states_per_cell,
        num_teleporters=args.num_teleporters,
        max_episode_steps=args.max_episode_steps,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        distance_range=args.distances,
        num_training_pairs_per_distance=args.num_training_pairs,
        num_eval_episodes=args.num_eval_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Run experiment
    results_df = run_distance_experiment(config)

    print("\nExperiment completed successfully!")
    print(f"Results saved to {config.output_dir}")

    return results_df


if __name__ == "__main__":
    main()
