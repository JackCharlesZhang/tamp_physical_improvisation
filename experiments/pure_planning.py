"""Pure planning baselines for PyBullet environments."""

import argparse
from pathlib import Path

import numpy as np
from task_then_motion_planning.planning import TaskThenMotionPlanner

from tamp_improv.benchmarks.pybullet_cleanup_table import (
    BaseCleanupTableTAMPSystem,
)
from tamp_improv.benchmarks.pybullet_cluttered_drawer import (
    BaseClutteredDrawerTAMPSystem,
)
from tamp_improv.benchmarks.pybullet_obstacle_tower_graph import (
    BaseGraphObstacleTowerTAMPSystem,
)


def run_obstacle_tower_planning(
    seed: int = 124,
    render_mode: str | None = None,
    max_steps: int = 500,
    num_episodes: int = 100,
    save_dir: str = "results/pure_planning",
) -> dict:
    """Run pure planning baseline on GraphObstacleTower environment."""
    print("\n" + "=" * 80)
    print("Running Pure Planning on GraphObstacleTower Environment")
    print(f"Evaluating over {num_episodes} episodes")
    print("=" * 80)

    successes = []
    lengths = []
    rewards = []

    for episode in range(num_episodes):
        print(f"\nEvaluation Episode {episode + 1}/{num_episodes}")

        system = BaseGraphObstacleTowerTAMPSystem.create_default(
            seed=seed + episode,
            render_mode=render_mode,
            num_obstacle_blocks=3,
        )

        planner = TaskThenMotionPlanner(
            system.types,
            system.predicates,
            system.perceiver,
            system.operators,
            system.skills,
            planner_id="pyperplan",
        )

        obs, info = system.env.reset(seed=seed + episode)
        planner.reset(obs, info)

        for step in range(max_steps):
            action = planner.step(obs)
            obs, reward, done, _, info = system.env.step(action)
            if done:
                success = float(reward) > 0
                successes.append(success)
                lengths.append(step + 1)
                rewards.append(float(reward))
                print(f"  {'SUCCESS' if success else 'FAIL'} in {step + 1} steps, reward: {reward}")
                break
        else:
            successes.append(False)
            lengths.append(max_steps)
            rewards.append(0.0)
            print(f"  TIMEOUT at {max_steps} steps")

        system.env.close()  # type: ignore[no-untyped-call]

        print(f"Current Success Rate: {np.mean(successes):.2%}")
        print(f"Current Avg Episode Length: {np.mean(lengths):.2f}")

    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    print(f"Success Rate: {np.mean(successes):.2%}")
    print(f"Average Episode Length: {np.mean(lengths):.2f}")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print("=" * 80)

    # Save results
    results_dir = Path(save_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "obstacle_tower_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("env_name: GraphObstacleTowerTAMPSystem\n")
        f.write(f"method: Pure Planning\n")
        f.write(f"seed: {seed}\n")
        f.write(f"num_episodes: {num_episodes}\n")
        f.write(f"success_rate: {np.mean(successes)}\n")
        f.write(f"avg_episode_length: {np.mean(lengths)}\n")
        f.write(f"avg_reward: {np.mean(rewards)}\n")

    print(f"\nResults saved to {results_file}")

    return {
        "success_rate": float(np.mean(successes)),
        "avg_episode_length": float(np.mean(lengths)),
        "avg_reward": float(np.mean(rewards)),
    }


def run_cluttered_drawer_planning(
    seed: int = 123,
    render_mode: str | None = None,
    max_steps: int = 10000,
    num_episodes: int = 100,
    save_dir: str = "results/pure_planning",
) -> dict:
    """Run pure planning baseline on ClutteredDrawer environment."""
    print("\n" + "=" * 80)
    print("Running Pure Planning on ClutteredDrawer Environment")
    print(f"Evaluating over {num_episodes} episodes")
    print("=" * 80)

    successes = []
    lengths = []
    rewards = []

    for episode in range(num_episodes):
        print(f"\nEvaluation Episode {episode + 1}/{num_episodes}")

        system = BaseClutteredDrawerTAMPSystem.create_default(
            seed=seed + episode,
            render_mode=render_mode,
        )

        planner = TaskThenMotionPlanner(
            system.types,
            system.predicates,
            system.perceiver,
            system.operators,
            system.skills,
            planner_id="pyperplan",
        )

        obs, info = system.env.reset(seed=seed + episode)
        planner.reset(obs, info)

        for step in range(max_steps):
            action = planner.step(obs)
            obs, reward, done, _, info = system.env.step(action)
            if done:
                success = float(reward) > 0
                successes.append(success)
                lengths.append(step + 1)
                rewards.append(float(reward))
                print(f"  {'SUCCESS' if success else 'FAIL'} in {step + 1} steps, reward: {reward}")
                break
        else:
            successes.append(False)
            lengths.append(max_steps)
            rewards.append(0.0)
            print(f"  TIMEOUT at {max_steps} steps")

        system.env.close()  # type: ignore[no-untyped-call]

        print(f"Current Success Rate: {np.mean(successes):.2%}")
        print(f"Current Avg Episode Length: {np.mean(lengths):.2f}")

    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    print(f"Success Rate: {np.mean(successes):.2%}")
    print(f"Average Episode Length: {np.mean(lengths):.2f}")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print("=" * 80)

    # Save results
    results_dir = Path(save_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "cluttered_drawer_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("env_name: ClutteredDrawerTAMPSystem\n")
        f.write(f"method: Pure Planning\n")
        f.write(f"seed: {seed}\n")
        f.write(f"num_episodes: {num_episodes}\n")
        f.write(f"success_rate: {np.mean(successes)}\n")
        f.write(f"avg_episode_length: {np.mean(lengths)}\n")
        f.write(f"avg_reward: {np.mean(rewards)}\n")

    print(f"\nResults saved to {results_file}")

    return {
        "success_rate": float(np.mean(successes)),
        "avg_episode_length": float(np.mean(lengths)),
        "avg_reward": float(np.mean(rewards)),
    }


def run_cleanup_table_planning(
    seed: int = 123,
    render_mode: str | None = None,
    max_steps: int = 10000,
    max_replans: int = 5,
    max_steps_per_plan: int = 500,
    num_episodes: int = 100,
    save_dir: str = "results/pure_planning",
) -> dict:
    """Run pure planning baseline on CleanupTable environment with
    replanning."""
    print("\n" + "=" * 80)
    print("Running Pure Planning on CleanupTable Environment (with replanning)")
    print(f"Evaluating over {num_episodes} episodes")
    print("=" * 80)

    successes = []
    lengths = []
    rewards = []

    for episode in range(num_episodes):
        print(f"\nEvaluation Episode {episode + 1}/{num_episodes}")

        system = BaseCleanupTableTAMPSystem.create_default(
            seed=seed + episode,
            render_mode=render_mode,
        )

        planner = TaskThenMotionPlanner(
            system.types,
            system.predicates,
            system.perceiver,
            system.operators,
            system.skills,
            planner_id="pyperplan",
        )

        obs, info = system.env.reset(seed=seed + episode)

        total_steps = 0
        episode_success = False
        episode_reward = 0.0

        for replan_attempt in range(max_replans):
            planner.reset(obs, info)
            steps_in_current_plan = 0

            for _ in range(max_steps_per_plan):
                steps_in_current_plan += 1
                total_steps += 1

                try:
                    action = planner.step(obs)
                except Exception as e:
                    print(f"  Planner failed with exception: {e}. Replanning...")
                    break

                obs, reward, done, _, info = system.env.step(action)
                if done:
                    episode_success = float(reward) > 0
                    episode_reward = float(reward)
                    print(f"  {'SUCCESS' if episode_success else 'FAIL'} in {total_steps} steps, reward: {reward}")
                    break

                if total_steps >= max_steps:
                    break

            if done or total_steps >= max_steps:
                break

        successes.append(episode_success)
        lengths.append(total_steps)
        rewards.append(episode_reward)

        system.env.close()  # type: ignore[no-untyped-call]

        print(f"Current Success Rate: {np.mean(successes):.2%}")
        print(f"Current Avg Episode Length: {np.mean(lengths):.2f}")

    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    print(f"Success Rate: {np.mean(successes):.2%}")
    print(f"Average Episode Length: {np.mean(lengths):.2f}")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print("=" * 80)

    # Save results
    results_dir = Path(save_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "cleanup_table_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("env_name: CleanupTableTAMPSystem\n")
        f.write(f"method: Pure Planning\n")
        f.write(f"seed: {seed}\n")
        f.write(f"num_episodes: {num_episodes}\n")
        f.write(f"success_rate: {np.mean(successes)}\n")
        f.write(f"avg_episode_length: {np.mean(lengths)}\n")
        f.write(f"avg_reward: {np.mean(rewards)}\n")

    print(f"\nResults saved to {results_file}")

    return {
        "success_rate": float(np.mean(successes)),
        "avg_episode_length": float(np.mean(lengths)),
        "avg_reward": float(np.mean(rewards)),
    }


def main() -> None:
    """Main function to run pure planning baselines."""
    parser = argparse.ArgumentParser(
        description="Run pure planning baselines on PyBullet environments"
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=[
            "obstacle_tower",
            "cluttered_drawer",
            "cleanup_table",
        ],
        help="Environment to run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (rgb_array mode)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate (default: 100)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/pure_planning",
        help="Directory to save results (default: results/pure_planning)",
    )

    args = parser.parse_args()
    render_mode = "rgb_array" if args.render else None

    if args.env == "obstacle_tower":
        run_obstacle_tower_planning(
            seed=args.seed,
            render_mode=render_mode,
            max_steps=args.max_steps,
            num_episodes=args.num_episodes,
            save_dir=args.save_dir,
        )
    elif args.env == "cluttered_drawer":
        run_cluttered_drawer_planning(
            seed=args.seed,
            render_mode=render_mode,
            max_steps=args.max_steps,
            num_episodes=args.num_episodes,
            save_dir=args.save_dir,
        )
    elif args.env == "cleanup_table":
        run_cleanup_table_planning(
            seed=args.seed,
            render_mode=render_mode,
            max_steps=args.max_steps,
            num_episodes=args.num_episodes,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()  # type: ignore[no-untyped-call]
