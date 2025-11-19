"""Pure planning baselines for PyBullet environments."""

import argparse
from pathlib import Path

from gymnasium.wrappers import RecordVideo
from task_then_motion_planning.planning import TaskThenMotionPlanner

from tamp_improv.benchmarks.dyn_obstruction2d import BaseDynObstruction2DTAMPSystem

try:
    from prbench_bilevel_planning.agent import BilevelPlanningAgent
    SESAME_AVAILABLE = True
except ImportError:
    SESAME_AVAILABLE = False
# from tamp_improv.benchmarks.pybullet_cleanup_table import (
#     BaseCleanupTableTAMPSystem,
# )
# from tamp_improv.benchmarks.pybullet_cluttered_drawer import (
#     BaseClutteredDrawerTAMPSystem,
# )
# from tamp_improv.benchmarks.pybullet_obstacle_tower_graph import (
#     BaseGraphObstacleTowerTAMPSystem,
# )


# def run_obstacle_tower_planning(
#     seed: int = 124,
#     render_mode: str | None = None,
#     max_steps: int = 500,
# ) -> None:
#     """Run pure planning baseline on GraphObstacleTower environment."""
#     print("\n" + "=" * 80)
#     print("Running Pure Planning on GraphObstacleTower Environment")
#     print("=" * 80)

#     system = BaseGraphObstacleTowerTAMPSystem.create_default(
#         seed=seed,
#         render_mode=render_mode,
#         num_obstacle_blocks=3,
#     )

#     planner = TaskThenMotionPlanner(
#         system.types,
#         system.predicates,
#         system.perceiver,
#         system.operators,
#         system.skills,
#         planner_id="pyperplan",
#     )

#     obs, info = system.env.reset(seed=seed)
#     planner.reset(obs, info)

#     for step in range(max_steps):
#         action = planner.step(obs)
#         obs, reward, done, _, info = system.env.step(action)
#         if done:
#             print(f"\nGoal reached in {step + 1} steps!")
#             print(f"Final reward: {reward}")
#             if float(reward) > 0:
#                 print("SUCCESS: Task completed successfully!")
#             break
#     else:
#         print(f"\nFAILED: Goal not reached within {max_steps} steps")

#     system.env.close()  # type: ignore[no-untyped-call]


# def run_cluttered_drawer_planning(
#     seed: int = 123,
#     render_mode: str | None = None,
#     max_steps: int = 10000,
# ) -> None:
#     """Run pure planning baseline on ClutteredDrawer environment."""
#     print("\n" + "=" * 80)
#     print("Running Pure Planning on ClutteredDrawer Environment")
#     print("=" * 80)

#     system = BaseClutteredDrawerTAMPSystem.create_default(
#         seed=seed,
#         render_mode=render_mode,
#     )

#     planner = TaskThenMotionPlanner(
#         system.types,
#         system.predicates,
#         system.perceiver,
#         system.operators,
#         system.skills,
#         planner_id="pyperplan",
#     )

#     obs, info = system.env.reset(seed=seed)
#     planner.reset(obs, info)

#     for step in range(max_steps):
#         action = planner.step(obs)
#         obs, reward, done, _, info = system.env.step(action)
#         if done:
#             print(f"\nGoal reached in {step + 1} steps!")
#             print(f"Final reward: {reward}")
#             if float(reward) > 0:
#                 print("SUCCESS: Task completed successfully!")
#             break
#     else:
#         print(f"\nFAILED: Goal not reached within {max_steps} steps")

#     system.env.close()  # type: ignore[no-untyped-call]


# def run_cleanup_table_planning(
#     seed: int = 123,
#     render_mode: str | None = None,
#     max_steps: int = 10000,
#     max_replans: int = 5,
#     max_steps_per_plan: int = 500,
# ) -> None:
#     """Run pure planning baseline on CleanupTable environment with
#     replanning."""
#     print("\n" + "=" * 80)
#     print("Running Pure Planning on CleanupTable Environment (with replanning)")
#     print("=" * 80)

#     system = BaseCleanupTableTAMPSystem.create_default(
#         seed=seed,
#         render_mode=render_mode,
#     )

#     planner = TaskThenMotionPlanner(
#         system.types,
#         system.predicates,
#         system.perceiver,
#         system.operators,
#         system.skills,
#         planner_id="pyperplan",
#     )

#     obs, info = system.env.reset(seed=seed)

#     total_steps = 0
#     for replan_attempt in range(max_replans):
#         print(f"\nPlanning attempt {replan_attempt + 1}/{max_replans}")

#         planner.reset(obs, info)
#         steps_in_current_plan = 0

#         for _ in range(max_steps_per_plan):
#             steps_in_current_plan += 1
#             total_steps += 1

#             try:
#                 action = planner.step(obs)
#             except Exception as e:
#                 print(f"Planner failed with exception: {e}. Replanning...")
#                 break

#             obs, reward, done, _, info = system.env.step(action)
#             if done:
#                 print(f"\nGoal reached in {total_steps} total steps!")
#                 print(
#                     f"Steps in final plan: {steps_in_current_plan} "
#                     f"(attempt {replan_attempt + 1})"
#                 )
#                 print(f"Final reward: {reward}")
#                 if float(reward) > 0:
#                     print("SUCCESS: Task completed successfully!")
#                 system.env.close()  # type: ignore[no-untyped-call]
#                 return

#             if total_steps >= max_steps:
#                 system.env.close()  # type: ignore[no-untyped-call]
#                 return

#     print(f"\nFAILED: Goal not reached after {max_replans} planning attempts")
#     system.env.close()  # type: ignore[no-untyped-call]


def run_dyn_obstruction2d_planning(
    seed: int = 42,
    render_mode: str | None = None,
    max_steps: int = 200,
    num_obstructions: int = 2,
    record_video: bool = False,
    video_folder: str = "videos/dyn_obstruction2d_planning",
) -> None:
    """Run pure planning baseline on DynObstruction2D environment."""
    print("\n" + "=" * 80)
    print("Running Pure Planning on DynObstruction2D Environment")
    print("=" * 80)
    print(f"Number of obstructions: {num_obstructions}")
    print(f"Seed: {seed}")
    print(f"Max steps: {max_steps}")
    if record_video:
        print(f"Recording video to: {video_folder}")
    print("=" * 80)

    system = BaseDynObstruction2DTAMPSystem.create_default(
        seed=seed,
        render_mode=render_mode,
        num_obstructions=num_obstructions,
    )

    # Wrap with video recording if requested
    if record_video:
        if render_mode is None:
            print("WARNING: --record-video requires --render to be enabled!")
            print("Enabling render mode automatically...")
            system.env.render_mode = "rgb_array"

        Path(video_folder).mkdir(parents=True, exist_ok=True)
        system.env = RecordVideo(
            system.env,
            video_folder,
            episode_trigger=lambda _: True,
            name_prefix=f"seed_{seed}",
        )

    planner = TaskThenMotionPlanner(
        system.types,
        system.predicates,
        system.perceiver,
        system.operators,
        system.skills,
        planner_id="pyperplan",
    )

    obs, info = system.env.reset(seed=seed)
    objects, atoms, goal = system.perceiver.reset(obs, info)

    print(f"\nObjects: {[obj.name for obj in objects]}")
    print(f"Initial atoms ({len(atoms)}):")
    for atom in sorted(atoms, key=str)[:10]:  # Show first 10
        print(f"  - {atom}")
    if len(atoms) > 10:
        print(f"  ... and {len(atoms) - 10} more")
    print(f"Goal ({len(goal)}):")
    for atom in sorted(goal, key=str):
        print(f"  - {atom}")

    planner.reset(obs, info)

    for step in range(max_steps):
        action = planner.step(obs)
        obs, reward, done, _, info = system.env.step(action)

        if step % 20 == 0:
            print(f"Step {step}: reward={reward:.3f}")

        if done:
            print(f"\nGoal reached in {step + 1} steps!")
            print(f"Final reward: {reward}")
            if float(reward) > 0:
                print("SUCCESS: Task completed successfully!")
            break
    else:
        print(f"\nFAILED: Goal not reached within {max_steps} steps")

    system.env.close()  # type: ignore[no-untyped-call]


def run_dyn_obstruction2d_sesame_planning(
    seed: int = 42,
    render_mode: str | None = None,
    max_steps: int = 200,
    num_obstructions: int = 2,
    record_video: bool = False,
    video_folder: str = "videos/dyn_obstruction2d_sesame",
    max_abstract_plans: int = 10,
    samples_per_step: int = 10,
    max_skill_horizon: int = 100,
) -> None:
    """Run SESAME bilevel planning baseline on DynObstruction2D environment."""
    if not SESAME_AVAILABLE:
        print("ERROR: SESAME planner not available!")
        print("Install with: pip install -e 'path/to/prbench-bilevel-planning'")
        return

    print("\n" + "=" * 80)
    print("Running SESAME Bilevel Planning on DynObstruction2D Environment")
    print("=" * 80)
    print(f"Number of obstructions: {num_obstructions}")
    print(f"Seed: {seed}")
    print(f"Max steps: {max_steps}")
    print(f"Max abstract plans: {max_abstract_plans}")
    print(f"Samples per step: {samples_per_step}")
    print(f"Max skill horizon: {max_skill_horizon}")
    if record_video:
        print(f"Recording video to: {video_folder}")
    print("=" * 80)

    system = BaseDynObstruction2DTAMPSystem.create_default(
        seed=seed,
        render_mode=render_mode,
        num_obstructions=num_obstructions,
    )

    # Wrap with video recording if requested
    if record_video:
        if render_mode is None:
            print("WARNING: --record-video requires --render to be enabled!")
            print("Enabling render mode automatically...")
            system.env.render_mode = "rgb_array"

        Path(video_folder).mkdir(parents=True, exist_ok=True)
        system.env = RecordVideo(
            system.env,
            video_folder,
            episode_trigger=lambda _: True,
            name_prefix=f"seed_{seed}",
        )

    # Create SESAME models
    sesame_models = system.create_sesame_models()

    # Create SESAME agent
    agent = BilevelPlanningAgent(
        env_models=sesame_models,
        seed=seed,
        max_abstract_plans=max_abstract_plans,
        samples_per_step=samples_per_step,
        max_skill_horizon=max_skill_horizon,
        heuristic_name="hff",
        planning_timeout=120.0,  # Increased from 30 to 120 seconds
    )

    obs, info = system.env.reset(seed=seed)
    objects, atoms, goal = system.perceiver.reset(obs, info)

    print(f"\nObjects: {[obj.name for obj in objects]}")
    print(f"Initial atoms ({len(atoms)}):")
    for atom in sorted(atoms, key=str)[:10]:  # Show first 10
        print(f"  - {atom}")
    if len(atoms) > 10:
        print(f"  ... and {len(atoms) - 10} more")
    print(f"Goal ({len(goal)}):")
    for atom in sorted(goal, key=str):
        print(f"  - {atom}")

    # Reset agent
    print("\nPlanning...")

    # Debug: Print what SESAME models sees
    sesame_models = system.create_sesame_models()
    initial_state = sesame_models.observation_to_state(obs)
    initial_abstract = sesame_models.state_abstractor(initial_state)
    goal = sesame_models.goal_deriver(initial_state)

    print(f"\nDEBUG - Initial abstract state atoms ({len(initial_abstract.atoms)}):")
    for atom in sorted(initial_abstract.atoms, key=str):
        print(f"  - {atom}")

    print(f"\nDEBUG - Goal atoms ({len(goal.atoms)}):")
    for atom in sorted(goal.atoms, key=str):
        print(f"  - {atom}")

    print(f"\nDEBUG - Operators available:")
    for op in sesame_models.operators:
        print(f"  - {op.name}: params={[v.name for v in op.parameters]}")
        print(f"    Preconditions: {op.preconditions}")
        print(f"    Add effects: {op.add_effects}")
        print(f"    Delete effects: {op.delete_effects}")

    print(f"\nDEBUG - Object types:")
    for obj in initial_abstract.objects:
        print(f"  - {obj.name}: {obj.type.name}")

    try:
        agent.reset(obs, info)
        print("Planning completed successfully!")
    except Exception as e:
        print(f"Planning failed: {e}")
        import traceback
        traceback.print_exc()
        system.env.close()  # type: ignore[no-untyped-call]
        return

    # Execute plan
    for step in range(max_steps):
        try:
            action = agent.get_action()
        except Exception as e:
            print(f"\nAgent failed at step {step}: {e}")
            break

        obs, reward, done, _, info = system.env.step(action)

        if step % 20 == 0:
            print(f"Step {step}: reward={reward:.3f}")

        if done:
            print(f"\nGoal reached in {step + 1} steps!")
            print(f"Final reward: {reward}")
            if float(reward) > 0:
                print("SUCCESS: Task completed successfully!")
            break
    else:
        print(f"\nFAILED: Goal not reached within {max_steps} steps")

    system.env.close()  # type: ignore[no-untyped-call]


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
            "dyn_obstruction2d",
        ],
        help="Environment to run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
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
        "--num-obstructions",
        type=int,
        default=2,
        help="Number of obstruction blocks (for dyn_obstruction2d)",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record video of the episode (implies --render)",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="videos",
        help="Folder to save videos (default: videos/)",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="task-then-motion",
        choices=["task-then-motion", "sesame"],
        help="Planner type (for dyn_obstruction2d): task-then-motion or sesame",
    )
    parser.add_argument(
        "--max-abstract-plans",
        type=int,
        default=10,
        help="Max abstract plans for SESAME planner",
    )
    parser.add_argument(
        "--samples-per-step",
        type=int,
        default=10,
        help="Samples per step for SESAME planner",
    )
    parser.add_argument(
        "--max-skill-horizon",
        type=int,
        default=100,
        help="Max skill horizon for SESAME planner",
    )

    args = parser.parse_args()
    render_mode = "rgb_array" if (args.render or args.record_video) else None

    if args.env == "obstacle_tower":
        run_obstacle_tower_planning(
            seed=args.seed,
            render_mode=render_mode,
            max_steps=args.max_steps,
        )
    elif args.env == "cluttered_drawer":
        run_cluttered_drawer_planning(
            seed=args.seed,
            render_mode=render_mode,
            max_steps=args.max_steps,
        )
    elif args.env == "cleanup_table":
        run_cleanup_table_planning(
            seed=args.seed,
            render_mode=render_mode,
            max_steps=args.max_steps,
        )
    elif args.env == "dyn_obstruction2d":
        if args.planner == "sesame":
            video_folder = f"{args.video_folder}/dyn_obstruction2d_sesame"
            run_dyn_obstruction2d_sesame_planning(
                seed=args.seed,
                render_mode=render_mode,
                max_steps=args.max_steps,
                num_obstructions=args.num_obstructions,
                record_video=args.record_video,
                video_folder=video_folder,
                max_abstract_plans=args.max_abstract_plans,
                samples_per_step=args.samples_per_step,
                max_skill_horizon=args.max_skill_horizon,
            )
        else:  # task-then-motion
            video_folder = f"{args.video_folder}/dyn_obstruction2d_planning"
            run_dyn_obstruction2d_planning(
                seed=args.seed,
                render_mode=render_mode,
                max_steps=args.max_steps,
                num_obstructions=args.num_obstructions,
                record_video=args.record_video,
                video_folder=video_folder,
            )


if __name__ == "__main__":
    main()  # type: ignore[no-untyped-call]
