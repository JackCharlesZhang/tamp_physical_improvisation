"""Pure planning baselines for PyBullet environments."""

import argparse
import logging
from pathlib import Path

from gymnasium.wrappers import RecordVideo
from task_then_motion_planning.planning import TaskThenMotionPlanner

from tamp_improv.benchmarks.dyn_obstruction2d import BaseDynObstruction2DTAMPSystem

# Enable logging for debugging
logging.basicConfig(level=logging.INFO)
logging.getLogger("relational_structs").setLevel(logging.DEBUG)
logging.getLogger("task_then_motion_planning").setLevel(logging.DEBUG)
logging.getLogger("pyperplan").setLevel(logging.DEBUG)
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
    for atom in sorted(atoms, key=str):
        print(f"  - {atom}")
    print(f"Goal ({len(goal)}):")
    for atom in sorted(goal, key=str):
        print(f"  - {atom}")

    print(f"\n[DEBUG] Available operators:")
    for op in system.operators:
        print(f"  - {op.name}: params={op.parameters}")
        print(f"    preconditions: {op.preconditions}")
        print(f"    add_effects: {op.add_effects}")
        print(f"    delete_effects: {op.delete_effects}")

    print(f"\n[DEBUG] Calling planner.reset() to generate initial plan...")
    try:
        planner.reset(obs, info)
        print(f"[DEBUG] Planner.reset() completed successfully")
    except Exception as e:
        print(f"[DEBUG] ERROR during planner.reset(): {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n[DEBUG] Checking planner internal state:")
    print(f"  Planner type: {type(planner)}")
    print(f"  hasattr(planner, '_current_task_plan'): {hasattr(planner, '_current_task_plan')}")
    print(f"  hasattr(planner, '_current_operator'): {hasattr(planner, '_current_operator')}")
    print(f"  hasattr(planner, '_current_problem'): {hasattr(planner, '_current_problem')}")

    if hasattr(planner, '_current_task_plan'):
        print(f"  planner._current_task_plan: {planner._current_task_plan}")
        print(f"  type: {type(planner._current_task_plan)}")
        print(f"  length: {len(planner._current_task_plan) if planner._current_task_plan else 0}")
    if hasattr(planner, '_current_operator'):
        print(f"  planner._current_operator: {planner._current_operator}")

    print(f"\n[DEBUG] Current plan:")
    if hasattr(planner, '_current_task_plan') and planner._current_task_plan:
        print(f"  Task plan length: {len(planner._current_task_plan)}")
        for i, ground_op in enumerate(planner._current_task_plan):
            print(f"  {i+1}. {ground_op}")
    else:
        print("  (No plan found or plan is empty)")

    print(f"\n[DEBUG] Starting execution...")

    prev_obs = obs
    for step in range(max_steps):
        # Track operator transitions
        prev_operator = getattr(planner, '_current_operator', None)

        try:
            action = planner.step(obs)

            # Detect operator change
            curr_operator = getattr(planner, '_current_operator', None)
            if prev_operator != curr_operator and curr_operator is not None:
                # Operator changed! Log robot position at transition
                from relational_structs.spaces import ObjectCentricBoxSpace
                if isinstance(system.env.observation_space, ObjectCentricBoxSpace):
                    # Log the PREVIOUS observation (end state of previous operator)
                    if 'prev_obs' in locals():
                        obj_state_prev = system.env.observation_space.devectorize(prev_obs)
                        robot_objs_prev = [obj for obj in obj_state_prev if 'robot' in str(obj.name).lower()]
                        if robot_objs_prev:
                            robot_prev = robot_objs_prev[0]
                            rx_prev, ry_prev, rtheta_prev = obj_state_prev.get(robot_prev, 'x'), obj_state_prev.get(robot_prev, 'y'), obj_state_prev.get(robot_prev, 'theta')
                            print(f"\n[OPERATOR_CHANGE] Step {step-1}: {prev_operator.name} ended at: ({rx_prev:.3f}, {ry_prev:.3f}, θ={rtheta_prev:.3f})")

                    # Log the CURRENT observation (start state of new operator)
                    obj_state = system.env.observation_space.devectorize(obs)
                    robot_objs = [obj for obj in obj_state if 'robot' in str(obj.name).lower()]
                    if robot_objs:
                        robot = robot_objs[0]
                        rx, ry, rtheta = obj_state.get(robot, 'x'), obj_state.get(robot, 'y'), obj_state.get(robot, 'theta')
                        print(f"[OPERATOR_CHANGE] Step {step}: {curr_operator.name} starts at: ({rx:.3f}, {ry:.3f}, θ={rtheta:.3f})")
            # if print_action:
            #     print(f"[STEP {step}] Action returned: {action}, all_zeros={all(action == 0)}")
        except Exception as e:
            print(f"\n[SKILL FAILURE] Skill failed with exception: {type(e).__name__}: {e}")
            print("[PLANNER] Attempting to replan...")
            # Reset planner to trigger replanning
            planner.reset(obs, info)
            action = planner.step(obs)

        prev_obs = obs
        obs, reward, done, _, info = system.env.step(action)

        # if step % 20 == 0:
        #     print(f"Step {step}: reward={reward:.3f}")

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
