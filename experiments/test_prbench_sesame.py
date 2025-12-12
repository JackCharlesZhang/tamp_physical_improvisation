"""Test PRBench integration with Sesame bilevel planner."""

import argparse
from pathlib import Path

import numpy as np
import tomsgeoms2d.structs

# Monkey-patch Tobject before importing prbench
if not hasattr(tomsgeoms2d.structs, "Tobject"):
    from tomsgeoms2d.structs import Lobject

    tomsgeoms2d.structs.Tobject = Lobject

from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.structs import PlanningProblem
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
)
from gymnasium.wrappers import RecordVideo
from prbench.envs.dynamic2d.dyn_obstruction2d import DynObstruction2DEnv
from prbench_bilevel_planning.env_models import create_bilevel_planning_models


def run_sesame_planning(
    seed: int = 42,
    num_obstructions: int = 2,
    max_abstract_plans: int = 10,
    samples_per_step: int = 10,
    timeout: float = 30.0,
    record_video: bool = False,
    video_folder: str = "videos/sesame_prbench",
) -> None:
    """Run Sesame bilevel planner on PRBench DynObstruction2D environment."""
    print("\n" + "=" * 80)
    print("Testing PRBench Integration with Sesame Bilevel Planner")
    print("=" * 80)
    print(f"Number of obstructions: {num_obstructions}")
    print(f"Seed: {seed}")
    print(f"Max abstract plans: {max_abstract_plans}")
    print(f"Samples per step: {samples_per_step}")
    print(f"Timeout: {timeout}s")
    if record_video:
        print(f"Recording video to: {video_folder}")
    print("=" * 80)

    # Create environment
    render_mode = "rgb_array" if record_video else None
    env = DynObstruction2DEnv(num_obstructions=num_obstructions, render_mode=render_mode)

    # Wrap with video recording if requested
    if record_video:
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder,
            episode_trigger=lambda _: True,
            name_prefix=f"seed_{seed}",
        )

    # Create PRBench bilevel planning models
    sesame_models = create_bilevel_planning_models(
        "dynobstruction2d",
        env.observation_space,
        env.action_space,
        num_obstructions=num_obstructions,
    )

    print(f"\nPRBench Models:")
    print(f"  Predicates ({len(sesame_models.predicates)}):")
    for pred in sorted(sesame_models.predicates, key=lambda p: p.name):
        print(f"    - {pred.name}{pred.types}")
    print(f"  Operators ({len(sesame_models.operators)}):")
    for op in sorted(sesame_models.operators, key=lambda o: o.name):
        print(f"    - {op.name}: {len(op.preconditions)} preconds → +{len(op.add_effects)}/-{len(op.delete_effects)} effects")
    print(f"  Skills ({len(sesame_models.skills)})")

    # Reset environment
    obs, _ = env.reset(seed=seed)
    initial_state = sesame_models.observation_to_state(obs)
    goal = sesame_models.goal_deriver(initial_state)

    print(f"\nInitial state:")
    initial_abstract = sesame_models.state_abstractor(initial_state)
    print(f"  Objects: {[str(o) for o in initial_abstract.objects]}")
    print(f"  Atoms ({len(initial_abstract.atoms)}):")
    for atom in sorted(initial_abstract.atoms, key=str):
        print(f"    - {atom}")
    print(f"\nGoal ({len(goal.atoms)}):")
    for atom in sorted(goal.atoms, key=str):
        print(f"    - {atom}")

    # Create planning problem
    def transition_fn(state, action):
        """Execute action in environment and return next state."""
        # Convert state to observation
        # This is simplified - in practice we'd need to sync with env
        obs, reward, done, truncated, info = env.step(action)
        next_state = sesame_models.observation_to_state(obs)
        return next_state

    problem = PlanningProblem(
        env.observation_space,
        env.action_space,
        initial_state,
        transition_fn,
        goal,
    )

    # Create trajectory sampler
    trajectory_sampler = ParameterizedControllerTrajectorySampler(
        controller_generator=RelationalControllerGenerator(sesame_models.skills),
        transition_function=transition_fn,
        state_abstractor=sesame_models.state_abstractor,
        max_trajectory_steps=500,  # Max steps per operator execution
    )

    # Create abstract plan generator (uses PDDL planner on high level)
    abstract_plan_generator = RelationalHeuristicSearchAbstractPlanGenerator(
        sesame_models.types,
        sesame_models.predicates,
        sesame_models.operators,
        heuristic_name="hff",  # Fast-forward heuristic
        seed=seed,
    )

    # Create abstract successor generator
    abstract_successor_fn = RelationalAbstractSuccessorGenerator(sesame_models.operators)

    # Create Sesame planner
    planner = SesamePlanner(
        abstract_plan_generator,
        trajectory_sampler,
        max_abstract_plans,
        samples_per_step,
        abstract_successor_fn,
        sesame_models.state_abstractor,
        seed=seed,
    )

    # Run planning
    print(f"\n{'='*80}")
    print("Running Sesame bilevel planner...")
    print(f"{'='*80}")

    print(f"\n[DEBUG] Problem details:")
    print(f"  Initial state type: {type(initial_state)}")
    print(f"  Goal type: {type(goal)}")
    print(f"  Goal atoms: {list(goal.atoms)}")

    print(f"\n[DEBUG] Testing abstract plan generator directly...")
    test_gen = abstract_plan_generator(
        initial_state,
        sesame_models.state_abstractor(initial_state),
        goal,
        timeout=10.0,
        bilevel_planning_graph=None,
    )
    try:
        s_plan, a_plan = next(test_gen)
        print(f"[DEBUG] ✓ Abstract planner generated a plan:")
        print(f"  State plan length: {len(s_plan)}")
        print(f"  Action plan length: {len(a_plan)}")
        for i, (s, a) in enumerate(zip(s_plan, a_plan)):
            print(f"  Step {i}: {a}")
    except StopIteration:
        print(f"[DEBUG] ✗ Abstract planner generated NO plans (StopIteration)")
    except Exception as e:
        print(f"[DEBUG] ✗ Abstract planner error: {type(e).__name__}: {e}")

    print(f"\n[DEBUG] Calling planner.run()...")
    try:
        plan, bpg = planner.run(problem, timeout=timeout)
        print(f"[DEBUG] planner.run() returned successfully")
    except Exception as e:
        print(f"[DEBUG] ERROR in planner.run(): {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    if plan is not None:
        print(f"\n✓ SUCCESS! Found plan with {len(plan.actions)} actions")
        print(f"  States: {len(plan.states)}")
        print(f"  Final state reached goal: {goal.check(plan.states[-1])}")
    else:
        print(f"\n✗ FAILED: No plan found within {timeout}s")
        if bpg is not None:
            print(f"  Bilevel planning graph nodes: {len(bpg._state_ids)}")

    env.close()

    if plan is not None and record_video:
        print(f"\nVideo saved to: {video_folder}/seed_{seed}-episode-0.mp4")
        print(f"To view: rsync -avz jz4267@della-gpu.princeton.edu:~/tamp_physical_improvisation/{video_folder}/*.mp4 ./")

    return plan is not None


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Test PRBench with Sesame planner")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-obstructions", type=int, default=2, help="Number of obstruction blocks"
    )
    parser.add_argument(
        "--max-abstract-plans",
        type=int,
        default=10,
        help="Max abstract plans to try",
    )
    parser.add_argument(
        "--samples-per-step",
        type=int,
        default=10,
        help="Trajectory samples per abstract step",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Planning timeout in seconds"
    )
    parser.add_argument(
        "--record-video", action="store_true", help="Record video of execution"
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="videos/sesame_prbench",
        help="Folder to save videos",
    )

    args = parser.parse_args()

    success = run_sesame_planning(
        seed=args.seed,
        num_obstructions=args.num_obstructions,
        max_abstract_plans=args.max_abstract_plans,
        samples_per_step=args.samples_per_step,
        timeout=args.timeout,
        record_video=args.record_video,
        video_folder=args.video_folder,
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
