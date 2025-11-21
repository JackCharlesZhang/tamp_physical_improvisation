"""Test if PickUp skill completes and planner transitions to PlaceOnTarget."""

from src.tamp_improv.benchmarks.dyn_obstruction2d import BaseDynObstruction2DTAMPSystem
from task_then_motion_planning.planning import TaskThenMotionPlanner

def main():
    print("=" * 80)
    print("Testing PickUp Skill Completion and Planner Transition")
    print("=" * 80)

    # Create system
    system = BaseDynObstruction2DTAMPSystem.create_default(
        num_obstructions=2,
        seed=42,
        render_mode="rgb_array"
    )

    # Create planner
    planner = TaskThenMotionPlanner(
        system.types,
        system.predicates,
        system.perceiver,
        system.operators,
        system.skills,
        planner_id="pyperplan",
    )

    # Reset environment
    obs, info = system.env.reset(seed=42)
    planner.reset(obs, info)

    print("\nExecuting plan...")
    print(f"Plan: {[str(op) for op in planner._current_task_plan]}")

    # Run for max 500 steps
    for step in range(500):
        action = planner.step(obs)
        obs, reward, done, _, info = system.env.step(action)

        if step % 10 == 0:
            print(f"Step {step}: Current operator: {planner._current_operator}")

        if done:
            print(f"\n✓ Goal reached in {step + 1} steps!")
            print(f"  Final reward: {reward}")
            break
    else:
        print(f"\n✗ Goal not reached within 500 steps")
        print(f"  Current operator: {planner._current_operator}")
        print(f"  Remaining plan: {planner._current_task_plan}")

    system.env.close()

if __name__ == "__main__":
    main()
