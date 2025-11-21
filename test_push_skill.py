#!/usr/bin/env python3
"""Test script to verify Push skill is loaded correctly."""

from src.tamp_improv.benchmarks.dyn_obstruction2d import DynObstruction2DTAMPSystem

def test_push_skill():
    print("Creating DynObstruction2D system with Push skill...")
    system = DynObstruction2DTAMPSystem.create_default(num_obstructions=2, seed=42)

    print("\nSkills loaded:")
    for skill in system.components.skills:
        print(f"  - {skill._get_operator_name()}")

    print("\nOperators loaded:")
    for op in system.components.operators:
        print(f"  - {op.name}: params={[v.name for v in op.parameters]}")

    print("\nPush skill details:")
    push_skills = [s for s in system.components.skills if s._get_operator_name() == "Push"]
    if push_skills:
        push = push_skills[0]
        print(f"  Push height offset: {push.PUSH_HEIGHT_OFFSET}")
        print(f"  Push distance: {push.PUSH_DISTANCE}")
        print("  ✓ Push skill successfully loaded!")
    else:
        print("  ✗ Push skill not found!")

    print("\nTesting basic environment interaction...")
    obs, info = system.env.reset(seed=42)
    print(f"  Environment reset successful. Observation shape: {obs.shape}")

    # Test a single step
    action = system.env.action_space.sample()
    obs, reward, terminated, truncated, info = system.env.step(action)
    print(f"  Step executed successfully. Reward: {reward}")

    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_push_skill()
