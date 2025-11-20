#!/usr/bin/env python3
"""Simple test script for DynObstruction2D SLAP-compatible skills."""

import numpy as np
from src.tamp_improv.benchmarks.dyn_obstruction2d import DynObstruction2DTAMPSystem


def test_system_creation():
    """Test that the system can be created with skills."""
    print("Creating DynObstruction2DTAMPSystem...")
    system = DynObstruction2DTAMPSystem.create_default(
        num_obstructions=2,
        seed=42,
        render_mode=None,
    )

    print(f"✓ System created successfully")
    print(f"  - Environment: {type(system.env).__name__}")
    print(f"  - Wrapped environment: {type(system.wrapped_env).__name__}")
    print(f"  - Number of operators: {len(system.components.operators)}")
    print(f"  - Number of skills: {len(system.components.skills)}")

    # Check skills
    skill_names = {skill._get_operator_name() for skill in system.components.skills}
    print(f"  - Skill names: {sorted(skill_names)}")

    expected_skills = {"PickUp", "Place", "PlaceOnTarget"}
    if skill_names == expected_skills:
        print("✓ All expected skills present")
    else:
        print(f"✗ Missing skills: {expected_skills - skill_names}")
        print(f"✗ Extra skills: {skill_names - expected_skills}")
        return False

    return True


def test_environment_step():
    """Test that we can reset and step the environment."""
    print("\nTesting environment interactions...")
    system = DynObstruction2DTAMPSystem.create_default(num_obstructions=2, seed=42)

    # Reset environment
    obs, info = system.env.reset(seed=42)
    print(f"✓ Environment reset")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Observation type: {obs.dtype}")

    # Test a simple action
    action = np.array([0.01, 0.01, 0.0, 0.0, 0.0], dtype=np.float64)
    obs, reward, terminated, truncated, info = system.env.step(action)
    print(f"✓ Environment step completed")
    print(f"  - New observation shape: {obs.shape}")

    return True


def test_skill_execution():
    """Test that skills can generate actions."""
    print("\nTesting skill action generation...")
    system = DynObstruction2DTAMPSystem.create_default(num_obstructions=2, seed=42)

    # Reset environment
    obs, info = system.env.reset(seed=42)

    # Get PickUpSkill
    pickup_skill = next(
        s for s in system.components.skills
        if s._get_operator_name() == "PickUp"
    )

    # Get objects for the skill (robot, target_block)
    perceiver = system.components.perceiver
    objects = [perceiver._robot, perceiver._target_block]

    # Generate action
    action = pickup_skill._get_action_given_objects(objects, obs)
    print(f"✓ PickUpSkill generated action")
    print(f"  - Action: {action}")
    print(f"  - Action shape: {action.shape}")
    print(f"  - Action dtype: {action.dtype}")

    # Verify action is valid
    assert action.shape == (5,), f"Expected shape (5,), got {action.shape}"
    assert action.dtype == np.float64, f"Expected float64, got {action.dtype}"
    assert system.env.action_space.contains(action), "Action not in action space!"
    print(f"✓ Action is valid and in action space")

    return True


def test_wrapped_environment():
    """Test that the wrapped environment works for SLAP."""
    print("\nTesting wrapped environment (ImprovWrapper)...")
    system = DynObstruction2DTAMPSystem.create_default(num_obstructions=2, seed=42)

    # Reset wrapped environment
    obs, info = system.wrapped_env.reset(seed=42)
    print(f"✓ Wrapped environment reset")
    print(f"  - Observation shape: {obs.shape}")

    # Test a simple action
    action = np.array([0.01, 0.01, 0.0, 0.0, 0.0], dtype=np.float64)
    obs, reward, terminated, truncated, info = system.wrapped_env.step(action)
    print(f"✓ Wrapped environment step completed")
    print(f"  - Reward: {reward}")
    print(f"  - Terminated: {terminated}")
    print(f"  - Truncated: {truncated}")

    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing DynObstruction2D SLAP-Compatible Skills")
    print("=" * 70)

    tests = [
        ("System Creation", test_system_creation),
        ("Environment Step", test_environment_step),
        ("Skill Execution", test_skill_execution),
        ("Wrapped Environment", test_wrapped_environment),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} failed with error:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
