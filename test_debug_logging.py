#!/usr/bin/env python3
"""Test script to demonstrate debug logging for physics collisions and state tracking."""

import numpy as np
from src.tamp_improv.benchmarks.dyn_obstruction2d import DynObstruction2DTAMPSystem

def test_debug_logging():
    """Run a simple episode to test debug logging."""
    print("=" * 80)
    print("Testing Physics Debug Logging")
    print("=" * 80)
    print()

    # Create system (debug patches will be applied automatically)
    system = DynObstruction2DTAMPSystem.create_default(
        num_obstructions=2,
        seed=137,  # Use a different seed to get different layout
        render_mode=None,
    )

    # Reset environment
    obs, info = system.env.reset(seed=137)
    print(f"Environment reset complete. Observation shape: {obs.shape}\n")

    print("=" * 80)
    print("Running episode with random actions...")
    print("Watch for debug output:")
    print("  [BOUNDS WARNING] - Object getting close to walls")
    print("  [BOUNDS ERROR] - Object outside world bounds")
    print("  [COLLISION] - Robot colliding with walls")
    print("  [ROBOT_REVERT] - Robot reverting after wall collision")
    print("  [HELD_STATE] - Object transitioning DYNAMIC <-> KINEMATIC")
    print("=" * 80)
    print()

    # Run some steps with random actions
    for step in range(100):
        # Random action
        action = system.env.action_space.sample()

        # Occasionally try to move towards boundaries to trigger warnings
        if step % 20 == 0:
            # Try to move right
            action = np.array([0.05, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        elif step % 20 == 10:
            # Try to move left
            action = np.array([-0.05, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        obs, reward, terminated, truncated, info = system.env.step(action)

        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break

    print("\n" + "=" * 80)
    print("Test complete! Review the output above for debug messages.")
    print("=" * 80)


if __name__ == "__main__":
    test_debug_logging()
