"""Tests for DynObstruction2D environment with TAMP."""

import pytest

from tamp_improv.benchmarks.dyn_obstruction2d import BaseDynObstruction2DTAMPSystem


def test_dyn_obstruction2d_system_creation():
    """Test that DynObstruction2D TAMP system can be created."""
    try:
        tamp_system = BaseDynObstruction2DTAMPSystem.create_default(
            num_obstructions=2, render_mode="rgb_array", seed=42
        )
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    # Check that all components are initialized
    assert tamp_system is not None
    assert len(tamp_system.types) == 4  # robot, block, obstruction, surface
    assert len(tamp_system.predicates) == 6  # On, Clear, Holding, GripperEmpty, Obstructing, ObstructionClear
    assert len(tamp_system.operators) == 3  # PickUp, PlaceOnTarget, Push
    assert len(tamp_system.skills) == 3  # PickUpSkill, PlaceOnTargetSkill, PushSkill


def test_dyn_obstruction2d_env_reset():
    """Test that the environment can be reset."""
    try:
        tamp_system = BaseDynObstruction2DTAMPSystem.create_default(
            num_obstructions=2, render_mode="rgb_array", seed=42
        )
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    env = tamp_system.env
    obs, info = env.reset()

    # Check observation shape
    # For 2 obstructions: 14 (surface) + 15 (target_block) + 15*2 (obstructions) + 21 (robot) = 80
    assert obs.shape == (80,)

    # Check that perceiver can process the observation
    objects, atoms, goal = tamp_system.perceiver.reset(obs, info)

    assert len(objects) == 5  # robot, target_block, target_surface, obstruction0, obstruction1
    assert len(goal) == 2  # On(target_block, target_surface), GripperEmpty(robot)
    assert len(atoms) > 0  # Should have some initial atoms


def test_dyn_obstruction2d_env_step():
    """Test that the environment can take steps."""
    try:
        tamp_system = BaseDynObstruction2DTAMPSystem.create_default(
            num_obstructions=2, render_mode="rgb_array", seed=42
        )
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    env = tamp_system.env
    obs, info = env.reset()

    # Take a random action
    action = env.action_space.sample()
    obs_new, reward, terminated, truncated, info_new = env.step(action)

    # Check that observation changed
    assert obs_new.shape == obs.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_dyn_obstruction2d_domain():
    """Test that the PDDL domain is correctly constructed."""
    try:
        tamp_system = BaseDynObstruction2DTAMPSystem.create_default(
            num_obstructions=2, seed=42
        )
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    domain = tamp_system.get_domain()

    assert domain.name == "dyn-obstruction2d-domain"
    assert len(domain.operators) == 3
    assert len(domain.predicates) == 6
    assert len(domain.types) == 4


@pytest.mark.slow
def test_dyn_obstruction2d_with_planner():
    """Test DynObstruction2D environment with TAMP planner."""
    try:
        from gymnasium.wrappers import TimeLimit
        from task_then_motion_planning.planning import TaskThenMotionPlanner

        tamp_system = BaseDynObstruction2DTAMPSystem.create_default(
            num_obstructions=2, render_mode="rgb_array", seed=42
        )
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    env = TimeLimit(tamp_system.env, max_episode_steps=100)

    planner = TaskThenMotionPlanner(
        types=tamp_system.types,
        predicates=tamp_system.predicates,
        perceiver=tamp_system.perceiver,
        operators=tamp_system.operators,
        skills=tamp_system.skills,
        planner_id="pyperplan",
    )

    obs, info = env.reset()
    objects, atoms, goal = tamp_system.perceiver.reset(obs, info)
    print("Objects:", objects)
    print("Initial atoms:", atoms)
    print("Goal:", goal)

    try:
        planner.reset(obs, info)
    except Exception as e:
        print("Error during planner reset:", str(e))
        print(
            "Current problem:",
            planner._current_problem,  # pylint: disable=protected-access
        )
        print("Current domain:", planner._domain)  # pylint: disable=protected-access
        raise

    # Run a few steps to make sure planner works
    total_reward = 0
    for step in range(10):  # Just run 10 steps for quick test
        action = planner.step(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        print(f"Step {step + 1}: Action: {action}, Reward: {reward}")

        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps")
            print(f"Total reward: {total_reward}")
            break

    env.close()
