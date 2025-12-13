"""Tests for ClutteredStorage2D environment with TAMP."""

import numpy as np
import pytest
from gymnasium.wrappers import TimeLimit, RecordVideo # New import
from relational_structs import GroundAtom, Object
from task_then_motion_planning.planning import TaskThenMotionPlanner

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.benchmarks.clutteredstorage_system import (
    BaseGraphClutteredStorage2DTAMPSystem,
    GraphClutteredStorage2DTAMPSystem,
)


class TestClutteredStorage2DSystem:
    """Test suite for ClutteredStorage2D TAMP system."""

    def test_base_system_creation(self):
        """Test that BaseGraphClutteredStorage2DTAMPSystem can be created."""
        system = BaseGraphClutteredStorage2DTAMPSystem.create_default(
            n_blocks=3, render_mode="rgb_array", seed=42
        )

        assert system is not None
        assert system.env is not None
        assert system.components is not None

    def test_improvisational_system_creation(self):
        """Test that GraphClutteredStorage2DTAMPSystem can be created."""
        system = GraphClutteredStorage2DTAMPSystem.create_default(
            n_blocks=3, render_mode="rgb_array", seed=42
        )

        assert system is not None
        assert system.env is not None
        assert system.wrapped_env is not None
        assert system.components is not None

    def test_system_has_required_components(self):
        """Test that system has all required components for TAMP."""
        system = BaseGraphClutteredStorage2DTAMPSystem.create_default(
            n_blocks=3, render_mode="rgb_array", seed=42
        )

        # Check types
        assert len(system.types) == 6, "Should have robot, block, and shelf types and their parent types"
        type_names = {t.name for t in system.types}
        assert "crv_robot" in type_names
        assert "target_block" in type_names
        assert "shelf" in type_names

        # Check predicates
        assert len(system.predicates) == 4, "Should have 4 predicates"
        predicate_names = {p.name for p in system.predicates}
        assert "Holding" in predicate_names
        assert "HandEmpty" in predicate_names
        assert "OnShelf" in predicate_names
        assert "NotOnShelf" in predicate_names

        # Check operators
        assert len(system.operators) == 4, "Should have 4 operators"
        operator_names = {op.name for op in system.operators}
        assert "PickBlockNotOnShelf" in operator_names
        assert "PickBlockOnShelf" in operator_names
        assert "PlaceBlockNotOnShelf" in operator_names
        assert "PlaceBlockOnShelf" in operator_names

        # Check skills
        assert len(system.skills) == 4, "Should have 4 skills"
        skill_names = {skill.lifted_operator.name for skill in system.skills}
        assert "PickBlockNotOnShelf" in skill_names
        assert "PickBlockOnShelf" in skill_names
        assert "PlaceBlockNotOnShelf" in skill_names
        assert "PlaceBlockOnShelf" in skill_names

        # Check perceiver
        assert system.perceiver is not None

    def test_perceiver_reset(self):
        """Test that perceiver correctly generates objects, atoms, and goal."""
        system = BaseGraphClutteredStorage2DTAMPSystem.create_default(
            n_blocks=3, render_mode="rgb_array", seed=42
        )

        obs, info = system.env.reset()
        objects, atoms, goal = system.perceiver.reset(obs, info)

        # Check objects
        assert len(objects) == 5, "Should have 1 robot + 3 blocks + 1 shelf"
        object_names = {obj.name for obj in objects}
        assert "robot" in object_names
        assert "shelf" in object_names
        for i in range(3):
            assert f"block{i}" in object_names

        # Check atoms
        assert isinstance(atoms, set)
        assert all(isinstance(atom, GroundAtom) for atom in atoms)

        # Check goal
        assert isinstance(goal, set)
        assert len(goal) == 4, "Goal should have 3 OnShelf atoms + 1 HandEmpty atom"

        # Verify goal structure
        goal_predicates = {atom.predicate.name for atom in goal}
        assert "OnShelf" in goal_predicates
        assert "HandEmpty" in goal_predicates

    def test_perceiver_step(self):
        """Test that perceiver correctly updates atoms on step."""
        system = BaseGraphClutteredStorage2DTAMPSystem.create_default(
            n_blocks=3, render_mode="rgb_array", seed=42
        )

        obs, info = system.env.reset()
        objects, initial_atoms, goal = system.perceiver.reset(obs, info)

        # Take a step in the environment
        action = system.env.action_space.sample()
        next_obs, _, _, _, _ = system.env.step(action)

        # Get updated atoms
        updated_atoms = system.perceiver.step(next_obs)

        assert isinstance(updated_atoms, set)
        assert all(isinstance(atom, GroundAtom) for atom in updated_atoms)

    def test_perceiver_encode_atoms_to_vector(self):
        """Test that perceiver can encode atoms to vector."""
        system = BaseGraphClutteredStorage2DTAMPSystem.create_default(
            n_blocks=3, render_mode="rgb_array", seed=42
        )

        obs, info = system.env.reset()
        objects, atoms, goal = system.perceiver.reset(obs, info)

        # Encode atoms to vector
        atom_vector = system.perceiver.encode_atoms_to_vector(atoms)

        assert isinstance(atom_vector, np.ndarray)
        assert atom_vector.dtype == np.float32
        assert len(atom_vector.shape) == 1, "Should be 1D vector"
        assert len(atom_vector) > 0, "Vector should not be empty"
        assert all((v == 0.0 or v == 1.0) for v in atom_vector), "Should be binary vector"

        # Encode goal to vector
        goal_vector = system.perceiver.encode_atoms_to_vector(goal)

        assert isinstance(goal_vector, np.ndarray)
        assert goal_vector.dtype == np.float32
        assert len(goal_vector) == len(atom_vector), "Should have same dimensionality"

    def test_domain_creation(self):
        """Test that PDDL domain can be created."""
        system = BaseGraphClutteredStorage2DTAMPSystem.create_default(
            n_blocks=3, render_mode="rgb_array", seed=42
        )

        domain = system.get_domain()

        assert domain is not None
        assert domain.name == "graph-clutteredstorage2d-domain"
        assert len(domain.operators) == 4
        assert len(domain.predicates) == 4
        assert len(domain.types) == 6

    def test_wrapped_env_creation(self):
        """Test that wrapped environment is created for improvisational system."""
        system = GraphClutteredStorage2DTAMPSystem.create_default(
            n_blocks=3, render_mode="rgb_array", seed=42
        )

        assert system.wrapped_env is not None
        assert hasattr(system.wrapped_env, "observation_space")
        assert hasattr(system.wrapped_env, "action_space")
        assert hasattr(system.wrapped_env, "perceiver")

    def test_improvisational_approach_creation(self):
        """Test that ImprovisationalTAMPApproach can be created with the system."""
        system = GraphClutteredStorage2DTAMPSystem.create_default(
            n_blocks=3, render_mode="rgb_array", seed=42
        )

        # Create a simple MultiRl policy for testing
        policy = MultiRLPolicy(seed=42)

        # Create approach
        approach = ImprovisationalTAMPApproach(
            system=system,
            policy=policy,
            seed=42,
            planner_id="pyperplan",
            max_skill_steps=150,
        )

        assert approach is not None
        assert approach.system == system
        assert approach.policy == policy

    def test_skill_lifted_operator_property(self):
        """Test that skills have the lifted_operator property."""
        system = BaseGraphClutteredStorage2DTAMPSystem.create_default(
            n_blocks=3, render_mode="rgb_array", seed=42
        )

        for skill in system.skills:
            assert hasattr(skill, "lifted_operator")
            assert skill.lifted_operator is not None
            assert skill.lifted_operator.name in {
                "PickBlockNotOnShelf",
                "PickBlockOnShelf",
                "PlaceBlockNotOnShelf",
                "PlaceBlockOnShelf",
            }

    def test_skill_can_execute(self):
        """Test that skills can check if they can execute operators."""
        system = BaseGraphClutteredStorage2DTAMPSystem.create_default(
            n_blocks=3, render_mode="rgb_array", seed=42
        )

        # Get a skill
        skill = next(iter(system.skills))

        # Create a ground operator from the skill's lifted operator
        obs, info = system.env.reset()
        objects, atoms, goal = system.perceiver.reset(obs, info)

        # Ground the operator with actual objects, matching types correctly
        grounding = []
        for param in skill.lifted_operator.parameters:
            # Find an object of the correct type
            matching_obj = next(obj for obj in objects if obj.is_instance(param.type))
            grounding.append(matching_obj)

        ground_op = skill.lifted_operator.ground(tuple(grounding))

        # Check if skill can execute its own operator
        assert skill.can_execute(ground_op)


@pytest.mark.parametrize("n_blocks", [1])
def test_different_num_blocks(n_blocks):
    """Test system creation with different numbers of blocks."""
    system = BaseGraphClutteredStorage2DTAMPSystem.create_default(
        n_blocks=n_blocks, render_mode="rgb_array", seed=42
    )

    obs, info = system.env.reset()
    objects, atoms, goal = system.perceiver.reset(obs, info)

    # Check correct number of objects
    assert len(objects) == n_blocks + 2, f"Should have {n_blocks} blocks + robot + shelf"

    # Check goal has correct number of OnShelf atoms
    on_shelf_goals = [atom for atom in goal if atom.predicate.name == "OnShelf"]
    assert len(on_shelf_goals) == n_blocks


def test_tamp_planner_integration():
    """Test that the system can be used with TaskThenMotionPlanner."""
    system = BaseGraphClutteredStorage2DTAMPSystem.create_default(
        n_blocks=1, render_mode="rgb_array", seed=42
    )

    env_time_limited = TimeLimit(system.env, max_episode_steps=500)

    env = RecordVideo(
        env_time_limited,
        video_folder="cluttered_storage_videos",
        episode_trigger=lambda x: True,
        name_prefix="run"
    )

    planner = TaskThenMotionPlanner(
        types=system.types,
        predicates=system.predicates,
        perceiver=system.perceiver,
        operators=system.operators,
        skills=system.skills,
        planner_id="pyperplan",
    )

    obs, info = env.reset()
    objects, atoms, goal = system.perceiver.reset(obs, info)

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

    # Try to take a few steps with the planner
    total_reward = 0
    for step in range(200):  # Just test a few steps
        # import ipdb

        # ipdb.set_trace()
        action = planner.step(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        print(f"Step {step + 1}: Reward: {reward}")

        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps")
            print(f"Total reward: {total_reward}")
            break

    env.close()


def test_multiple_inheritance_initialization():
    """Test that multiple inheritance is correctly initialized."""
    system = GraphClutteredStorage2DTAMPSystem.create_default(
        n_blocks=3, render_mode="rgb_array", seed=42
    )

    # Verify both base classes are properly initialized
    assert hasattr(system, "env")  # From BaseTAMPSystem
    assert hasattr(system, "wrapped_env")  # From ImprovisationalTAMPSystem
    assert hasattr(system, "components")  # From BaseTAMPSystem
    assert hasattr(system, "n_blocks")  # From both init methods

    # Verify the environment hierarchy
    assert system.env is not None
    assert system.wrapped_env is not None
    assert system.wrapped_env != system.env  # Should be wrapped


def test_predicate_container_interface():
    """Test that predicate container implements required protocol."""
    system = BaseGraphClutteredStorage2DTAMPSystem.create_default(
        n_blocks=3, render_mode="rgb_array", seed=42
    )

    # Test __getitem__
    holding = system.components.predicate_container["Holding"]
    assert holding.name == "Holding"

    hand_empty = system.components.predicate_container["HandEmpty"]
    assert hand_empty.name == "HandEmpty"

    on_shelf = system.components.predicate_container["OnShelf"]
    assert on_shelf.name == "OnShelf"

    not_on_shelf = system.components.predicate_container["NotOnShelf"]
    assert not_on_shelf.name == "NotOnShelf"

    # Test as_set
    predicates = system.components.predicate_container.as_set()
    assert len(predicates) == 4
    assert all(hasattr(p, "name") for p in predicates)
