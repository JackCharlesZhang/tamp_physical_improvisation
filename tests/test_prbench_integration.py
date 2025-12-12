"""Tests for PRBench integration adapters."""

import pytest
from prbench.envs.dynamic2d.dyn_obstruction2d import DynObstruction2DEnv
from prbench_bilevel_planning.env_models.dynamic2d.dynobstruction2d import (
    create_bilevel_planning_models,
)

from tamp_improv.benchmarks.prbench_integration import (
    PRBenchPerceiver,
    PRBenchPredicateContainer,
)


class TestPRBenchPredicateContainer:
    """Tests for PRBenchPredicateContainer."""

    def test_initialization(self):
        """Test that container can be created from PRBench predicates."""
        env = DynObstruction2DEnv(num_obstructions=2)
        sesame_models = create_bilevel_planning_models(
            env.observation_space, env.action_space, num_obstructions=2
        )

        container = PRBenchPredicateContainer(sesame_models.predicates)

        # Should have 4 predicates (PRBench's simple set)
        assert len(container) == 4

    def test_getitem(self):
        """Test predicate access by name."""
        env = DynObstruction2DEnv(num_obstructions=2)
        sesame_models = create_bilevel_planning_models(
            env.observation_space, env.action_space, num_obstructions=2
        )

        container = PRBenchPredicateContainer(sesame_models.predicates)

        # Should be able to access predicates by name
        hand_empty = container["HandEmpty"]
        assert hand_empty.name == "HandEmpty"

        holding_tgt = container["HoldingTgt"]
        assert holding_tgt.name == "HoldingTgt"

        holding_obs = container["HoldingObstruction"]
        assert holding_obs.name == "HoldingObstruction"

        on_surface = container["OnTgtSurface"]
        assert on_surface.name == "OnTgtSurface"

    def test_as_set(self):
        """Test conversion to set."""
        env = DynObstruction2DEnv(num_obstructions=2)
        sesame_models = create_bilevel_planning_models(
            env.observation_space, env.action_space, num_obstructions=2
        )

        container = PRBenchPredicateContainer(sesame_models.predicates)
        pred_set = container.as_set()

        # Should return a set
        assert isinstance(pred_set, set)
        assert len(pred_set) == 4

        # Should contain same predicates as original
        assert pred_set == sesame_models.predicates

    def test_contains(self):
        """Test membership checking."""
        env = DynObstruction2DEnv(num_obstructions=2)
        sesame_models = create_bilevel_planning_models(
            env.observation_space, env.action_space, num_obstructions=2
        )

        container = PRBenchPredicateContainer(sesame_models.predicates)

        assert "HandEmpty" in container
        assert "HoldingTgt" in container
        assert "NonExistentPredicate" not in container


class TestPRBenchPerceiver:
    """Tests for PRBenchPerceiver."""

    def test_initialization(self):
        """Test that perceiver can be created."""
        env = DynObstruction2DEnv(num_obstructions=2)
        sesame_models = create_bilevel_planning_models(
            env.observation_space, env.action_space, num_obstructions=2
        )

        perceiver = PRBenchPerceiver(
            state_abstractor_fn=sesame_models.state_abstractor,
            goal_deriver_fn=sesame_models.goal_deriver,
        )

        assert perceiver is not None
        assert perceiver.state_abstractor is not None
        assert perceiver.goal_deriver is not None

    def test_reset(self):
        """Test perceiver reset returns objects, atoms, and goal."""
        env = DynObstruction2DEnv(num_obstructions=2)
        sesame_models = create_bilevel_planning_models(
            env.observation_space, env.action_space, num_obstructions=2
        )

        perceiver = PRBenchPerceiver(
            state_abstractor_fn=sesame_models.state_abstractor,
            goal_deriver_fn=sesame_models.goal_deriver,
        )

        # Reset environment to get initial state
        obs, info = env.reset(seed=0)

        # Reset perceiver
        objects, atoms, goal = perceiver.reset(obs, info)

        # Should return sets
        assert isinstance(objects, set)
        assert isinstance(atoms, set)
        assert isinstance(goal, set)

        # Should have objects (robot, target_block, target_surface, 2 obstructions)
        assert len(objects) > 0
        print(f"Objects: {objects}")

        # Should have atoms (HandEmpty at start)
        assert len(atoms) > 0
        print(f"Initial atoms: {atoms}")

        # Should have goal (OnTgtSurface + HandEmpty)
        assert len(goal) > 0
        print(f"Goal atoms: {goal}")

    def test_step(self):
        """Test perceiver step returns current atoms."""
        env = DynObstruction2DEnv(num_obstructions=2)
        sesame_models = create_bilevel_planning_models(
            env.observation_space, env.action_space, num_obstructions=2
        )

        perceiver = PRBenchPerceiver(
            state_abstractor_fn=sesame_models.state_abstractor,
            goal_deriver_fn=sesame_models.goal_deriver,
        )

        # Reset environment
        obs, info = env.reset(seed=0)
        objects, initial_atoms, goal = perceiver.reset(obs, info)

        # Step should return same atoms as reset (state hasn't changed)
        step_atoms = perceiver.step(obs)
        assert step_atoms == initial_atoms

        # Take a random action
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)

        # Step should return atoms for new state
        new_atoms = perceiver.step(obs)
        assert isinstance(new_atoms, set)
        print(f"Atoms after action: {new_atoms}")

    def test_no_counting_predicates(self):
        """Verify PRBench uses simple predicates, not counting predicates."""
        env = DynObstruction2DEnv(num_obstructions=2)
        sesame_models = create_bilevel_planning_models(
            env.observation_space, env.action_space, num_obstructions=2
        )

        perceiver = PRBenchPerceiver(
            state_abstractor_fn=sesame_models.state_abstractor,
            goal_deriver_fn=sesame_models.goal_deriver,
        )

        obs, info = env.reset(seed=0)
        objects, atoms, goal = perceiver.reset(obs, info)

        # Get predicate names
        atom_pred_names = {atom.predicate.name for atom in atoms}
        goal_pred_names = {atom.predicate.name for atom in goal}
        all_pred_names = atom_pred_names | goal_pred_names

        # Should NOT have counting predicates
        assert "OneObstructionBlocking" not in all_pred_names
        assert "TwoObstructionsBlocking" not in all_pred_names
        assert "Clear" not in all_pred_names
        assert "Blocking" not in all_pred_names

        # Should have PRBench's simple predicates
        expected_predicates = {"HandEmpty", "HoldingTgt", "HoldingObstruction", "OnTgtSurface"}
        assert all_pred_names.issubset(expected_predicates)

        print(f"Predicates used: {all_pred_names}")
        print("✓ No counting predicates - using PRBench's clean predicate set!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
