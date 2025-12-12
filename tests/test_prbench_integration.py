"""Tests for PRBench integration adapters."""

import pytest

# Monkey-patch Tobject into tomsgeoms2d if it doesn't exist
# MUST happen BEFORE importing prbench!
import tomsgeoms2d.structs

if not hasattr(tomsgeoms2d.structs, "Tobject"):
    from tomsgeoms2d.structs import Lobject

    tomsgeoms2d.structs.Tobject = Lobject

from prbench.envs.dynamic2d.dyn_obstruction2d import DynObstruction2DEnv
from prbench_bilevel_planning.env_models import create_bilevel_planning_models

from tamp_improv.benchmarks.prbench_integration import (
    BasePRBenchSLAPSystem,
    PRBenchPerceiver,
    PRBenchPredicateContainer,
    PRBenchSkill,
    PRBenchSLAPSystem,
)


class TestPRBenchPredicateContainer:
    """Tests for PRBenchPredicateContainer."""

    def test_initialization(self):
        """Test that container can be created from PRBench predicates."""
        env = DynObstruction2DEnv(num_obstructions=2)
        sesame_models = create_bilevel_planning_models(
            "dynobstruction2d", env.observation_space, env.action_space, num_obstructions=2
        )

        container = PRBenchPredicateContainer(sesame_models.predicates)

        # Should have 5 predicates (PRBench's set including AboveTgt)
        assert len(container) == 5

    def test_getitem(self):
        """Test predicate access by name."""
        env = DynObstruction2DEnv(num_obstructions=2)
        sesame_models = create_bilevel_planning_models(
            "dynobstruction2d", env.observation_space, env.action_space, num_obstructions=2
        )

        container = PRBenchPredicateContainer(sesame_models.predicates)

        # Should be able to access predicates by name
        hand_empty = container["HandEmpty"]
        assert hand_empty.name == "HandEmpty"

        holding_tgt = container["HoldingTgt"]
        assert holding_tgt.name == "HoldingTgt"

        holding_obs = container["HoldingObstruction"]
        assert holding_obs.name == "HoldingObstruction"

        on_surface = container["OnTgt"]
        assert on_surface.name == "OnTgt"

    def test_as_set(self):
        """Test conversion to set."""
        env = DynObstruction2DEnv(num_obstructions=2)
        sesame_models = create_bilevel_planning_models(
            "dynobstruction2d", env.observation_space, env.action_space, num_obstructions=2
        )

        container = PRBenchPredicateContainer(sesame_models.predicates)
        pred_set = container.as_set()

        # Should return a set
        assert isinstance(pred_set, set)
        assert len(pred_set) == 5

        # Should contain same predicates as original
        assert pred_set == sesame_models.predicates

    def test_contains(self):
        """Test membership checking."""
        env = DynObstruction2DEnv(num_obstructions=2)
        sesame_models = create_bilevel_planning_models(
            "dynobstruction2d", env.observation_space, env.action_space, num_obstructions=2
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
            "dynobstruction2d", env.observation_space, env.action_space, num_obstructions=2
        )

        perceiver = PRBenchPerceiver(
            observation_to_state_fn=sesame_models.observation_to_state,
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
            "dynobstruction2d", env.observation_space, env.action_space, num_obstructions=2
        )

        perceiver = PRBenchPerceiver(
            observation_to_state_fn=sesame_models.observation_to_state,
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
            "dynobstruction2d", env.observation_space, env.action_space, num_obstructions=2
        )

        perceiver = PRBenchPerceiver(
            observation_to_state_fn=sesame_models.observation_to_state,
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
            "dynobstruction2d", env.observation_space, env.action_space, num_obstructions=2
        )

        perceiver = PRBenchPerceiver(
            observation_to_state_fn=sesame_models.observation_to_state,
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

        # Should have PRBench's simple predicates (including AboveTgt for dynamic2d)
        expected_predicates = {"HandEmpty", "HoldingTgt", "HoldingObstruction", "OnTgt", "AboveTgt"}
        assert all_pred_names.issubset(expected_predicates)

        print(f"Predicates used: {all_pred_names}")
        print("✓ No counting predicates - using PRBench's clean predicate set!")


class TestPRBenchSkill:
    """Tests for PRBenchSkill."""

    def test_skill_initialization(self):
        """Test that skills can be created from PRBench's LiftedSkills."""
        env = DynObstruction2DEnv(num_obstructions=2)
        sesame_models = create_bilevel_planning_models(
            "dynobstruction2d", env.observation_space, env.action_space, num_obstructions=2
        )

        # Create skills from each PRBench LiftedSkill
        skills = []
        for lifted_skill in sesame_models.skills:
            skill = PRBenchSkill(
                lifted_skill=lifted_skill,
                observation_to_state_fn=sesame_models.observation_to_state,
            )
            skills.append(skill)

        # Should have created skills for all PRBench operators
        assert len(skills) == len(sesame_models.skills)
        print(f"Created {len(skills)} skills")

    def test_skill_can_execute(self):
        """Test that skills correctly identify their operators."""
        env = DynObstruction2DEnv(num_obstructions=2)
        sesame_models = create_bilevel_planning_models(
            "dynobstruction2d", env.observation_space, env.action_space, num_obstructions=2
        )

        # Get a skill and its operator
        lifted_skill = list(sesame_models.skills)[0]
        skill = PRBenchSkill(
            lifted_skill=lifted_skill,
            observation_to_state_fn=sesame_models.observation_to_state,
        )

        # Reset environment to get objects
        obs, info = env.reset(seed=0)
        state = sesame_models.observation_to_state(obs)
        abstract_state = sesame_models.state_abstractor(state)

        # Try to ground the operator
        operator = lifted_skill.operator
        print(f"Testing operator: {operator.name}")
        print(f"Operator parameters: {operator.parameters}")

        # Get matching objects from state
        objects = list(abstract_state.objects)
        print(f"Available objects: {[str(o) for o in objects]}")

        # Create a ground operator by selecting appropriate objects
        # (this would normally be done by a planner)
        if len(operator.parameters) <= len(objects):
            ground_params = tuple(objects[:len(operator.parameters)])
            ground_op = operator.ground(ground_params)

            # Skill should recognize this operator
            assert skill.can_execute(ground_op)
            print(f"✓ Skill can execute {ground_op.name}")


class TestPRBenchSLAPSystem:
    """Tests for PRBenchSLAPSystem."""

    def test_base_system_initialization(self):
        """Test that base system can be created."""
        system = BasePRBenchSLAPSystem.create_default(num_obstructions=2)

        assert system is not None
        assert system.name == "PRBenchSLAPSystem"
        print(f"System name: {system.name}")

    def test_system_has_prbench_components(self):
        """Test that system has all PRBench components properly wrapped."""
        system = BasePRBenchSLAPSystem.create_default(num_obstructions=2)

        # Should have types
        assert len(system.types) > 0
        print(f"Types: {[t.name for t in system.types]}")

        # Should have predicates (5 for dynamic2d)
        assert len(system.predicates) == 5
        pred_names = {p.name for p in system.predicates}
        print(f"Predicates: {pred_names}")

        # Should have PRBench's simple predicates
        assert "HandEmpty" in pred_names
        assert "HoldingTgt" in pred_names
        assert "HoldingObstruction" in pred_names
        assert "OnTgt" in pred_names
        assert "AboveTgt" in pred_names

        # Should NOT have counting predicates
        assert "OneObstructionBlocking" not in pred_names
        assert "TwoObstructionsBlocking" not in pred_names
        assert "Clear" not in pred_names

    def test_system_has_operators(self):
        """Test that system has PRBench operators."""
        system = BasePRBenchSLAPSystem.create_default(num_obstructions=2)

        # Should have operators
        assert len(system.operators) > 0
        op_names = {op.name for op in system.operators}
        print(f"Operators: {op_names}")

        # Check for expected PRBench operators (generic, no counting!)
        # Exact names may vary, but should be generic

    def test_system_has_skills(self):
        """Test that system has PRBench skills wrapped as SLAP skills."""
        system = BasePRBenchSLAPSystem.create_default(num_obstructions=2)

        # Should have skills
        assert len(system.skills) > 0
        print(f"Number of skills: {len(system.skills)}")

        # All skills should be PRBenchSkill instances
        for skill in system.skills:
            assert isinstance(skill, PRBenchSkill)
            print(f"Skill operator: {skill._lifted_operator.name}")

    def test_system_has_perceiver(self):
        """Test that system has PRBench perceiver."""
        system = BasePRBenchSLAPSystem.create_default(num_obstructions=2)

        # Should have perceiver
        assert system.perceiver is not None
        assert isinstance(system.perceiver, PRBenchPerceiver)

    def test_system_environment(self):
        """Test that system creates correct environment."""
        system = BasePRBenchSLAPSystem.create_default(num_obstructions=2)

        # Should have environment
        assert system.env is not None
        assert isinstance(system.env, DynObstruction2DEnv)

        # Test reset
        obs, info = system.reset(seed=0)
        assert obs is not None
        print(f"Observation shape: {obs.shape}")

    def test_improvisational_system(self):
        """Test that improvisational system can be created."""
        system = PRBenchSLAPSystem.create_default(num_obstructions=2, seed=0)

        assert system is not None
        assert system.name == "PRBenchSLAPSystem"

        # Should have wrapped environment for training
        assert hasattr(system, "wrapped_env")
        assert system.wrapped_env is not None
        print("✓ Improvisational system created with wrapped environment")

    def test_domain_creation(self):
        """Test that PDDL domain can be created."""
        system = BasePRBenchSLAPSystem.create_default(num_obstructions=2)

        domain = system.get_domain()
        assert domain is not None
        assert domain.name == "prbench-dyn-obstruction2d-domain"
        assert len(domain.predicates) == 5
        assert len(domain.operators) > 0
        print(f"Domain: {domain.name}")
        print(f"  Predicates: {len(domain.predicates)}")
        print(f"  Operators: {len(domain.operators)}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
