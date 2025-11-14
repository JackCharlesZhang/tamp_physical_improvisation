"""Tests for pruning module."""

import numpy as np
import pytest

from tamp_improv.approaches.improvisational.graph import (
    GroundAtom,
    PlanningGraph,
    PlanningGraphEdge,
    PlanningGraphNode,
)
from tamp_improv.benchmarks.base import Predicate
from tamp_improv.approaches.improvisational.policies.base import GoalConditionedTrainingData
from tamp_improv.approaches.improvisational.pruning import (
    prune_none,
    prune_random,
    prune_training_data,
)


def _make_test_atom(name: str) -> GroundAtom:
    """Create a test ground atom with a given name."""
    pred = Predicate(name, [])  # Empty list for 0-arity predicate
    return GroundAtom(pred, [])


def test_prune_none():
    """Test that prune_none returns all shortcuts unchanged."""
    # Create minimal training data
    training_data = GoalConditionedTrainingData(
        states=["state1", "state2", "state3"],
        current_atoms=[{_make_test_atom("a")}, {_make_test_atom("b")}, {_make_test_atom("c")}],
        goal_atoms=[{_make_test_atom("b")}, {_make_test_atom("c")}, {_make_test_atom("d")}],
        config={},
        node_states={0: ["state1"], 1: ["state2"], 2: ["state3"]},
        valid_shortcuts=[(0, 1), (0, 2), (1, 2)],
        node_atoms={0: {_make_test_atom("a")}, 1: {_make_test_atom("b")}, 2: {_make_test_atom("c")}},
    )

    result = prune_none(training_data)

    assert len(result.states) == 3
    assert len(result.valid_shortcuts) == 3
    assert result.states == training_data.states


def test_prune_random():
    """Test that prune_random selects correct number of shortcuts."""
    rng = np.random.default_rng(42)

    # Create training data with 10 shortcuts
    training_data = GoalConditionedTrainingData(
        states=[f"state{i}" for i in range(10)],
        current_atoms=[{_make_test_atom(f"a{i}")} for i in range(10)],
        goal_atoms=[{_make_test_atom(f"b{i}")} for i in range(10)],
        config={},
        node_states={i: [f"state{i}"] for i in range(10)},
        valid_shortcuts=[(i, i + 1) for i in range(10)],
        node_atoms={i: {_make_test_atom(f"a{i}")} for i in range(10)},
    )

    # Prune to 5 shortcuts
    result = prune_random(training_data, max_shortcuts=5, rng=rng)

    assert len(result.states) == 5
    assert len(result.valid_shortcuts) == 5
    assert result.config["pruning_method"] == "random"
    assert result.config["max_shortcuts"] == 5


def test_prune_random_under_limit():
    """Test that prune_random keeps all shortcuts if under limit."""
    rng = np.random.default_rng(42)

    # Create training data with 3 shortcuts
    training_data = GoalConditionedTrainingData(
        states=["state1", "state2", "state3"],
        current_atoms=[{_make_test_atom("a")}, {_make_test_atom("b")}, {_make_test_atom("c")}],
        goal_atoms=[{_make_test_atom("b")}, {_make_test_atom("c")}, {_make_test_atom("d")}],
        config={},
        node_states={0: ["state1"], 1: ["state2"], 2: ["state3"]},
        valid_shortcuts=[(0, 1), (0, 2), (1, 2)],
        node_atoms={0: {_make_test_atom("a")}, 1: {_make_test_atom("b")}, 2: {_make_test_atom("c")}},
    )

    # Try to prune to 10 shortcuts (but only 3 exist)
    result = prune_random(training_data, max_shortcuts=10, rng=rng)

    assert len(result.states) == 3
    assert len(result.valid_shortcuts) == 3


def test_prune_training_data_dispatch():
    """Test that prune_training_data dispatches to correct method."""
    rng = np.random.default_rng(42)

    # Create minimal training data
    training_data = GoalConditionedTrainingData(
        states=["state1", "state2"],
        current_atoms=[{_make_test_atom("a")}, {_make_test_atom("b")}],
        goal_atoms=[{_make_test_atom("b")}, {_make_test_atom("c")}],
        config={},
        node_states={0: ["state1"], 1: ["state2"]},
        valid_shortcuts=[(0, 1)],
        node_atoms={0: {_make_test_atom("a")}, 1: {_make_test_atom("b")}},
    )

    # Create minimal planning graph
    node0 = PlanningGraphNode(id=0, atoms=frozenset([_make_test_atom("a")]))
    node1 = PlanningGraphNode(id=1, atoms=frozenset([_make_test_atom("b")]))
    planning_graph = PlanningGraph()
    planning_graph.nodes = [node0, node1]

    # Mock system (not used for 'none' pruning)
    system = None

    # Test 'none' method
    config = {"pruning_method": "none", "seed": 42}
    result = prune_training_data(training_data, system, planning_graph, config, rng)
    assert len(result.states) == 2

    # Test 'random' method
    config = {"pruning_method": "random", "max_shortcuts_per_graph": 1, "seed": 42}
    result = prune_training_data(training_data, system, planning_graph, config, rng)
    assert len(result.states) == 1
    assert result.config["pruning_method"] == "random"


def test_prune_training_data_unknown_method():
    """Test that unknown pruning method raises error."""
    rng = np.random.default_rng(42)

    # Create minimal training data
    training_data = GoalConditionedTrainingData(
        states=["state1"],
        current_atoms=[{_make_test_atom("a")}],
        goal_atoms=[{_make_test_atom("b")}],
        config={},
        node_states={0: ["state1"]},
        valid_shortcuts=[(0, 1)],
        node_atoms={0: {_make_test_atom("a")}},
    )

    # Create minimal planning graph
    node0 = PlanningGraphNode(id=0, atoms=frozenset([_make_test_atom("a")]))
    planning_graph = PlanningGraph()
    planning_graph.nodes = [node0]

    config = {"pruning_method": "invalid_method", "seed": 42}

    with pytest.raises(ValueError, match="Unknown pruning method"):
        prune_training_data(training_data, None, planning_graph, config, rng)
