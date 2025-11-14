"""Collection functions for SLAP training data.

This module provides clean data collection without any pruning logic.
Pruning is handled separately in pruning.py.
"""

from typing import Any

import numpy as np

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.graph_training import (
    ShortcutCandidate,
    collect_states_for_all_nodes,
    identify_shortcut_candidates,
)
from tamp_improv.approaches.improvisational.policies.base import (
    GoalConditionedTrainingData,
)
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem


def collect_all_shortcuts(
    system: ImprovisationalTAMPSystem,
    approach: ImprovisationalTAMPApproach,
    config: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> GoalConditionedTrainingData:
    """Collect ALL possible shortcuts without any pruning.

    This is the base collection function that builds planning graphs and
    identifies all valid shortcuts. No pruning is applied - that should be
    done separately using functions from pruning.py.

    Args:
        system: The TAMP system
        approach: The improvisational TAMP approach
        config: Configuration dictionary
        rng: Random number generator

    Returns:
        GoalConditionedTrainingData with all possible shortcuts
    """
    if rng is None:
        seed = config.get("seed", 42)
        rng = np.random.default_rng(seed)

    approach.training_mode = True
    collect_episodes = config.get("collect_episodes", 10)

    # We'll collect data across multiple episodes and aggregate
    all_states = []
    all_current_atoms = []
    all_goal_atoms = []
    all_node_states = {}
    all_valid_shortcuts = []
    all_node_atoms = {}

    print(f"\n{'=' * 80}")
    print(f"Collecting shortcuts from {collect_episodes} episodes")
    print(f"{'=' * 80}")

    for episode in range(collect_episodes):
        print(f"\n=== Episode {episode + 1}/{collect_episodes} ===", flush=True)

        # Reset and build planning graph
        print("[DEBUG] Resetting system...", flush=True)
        obs, info = system.reset()
        print("[DEBUG] Building planning graph (this may take a while)...", flush=True)
        _ = approach.reset(obs, info)
        print("[DEBUG] Planning graph built", flush=True)

        assert (
            hasattr(approach, "planning_graph") and approach.planning_graph is not None
        ), "Planning graph not created"
        planning_graph = approach.planning_graph

        # Collect states at each node
        if (
            hasattr(approach, "observed_states")
            and approach.observed_states is not None
        ):
            print("Using observed states from approach")
            observed_states = approach.observed_states
        else:
            print("Collecting states for all nodes...")
            observed_states = collect_states_for_all_nodes(
                system, planning_graph, max_attempts=3
            )
            # Convert to multi-state format
            observed_states = {k: [v] for k, v in observed_states.items()}

        print(
            f"Graph has {len(planning_graph.nodes)} nodes, "
            f"{len(planning_graph.edges)} edges"
        )
        print(f"Collected states for {len(observed_states)} nodes")

        # Identify ALL shortcut candidates (no pruning)
        shortcut_candidates = identify_shortcut_candidates(
            planning_graph,
            observed_states,
        )

        print(f"Identified {len(shortcut_candidates)} shortcut candidates")

        # Add all shortcuts to the training data
        for candidate in shortcut_candidates:
            source_id = candidate.source_node.id
            target_id = candidate.target_node.id

            if source_id in observed_states and observed_states[source_id] is not None:
                # Use the first state from source node
                for source_state in observed_states[source_id]:
                    all_states.append(source_state)
                    all_current_atoms.append(candidate.source_atoms)
                    all_goal_atoms.append(candidate.target_atoms)

                # Track this shortcut
                all_valid_shortcuts.append((source_id, target_id))

        # Merge node states (using episode-specific node IDs if needed)
        # For simplicity, we only keep the last episode's planning graph
        # This is consistent with collect_goal_conditioned_training_data
        all_node_states = observed_states
        for node in planning_graph.nodes:
            if node.id in observed_states:
                all_node_atoms[node.id] = set(node.atoms)

    print(
        f"\n{'=' * 80}\n"
        f"Collection complete: {len(all_states)} training examples, "
        f"{len(all_valid_shortcuts)} shortcuts\n"
        f"{'=' * 80}"
    )

    approach.training_mode = False

    # Return as GoalConditionedTrainingData
    return GoalConditionedTrainingData(
        states=all_states,
        current_atoms=all_current_atoms,
        goal_atoms=all_goal_atoms,
        config={
            **config,
            "collection_method": "all_shortcuts",
            "collect_episodes": collect_episodes,
            "num_shortcuts_collected": len(all_valid_shortcuts),
        },
        node_states=all_node_states,
        valid_shortcuts=all_valid_shortcuts,
        node_atoms=all_node_atoms,
    )


def collect_shortcuts_single_episode(
    system: ImprovisationalTAMPSystem,
    approach: ImprovisationalTAMPApproach,
) -> GoalConditionedTrainingData:
    """Collect all shortcuts from a single episode.

    This is a simplified version that works with a single planning graph.
    Useful for quick tests or when you want to work with one graph at a time.

    Args:
        system: The TAMP system
        approach: The improvisational TAMP approach (should be reset already)

    Returns:
        GoalConditionedTrainingData with all shortcuts from this episode
    """
    approach.training_mode = True

    assert (
        hasattr(approach, "planning_graph") and approach.planning_graph is not None
    ), "Planning graph not created - call approach.reset() first"
    planning_graph = approach.planning_graph

    # Collect states at each node
    if hasattr(approach, "observed_states") and approach.observed_states is not None:
        observed_states = approach.observed_states
    else:
        observed_states = collect_states_for_all_nodes(
            system, planning_graph, max_attempts=3
        )
        observed_states = {k: [v] for k, v in observed_states.items()}

    # Identify all shortcut candidates
    shortcut_candidates = identify_shortcut_candidates(
        planning_graph,
        observed_states,
    )

    # Build training data
    states = []
    current_atoms_list = []
    goal_atoms_list = []
    valid_shortcuts = []
    node_atoms = {}

    for candidate in shortcut_candidates:
        source_id = candidate.source_node.id
        target_id = candidate.target_node.id

        if source_id in observed_states and observed_states[source_id] is not None:
            for source_state in observed_states[source_id]:
                states.append(source_state)
                current_atoms_list.append(candidate.source_atoms)
                goal_atoms_list.append(candidate.target_atoms)

            valid_shortcuts.append((source_id, target_id))

    for node in planning_graph.nodes:
        if node.id in observed_states:
            node_atoms[node.id] = set(node.atoms)

    approach.training_mode = False

    return GoalConditionedTrainingData(
        states=states,
        current_atoms=current_atoms_list,
        goal_atoms=goal_atoms_list,
        config={
            "collection_method": "single_episode",
            "num_shortcuts_collected": len(valid_shortcuts),
        },
        node_states=observed_states,
        valid_shortcuts=valid_shortcuts,
        node_atoms=node_atoms,
    )
