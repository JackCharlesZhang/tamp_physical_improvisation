"""Collection functions for SLAP training data.

This module provides clean data collection without any pruning logic.
Pruning is handled separately in pruning.py.
"""

from typing import Any

import numpy as np

from tamp_improv.approaches.improvisational.base import (
    ImprovisationalTAMPApproach,
    ShortcutSignature,
)
from tamp_improv.approaches.improvisational.graph import PlanningGraph
from tamp_improv.approaches.improvisational.graph_training import (
    ShortcutCandidate,
    collect_states_for_all_nodes,
    identify_shortcut_candidates,
)
from tamp_improv.approaches.improvisational.policies.base import (
    GoalConditionedTrainingData,
)
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
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
    all_shortcut_info = []

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
                    # Store shortcut info with node IDs (needed for MultiRL policy keys)
                    all_shortcut_info.append({
                        "source_node_id": source_id,
                        "target_node_id": target_id,
                    })

                # Track this shortcut
                all_valid_shortcuts.append((source_id, target_id))

                # Register signature with approach (matches collect_graph_based_training_data)
                signature = ShortcutSignature.from_context(
                    candidate.source_atoms,
                    candidate.target_atoms,
                )
                if signature not in approach.trained_signatures:
                    approach.trained_signatures.append(signature)

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
            "shortcut_info": all_shortcut_info,
        },
        node_states=all_node_states,
        valid_shortcuts=all_valid_shortcuts,
        node_atoms=all_node_atoms,
    )


def collect_total_shortcuts(
    system: ImprovisationalTAMPSystem,
    approach: ImprovisationalTAMPApproach,
    config: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> GoalConditionedTrainingData:
    """Collect all shortcuts from a unified total planning graph.

    This is more efficient than collect_all_shortcuts because it:
    1. Builds one large planning graph across multiple episodes
    2. Collects states for all nodes in the unified graph
    3. Identifies shortcuts once from the complete graph

    Args:
        system: The TAMP system
        approach: The improvisational TAMP approach
        config: Configuration dictionary
        rng: Random number generator

    Returns:
        GoalConditionedTrainingData with all shortcuts from the total graph
    """
    if rng is None:
        seed = config.get("seed", 42)
        rng = np.random.default_rng(seed)

    approach.training_mode = True
    collect_episodes = config.get("collect_episodes", 10)

    print(f"\n{'=' * 80}")
    print(f"Collecting total planning graph from {collect_episodes} episodes")
    print(f"{'=' * 80}")

    # Build unified planning graph across all episodes
    total_graph, node_states = collect_total_planning_graph(
        system=system,
        collect_episodes=collect_episodes,
        seed=config.get("seed", 42),
        planner_id=config.get("planner_id", "pyperplan"),
    )

    # Store the total graph in the approach
    approach.planning_graph = total_graph

    print(f"\n{'=' * 80}")
    print(f"Identifying shortcuts from total graph")
    print(f"{'=' * 80}")

    # Identify all shortcut candidates from the total graph
    # Note: node_states keys are frozensets of atoms, need to convert to node IDs
    # First, create a mapping from atoms to node IDs
    atom_to_node_id = {node.atoms: node.id for node in total_graph.nodes}

    # Convert node_states to use node IDs
    observed_states = {}
    for atom_set, states in node_states.items():
        if atom_set in atom_to_node_id:
            node_id = atom_to_node_id[atom_set]
            observed_states[node_id] = states

    shortcut_candidates = identify_shortcut_candidates(
        total_graph,
        observed_states,
    )

    print(f"Identified {len(shortcut_candidates)} shortcut candidates")

    # Build training data
    all_states = []
    all_current_atoms = []
    all_goal_atoms = []
    all_valid_shortcuts = []
    all_node_atoms = {}
    all_shortcut_info = []

    for candidate in shortcut_candidates:
        source_id = candidate.source_node.id
        target_id = candidate.target_node.id

        if source_id in observed_states and observed_states[source_id] is not None:
            for source_state in observed_states[source_id]:
                all_states.append(source_state)
                all_current_atoms.append(candidate.source_atoms)
                all_goal_atoms.append(candidate.target_atoms)
                all_shortcut_info.append({
                    "source_node_id": source_id,
                    "target_node_id": target_id,
                })

            all_valid_shortcuts.append((source_id, target_id))

            # Register signature with approach
            signature = ShortcutSignature.from_context(
                candidate.source_atoms,
                candidate.target_atoms,
            )
            if signature not in approach.trained_signatures:
                approach.trained_signatures.append(signature)

    # Store node atoms
    for node in total_graph.nodes:
        if node.id in observed_states:
            all_node_atoms[node.id] = set(node.atoms)

    print(
        f"\n{'=' * 80}\n"
        f"Collection complete: {len(all_states)} training examples, "
        f"{len(all_valid_shortcuts)} shortcuts\n"
        f"{'=' * 80}"
    )

    approach.training_mode = False

    return GoalConditionedTrainingData(
        states=all_states,
        current_atoms=all_current_atoms,
        goal_atoms=all_goal_atoms,
        config={
            **config,
            "collection_method": "total_shortcuts",
            "collect_episodes": collect_episodes,
            "num_shortcuts_collected": len(all_valid_shortcuts),
            "shortcut_info": all_shortcut_info,
        },
        node_states=observed_states,
        valid_shortcuts=all_valid_shortcuts,
        node_atoms=all_node_atoms,
        graph=total_graph,
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


def collect_total_planning_graph(
    system: ImprovisationalTAMPSystem,
    collect_episodes: int,
    seed: int = 42,
    planner_id: str = "pyperplan",
) -> tuple[PlanningGraph, dict[frozenset, list]]:
    """Build a unified planning graph across multiple episodes.

    This creates a single, comprehensive planning graph by running BFS
    collection across multiple random episodes. Unlike per-episode graphs,
    this gives a consistent atom-based representation that doesn't depend
    on episode-specific node IDs.

    Args:
        system: The TAMP system
        collect_episodes: Number of episodes to collect across
        seed: Random seed for environment resets
        planner_id: Planner to use for BFS graph construction

    Returns:
        Tuple of:
        - total_graph: Unified planning graph with all discovered nodes/edges
        - node_states: Dict mapping frozenset[atoms] -> list of states
    """
    rng = np.random.default_rng(seed)
    total_graph = PlanningGraph()
    node_states: dict[frozenset, list] = {}

    print(f"Building total planning graph from {collect_episodes} episodes...")

    for episode_idx in range(collect_episodes):
        # Reset environment with random seed
        obs, info = system.reset(seed=int(rng.integers(0, 2**31)))
        objects, atoms, goal = system.perceiver.reset(obs, info)

        # Create a temporary approach just for this episode's BFS
        from tamp_improv.approaches.improvisational.policies.base import Policy

        # Create a dummy policy (not used for collection)
        class DummyPolicy(Policy):
            def can_initiate(self, obs, context):
                return False

            def step(self, obs, context):
                raise NotImplementedError

        dummy_policy = MultiRLPolicy(seed=seed)
        temp_approach = ImprovisationalTAMPApproach(
            system=system,
            policy=dummy_policy,
            seed=seed + episode_idx,
            planner_id=planner_id,
        )
        temp_approach.training_mode = True

        # Build planning graph directly (faster - no path-dependent costs)
        temp_approach._goal = goal
        episode_graph = temp_approach._create_planning_graph(objects, atoms)
        temp_approach.planning_graph = episode_graph

        # Collect states using simple BFS (faster - no path dependency)
        episode_states = collect_states_for_all_nodes(
            system, episode_graph, max_attempts=3
        )
        # Convert to multi-state format to match observed_states structure
        episode_states = {k: [v] for k, v in episode_states.items()}

        # Merge nodes into total graph
        for node in episode_graph.nodes:
            # Get or create node in total graph (atom-based, so consistent across episodes)
            total_node = total_graph.get_or_add_node(set(node.atoms))

            # Accumulate states for this atom set
            if node.atoms not in node_states:
                node_states[node.atoms] = []

            # Add states from this episode (check for duplicates)
            if node.id in episode_states:
                for new_state in episode_states[node.id]:
                    # Check if this state is a duplicate
                    is_duplicate = False
                    if hasattr(new_state, "nodes"):
                        for existing_state in node_states[node.atoms]:
                            if hasattr(existing_state, "nodes") and np.array_equal(
                                existing_state.nodes, new_state.nodes
                            ):
                                is_duplicate = True
                                break
                    elif isinstance(new_state, np.ndarray):
                        for existing_state in node_states[node.atoms]:
                            if isinstance(existing_state, np.ndarray) and np.array_equal(
                                existing_state, new_state
                            ):
                                is_duplicate = True
                                break

                    if not is_duplicate:
                        node_states[node.atoms].append(new_state)

        # Merge edges (only add if not already present)
        for edge in episode_graph.edges:
            source_atoms = edge.source.atoms
            target_atoms = edge.target.atoms

            # Get the corresponding nodes in total graph
            source_node = total_graph.node_map[source_atoms]
            target_node = total_graph.node_map[target_atoms]

            # Check if this edge already exists
            edge_exists = any(
                e.source == source_node
                and e.target == target_node
                and e.operator == edge.operator
                for e in total_graph.edges
            )

            if not edge_exists:
                total_graph.add_edge(
                    source=source_node,
                    target=target_node,
                    operator=edge.operator,
                    cost=edge.cost,
                    is_shortcut=edge.is_shortcut,
                )

        if (episode_idx + 1) % 5 == 0 or episode_idx == collect_episodes - 1:
            print(
                f"  Episode {episode_idx + 1}/{collect_episodes}: "
                f"{len(total_graph.nodes)} nodes, {len(total_graph.edges)} edges"
            )

    print(
        f"Total graph complete: {len(total_graph.nodes)} nodes, "
        f"{len(total_graph.edges)} edges, "
        f"{sum(len(states) for states in node_states.values())} total states"
    )

    return total_graph, node_states
