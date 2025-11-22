"""Analysis utilities for improvisational TAMP approaches."""

from typing import TypeVar

import numpy as np
from relational_structs import GroundAtom, GroundOperator

from tamp_improv.approaches.improvisational.graph import PlanningGraph, PlanningGraphEdge
from tamp_improv.approaches.improvisational.policies.base import Policy, PolicyContext
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")


def execute_shortcut_once(
    policy: Policy,
    system: ImprovisationalTAMPSystem,
    start_state: ObsType,
    goal_atoms: set[GroundAtom],
    max_steps: int = 100,
    source_node_id: int | None = None,
    target_node_id: int | None = None,
) -> tuple[bool, int]:
    """Execute a shortcut policy once from start state to goal atoms.

    Args:
        policy: The policy to execute
        system: The TAMP system (provides env and perceiver)
        start_state: The initial state to start from
        goal_atoms: The goal atoms to reach
        max_steps: Maximum number of steps to execute
        source_node_id: Optional source node ID for policy key matching
        target_node_id: Optional target node ID for policy key matching

    Returns:
        Tuple of (success, num_steps)
            - success: Whether the goal atoms were reached
            - num_steps: Number of steps taken
    """
    # Get the base environment
    env = system.env

    # Reset environment to start state
    obs, _ = env.reset_from_state(start_state)

    # Get start atoms from perceiver
    start_atoms = system.perceiver.step(obs)

    # Configure policy with context
    info = {}
    if source_node_id is not None and target_node_id is not None:
        info["source_node_id"] = source_node_id
        info["target_node_id"] = target_node_id

    policy_context = PolicyContext(
        current_atoms=start_atoms,
        goal_atoms=goal_atoms,
        info=info,
    )
    policy.configure_context(policy_context)

    # Execute policy
    num_steps = 0
    success = False

    for _ in range(max_steps):
        # Get action from policy
        if not policy.can_initiate():
            break

        action = policy.get_action(obs)
        if action is None:
            break

        # Step environment
        obs, _, _, _, _ = env.step(action)
        num_steps += 1

        # Check if goal atoms are reached
        current_atoms = system.perceiver.step(obs)
        if goal_atoms == frozenset(current_atoms):
            success = True
            break

    return success, num_steps


def execute_edge_once(
    system: ImprovisationalTAMPSystem,
    edge: PlanningGraphEdge,
    start_state: ObsType,
    max_steps: int = 100,
) -> tuple[bool, int]:
    """Execute an edge (operator) once from a start state.

    Args:
        system: The TAMP system (provides env and perceiver)
        edge: The edge to execute (must have an operator)
        start_state: The initial state to start from
        max_steps: Maximum number of steps to execute

    Returns:
        Tuple of (success, num_steps)
            - success: Whether the target atoms were reached
            - num_steps: Number of steps taken (max_steps if failed)
    """
    if edge.operator is None or edge.is_shortcut:
        # Can't execute edge without operator, or if it's a shortcut
        return False, max_steps

    env = system.env

    # Reset environment to start state
    obs, info = env.reset_from_state(start_state)

    # Get target atoms from edge
    target_atoms = edge.target.atoms

    # Get the skill that can execute this operator
    skills = [s for s in system.skills if s.can_execute(edge.operator)]
    if not skills:
        return False, max_steps
    skill = skills[0]
    skill.reset(edge.operator)

    # Execute the operator using the skill
    num_steps = 0
    success = False

    for _ in range(max_steps):
        # Get action from skill
        action = skill.get_action(obs)
        if action is None:
            break

        # Step environment
        obs, _, terminated, truncated, info = env.step(action)
        num_steps += 1

        # Check if we reached target atoms
        current_atoms = system.perceiver.step(obs)
        if frozenset(current_atoms) == target_atoms:
            success = True
            break

        # Check if episode ended
        if terminated or truncated:
            break

    return success, num_steps


def compute_average_edge_cost(
    system: ImprovisationalTAMPSystem,
    edge: PlanningGraphEdge,
    start_states: list[ObsType],
    num_samples: int = 5,
    max_steps: int = 100,
) -> float:
    """Compute average cost of an edge by executing it from multiple start states.

    Args:
        system: The TAMP system
        edge: The edge to compute cost for
        start_states: List of start states to sample from
        num_samples: Number of samples to use (or all if fewer available)
        max_steps: Maximum steps per execution attempt

    Returns:
        Average cost (num_steps) across successful executions, or inf if all failed
    """
    if not start_states:
        return float("inf")

    # Sample states (or use all if fewer than num_samples)
    if len(start_states) <= num_samples:
        sampled_states = start_states
    else:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(start_states), size=num_samples, replace=False)
        sampled_states = [start_states[i] for i in indices]

    # Execute edge from each sampled state
    costs = []
    for start_state in sampled_states:
        success, num_steps = execute_edge_once(system, edge, start_state, max_steps)
        if success:
            costs.append(num_steps)
        else:
            costs.append(max_steps)  # Penalty for failure

    # Return average cost
    if costs:
        return float(np.mean(costs))
    return float("inf")


def compute_all_edge_costs(
    system: ImprovisationalTAMPSystem,
    graph: PlanningGraph,
    node_states: dict[frozenset, list],
    num_samples: int = 5,
    max_steps: int = 100,
) -> None:
    """Compute costs for all edges in a planning graph.

    This updates the edge costs in-place by executing each edge multiple times
    from different start states sampled from node_states.

    Args:
        system: The TAMP system
        graph: Planning graph with edges to compute costs for
        node_states: Dict mapping frozenset[atoms] -> list of states
        num_samples: Number of start states to sample per edge
        max_steps: Maximum steps allowed for edge execution
    """
    print(f"\nComputing edge costs using {num_samples} samples per edge...")

    for edge_idx, edge in enumerate(graph.edges):
        # Skip shortcut edges (they don't have operators)
        if edge.is_shortcut or edge.operator is None:
            edge.cost = 1.0  # Default cost for shortcuts
            continue

        # Get states for the source node
        source_atoms = edge.source.atoms
        if source_atoms not in node_states or not node_states[source_atoms]:
            edge.cost = float("inf")  # No states available
            continue

        # Compute average cost
        edge.cost = compute_average_edge_cost(
            system, edge, node_states[source_atoms], num_samples, max_steps
        )

        # Print progress every 10 edges
        if (edge_idx + 1) % 10 == 0:
            print(f"  Processed {edge_idx + 1}/{len(graph.edges)} edges")

    print(f"Edge cost computation complete!")
