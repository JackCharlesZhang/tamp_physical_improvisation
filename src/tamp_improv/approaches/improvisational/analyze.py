"""Analysis utilities for improvisational TAMP approaches."""

from typing import TypeVar

from relational_structs import GroundAtom

from tamp_improv.approaches.improvisational.policies.base import Policy, PolicyContext
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")


def execute_shortcut_once(
    policy: Policy,
    system: ImprovisationalTAMPSystem,
    start_state: ObsType,
    goal_atoms: set[GroundAtom],
    max_steps: int = 100,
) -> tuple[bool, int]:
    """Execute a shortcut policy once from start state to goal atoms.

    Args:
        policy: The policy to execute
        system: The TAMP system (provides env and perceiver)
        start_state: The initial state to start from
        goal_atoms: The goal atoms to reach
        max_steps: Maximum number of steps to execute

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
    policy_context = PolicyContext(
        current_atoms=start_atoms,
        goal_atoms=goal_atoms,
        info={},
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
        if goal_atoms.issubset(current_atoms):
            success = True
            break

    return success, num_steps
