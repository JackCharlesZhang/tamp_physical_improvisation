"""PRBench perceiver adapter for SLAP."""

from typing import Any, Callable

from relational_structs import GroundAtom, Object
from relational_structs.object_centric_state import ObjectCentricState
from task_then_motion_planning.structs import Perceiver


class PRBenchPerceiver(Perceiver[ObjectCentricState]):
    """Adapts PRBench's state_abstractor to SLAP's Perceiver interface.

    This eliminates the need for:
    - Manual observation parsing (no more obs[0], obs[21] > 0.5)
    - Custom geometry checks with thresholds
    - Counting predicates (TwoObstructionsBlocking, etc.)

    Instead, we directly use PRBench's state_abstractor which:
    - Gets objects directly from ObjectCentricState
    - Uses PRBench's geometric utilities (get_suctioned_objects, is_on)
    - Returns clean, simple predicates

    Example:
        >>> from prbench_bilevel_planning.env_models.dynamic2d.dynobstruction2d import (
        ...     create_bilevel_planning_models
        ... )
        >>> sesame_models = create_bilevel_planning_models(obs_space, action_space, 2)
        >>> perceiver = PRBenchPerceiver(
        ...     state_abstractor_fn=sesame_models.state_abstractor,
        ...     goal_deriver_fn=sesame_models.goal_deriver
        ... )
        >>> obs = env.reset()
        >>> objects, atoms, goal = perceiver.reset(obs, {})
    """

    def __init__(
        self,
        state_abstractor_fn: Callable[[ObjectCentricState], Any],
        goal_deriver_fn: Callable[[ObjectCentricState], Any],
    ) -> None:
        """Initialize perceiver with PRBench's abstraction functions.

        Args:
            state_abstractor_fn: PRBench's state_abstractor function that converts
                ObjectCentricState → RelationalAbstractState
            goal_deriver_fn: PRBench's goal_deriver function that extracts goal
                atoms from ObjectCentricState
        """
        self.state_abstractor = state_abstractor_fn
        self.goal_deriver = goal_deriver_fn

    def reset(
        self,
        obs: ObjectCentricState,
        info: dict[str, Any],
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Reset perceiver and return objects, initial atoms, and goal atoms.

        Args:
            obs: ObjectCentricState from PRBench environment
            info: Environment info dict (unused, for compatibility)

        Returns:
            objects: Set of objects in the scene
            atoms: Set of ground atoms describing initial state
            goal: Set of ground atoms describing goal
        """
        # Use PRBench's state_abstractor to get abstract state
        abstract_state = self.state_abstractor(obs)

        # Get goal from goal_deriver
        goal = self.goal_deriver(obs)

        # Extract objects and atoms from abstract state
        objects = abstract_state.objects
        atoms = abstract_state.atoms

        # Extract goal atoms from goal
        goal_atoms = goal.atoms

        return objects, atoms, goal_atoms

    def step(self, obs: ObjectCentricState) -> set[GroundAtom]:
        """Get current ground atoms from observation.

        Args:
            obs: ObjectCentricState from PRBench environment

        Returns:
            atoms: Set of ground atoms describing current state
        """
        abstract_state = self.state_abstractor(obs)
        return abstract_state.atoms
