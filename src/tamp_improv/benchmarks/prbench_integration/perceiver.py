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
        >>> from prbench_bilevel_planning.env_models import create_bilevel_planning_models
        >>> sesame_models = create_bilevel_planning_models(
        ...     "dynobstruction2d", obs_space, action_space, num_obstructions=2
        ... )
        >>> perceiver = PRBenchPerceiver(
        ...     observation_to_state_fn=sesame_models.observation_to_state,
        ...     state_abstractor_fn=sesame_models.state_abstractor,
        ...     goal_deriver_fn=sesame_models.goal_deriver
        ... )
        >>> obs = env.reset()
        >>> objects, atoms, goal = perceiver.reset(obs, {})
    """

    def __init__(
        self,
        observation_to_state_fn: Callable[[Any], ObjectCentricState],
        state_abstractor_fn: Callable[[ObjectCentricState], Any],
        goal_deriver_fn: Callable[[ObjectCentricState], Any],
    ) -> None:
        """Initialize perceiver with PRBench's abstraction functions.

        Args:
            observation_to_state_fn: PRBench's observation_to_state function that converts
                vectorized observation → ObjectCentricState
            state_abstractor_fn: PRBench's state_abstractor function that converts
                ObjectCentricState → RelationalAbstractState
            goal_deriver_fn: PRBench's goal_deriver function that extracts goal
                atoms from ObjectCentricState
        """
        self.observation_to_state = observation_to_state_fn
        self.state_abstractor = state_abstractor_fn
        self.goal_deriver = goal_deriver_fn

    def reset(
        self,
        obs: Any,
        info: dict[str, Any],
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Reset perceiver and return objects, initial atoms, and goal atoms.

        Args:
            obs: Vectorized observation from PRBench environment
            info: Environment info dict (unused, for compatibility)

        Returns:
            objects: Set of objects in the scene
            atoms: Set of ground atoms describing initial state
            goal: Set of ground atoms describing goal
        """
        # Convert vectorized observation to ObjectCentricState
        state = self.observation_to_state(obs)

        # Use PRBench's state_abstractor to get abstract state
        abstract_state = self.state_abstractor(state)

        # Get goal from goal_deriver
        goal = self.goal_deriver(state)

        # Extract objects and atoms from abstract state
        objects = abstract_state.objects
        atoms = abstract_state.atoms

        # Extract goal atoms from goal
        goal_atoms = goal.atoms

        return objects, atoms, goal_atoms

    def step(self, obs: Any) -> set[GroundAtom]:
        """Get current ground atoms from observation.

        Args:
            obs: Vectorized observation from PRBench environment

        Returns:
            atoms: Set of ground atoms describing current state
        """
        # Convert vectorized observation to ObjectCentricState
        state = self.observation_to_state(obs)
        abstract_state = self.state_abstractor(state)
        return abstract_state.atoms
