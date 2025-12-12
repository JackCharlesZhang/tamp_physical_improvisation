"""PRBench skill adapter for SLAP."""

from typing import Any, Callable, Sequence

import numpy as np
from bilevel_planning.structs import GroundParameterizedController, LiftedSkill
from relational_structs import GroundOperator, LiftedOperator, Object
from relational_structs.object_centric_state import ObjectCentricState
from task_then_motion_planning.structs import LiftedOperatorSkill


class PRBenchSkill(LiftedOperatorSkill[Any, np.ndarray]):
    """Adapts PRBench's ParameterizedController to SLAP's LiftedOperatorSkill.

    PRBench controllers are open-loop planners:
    1. reset(state, params) - samples parameters and generates full action plan
    2. step() - returns next action from plan
    3. observe(state) - updates internal state
    4. terminated() - checks if plan is complete

    SLAP expects reactive skills:
    1. reset(ground_operator) - prepares skill for execution
    2. get_action(obs) - returns action given current observation

    This adapter bridges the gap by:
    - Converting SLAP's observations to PRBench's ObjectCentricState
    - Calling observe() to update controller state
    - Returning step() actions to SLAP

    Example:
        >>> from prbench_bilevel_planning.env_models import create_bilevel_planning_models
        >>> sesame_models = create_bilevel_planning_models(
        ...     "dynobstruction2d", obs_space, action_space, num_obstructions=2
        ... )
        >>> # Create skill from PRBench's LiftedSkill
        >>> prbench_lifted_skill = list(sesame_models.skills)[0]
        >>> skill = PRBenchSkill(
        ...     lifted_skill=prbench_lifted_skill,
        ...     observation_to_state_fn=sesame_models.observation_to_state,
        ... )
    """

    def __init__(
        self,
        lifted_skill: LiftedSkill,
        observation_to_state_fn: Callable[[Any], ObjectCentricState],
    ) -> None:
        """Initialize skill with PRBench's LiftedSkill.

        Args:
            lifted_skill: PRBench's LiftedSkill containing operator and controller
            observation_to_state_fn: Function to convert obs → ObjectCentricState
        """
        super().__init__()
        self._lifted_operator = lifted_skill.operator
        self._lifted_controller = lifted_skill.controller
        self._observation_to_state = observation_to_state_fn
        self._ground_controller: GroundParameterizedController | None = None
        self._rng = np.random.default_rng()

    def _get_lifted_operator(self) -> LiftedOperator:
        """Return the lifted operator for this skill."""
        return self._lifted_operator

    def _get_action_given_objects(
        self, objects: Sequence[Object], obs: Any
    ) -> np.ndarray:
        """Get action given objects and observation.

        This is called by SLAP at each timestep. We:
        1. Convert observation to ObjectCentricState
        2. Update controller with observe()
        3. Return next action from controller's plan

        Args:
            objects: Ground operator parameters (objects)
            obs: Vectorized observation from environment

        Returns:
            Action array to execute
        """
        # Convert observation to state
        state = self._observation_to_state(obs)

        # Update controller's internal state
        if self._ground_controller is not None:
            self._ground_controller.observe(state)

            # Check if controller has terminated
            if self._ground_controller.terminated():
                # Return a zero action or signal completion
                # SLAP will check operator effects to determine success
                return np.zeros(5, dtype=np.float32)

            # Get next action from controller's plan
            return self._ground_controller.step()

        # Should not reach here if reset() was called
        raise RuntimeError("Controller not initialized. Call reset() first.")

    def reset(self, ground_operator: GroundOperator) -> None:
        """Reset skill for new execution.

        This is called by SLAP before executing an operator. We:
        1. Create ground controller from lifted controller
        2. Convert observation to state (will be done in first get_action call)
        3. Sample parameters and reset controller

        Args:
            ground_operator: The ground operator to execute
        """
        super().reset(ground_operator)

        # Ground the lifted controller with the operator's parameters
        objects = ground_operator.parameters
        self._ground_controller = self._lifted_controller.ground(objects)

        # Note: We can't reset the controller yet because we don't have
        # the initial state. That will happen in the first get_action() call.
        self._controller_needs_reset = True

    def get_action(self, obs: Any) -> np.ndarray:
        """Get action given current observation.

        This overrides the base class to handle controller initialization
        on the first call after reset().

        Args:
            obs: Vectorized observation from environment

        Returns:
            Action array to execute
        """
        # If this is the first call after reset, initialize the controller
        if hasattr(self, "_controller_needs_reset") and self._controller_needs_reset:
            state = self._observation_to_state(obs)

            # Sample parameters for this execution
            params = self._ground_controller.sample_parameters(state, self._rng)

            # Reset controller with state and parameters
            self._ground_controller.reset(state, params)

            self._controller_needs_reset = False

        # Use base class logic to call _get_action_given_objects
        return super().get_action(obs)
