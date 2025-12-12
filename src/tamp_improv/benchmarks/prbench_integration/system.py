"""PRBench SLAP system integration."""

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

# Monkey-patch Tobject into tomsgeoms2d if it doesn't exist
# (prbench imports it but dyn_obstruction2d doesn't actually use it)
# MUST happen BEFORE importing prbench!
import tomsgeoms2d.structs

if not hasattr(tomsgeoms2d.structs, "Tobject"):
    # Create a dummy Tobject class to satisfy prbench imports
    from tomsgeoms2d.structs import Lobject

    tomsgeoms2d.structs.Tobject = Lobject  # Use Lobject as a stand-in

from prbench.envs.dynamic2d.dyn_obstruction2d import DynObstruction2DEnv
from prbench_bilevel_planning.env_models import create_bilevel_planning_models
from relational_structs import PDDLDomain

from tamp_improv.benchmarks.base import (
    BaseTAMPSystem,
    ImprovisationalTAMPSystem,
    PlanningComponents,
)
from tamp_improv.benchmarks.prbench_integration.perceiver import PRBenchPerceiver
from tamp_improv.benchmarks.prbench_integration.skills import PRBenchSkill
from tamp_improv.benchmarks.prbench_integration.utils import (
    PRBenchPredicateContainer,
)
from tamp_improv.benchmarks.wrappers import ImprovWrapper


class BasePRBenchSLAPSystem(BaseTAMPSystem[Any, NDArray[np.float32]]):
    """Base TAMP system using PRBench components.

    This system combines:
    - PRBench's clean perception (state_abstractor, goal_deriver)
    - PRBench's 4-5 simple predicates (no counting!)
    - PRBench's generic operators (PickTgt, PickObstruction, PlaceTgt, PlaceObstruction)
    - PRBench's BiRRT motion planning
    - SLAP's shortcut learning pipeline (when used with ImprovisationalTAMPSystem)
    """

    def __init__(
        self,
        planning_components: PlanningComponents[Any],
        num_obstructions: int = 2,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize PRBench SLAP system."""
        self._num_obstructions = num_obstructions
        self._render_mode = render_mode
        super().__init__(
            planning_components, name="PRBenchSLAPSystem", seed=seed, render_mode=render_mode
        )

    def _create_env(self) -> gym.Env:
        """Create PRBench environment.

        Uses DynObstruction2DEnv which wraps ObjectCentricDynObstruction2DEnv
        with fixed object ordering and returns ObjectCentricBoxSpace observations.
        """
        return DynObstruction2DEnv(
            num_obstructions=self._num_obstructions, render_mode=self._render_mode
        )

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "prbench-dyn-obstruction2d-domain"

    def get_domain(self) -> PDDLDomain:
        """Get PDDL domain."""
        return PDDLDomain(
            self._get_domain_name(),
            self.components.operators,
            self.components.predicate_container.as_set(),
            self.components.types,
        )

    @classmethod
    def _create_planning_components(
        cls, num_obstructions: int = 2
    ) -> PlanningComponents[Any]:
        """Create planning components from PRBench models.

        This is where we use PRBench's create_bilevel_planning_models to get:
        - Types (robot, target_block, obstruction, target_surface)
        - Predicates (HandEmpty, HoldingTgt, HoldingObstruction, OnTgt, AboveTgt)
        - Operators (PickTgt, PickObstruction, PlaceTgt, PlaceObstruction, etc.)
        - Controllers (BiRRT motion planning)
        """
        # Create dummy env to get spaces for model creation
        env = DynObstruction2DEnv(num_obstructions=num_obstructions)

        # Create PRBench's bilevel planning models
        sesame_models = create_bilevel_planning_models(
            "dynobstruction2d",
            env.observation_space,
            env.action_space,
            num_obstructions=num_obstructions,
        )

        # Wrap PRBench components with SLAP-compatible adapters
        predicate_container = PRBenchPredicateContainer(sesame_models.predicates)
        perceiver = PRBenchPerceiver(
            observation_to_state_fn=sesame_models.observation_to_state,
            state_abstractor_fn=sesame_models.state_abstractor,
            goal_deriver_fn=sesame_models.goal_deriver,
        )

        # Wrap PRBench skills with SLAP-compatible skill interface
        skills = {
            PRBenchSkill(
                lifted_skill=lifted_skill,
                observation_to_state_fn=sesame_models.observation_to_state,
            )
            for lifted_skill in sesame_models.skills
        }

        # Add all parent types to satisfy PDDL type hierarchy
        # PRBench's types set doesn't include parents like 'dynamic2d', 'kin_rectangle'
        all_types = set(sesame_models.types)
        for t in sesame_models.types:
            current = t
            while current.parent is not None:
                all_types.add(current.parent)
                current = current.parent

        return PlanningComponents(
            types=all_types,
            predicate_container=predicate_container,
            operators=sesame_models.operators,
            skills=skills,
            perceiver=perceiver,
        )

    @classmethod
    def create_default(
        cls,
        num_obstructions: int = 2,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> "BasePRBenchSLAPSystem":
        """Factory method for creating system with PRBench components."""
        planning_components = cls._create_planning_components(num_obstructions)
        return cls(
            planning_components,
            num_obstructions=num_obstructions,
            seed=seed,
            render_mode=render_mode,
        )


class PRBenchSLAPSystem(
    ImprovisationalTAMPSystem[Any, NDArray[np.float32]], BasePRBenchSLAPSystem
):
    """TAMP system using PRBench components with improvisational policy learning.

    This is the main system that combines:
    - PRBench's ground truth perception and motion planning
    - SLAP's shortcut learning pipeline
    """

    def __init__(
        self,
        planning_components: PlanningComponents[Any],
        num_obstructions: int = 2,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize PRBench SLAP system with improvisational learning."""
        self._num_obstructions = num_obstructions
        self._render_mode = render_mode
        super().__init__(planning_components, seed=seed, render_mode=render_mode)

    def _create_wrapped_env(self, components: PlanningComponents[Any]) -> gym.Env:
        """Create wrapped environment for training.

        Uses ImprovWrapper to add rewards for:
        - Goal achievement (+10.0)
        - Step penalty (-0.5)
        """
        return ImprovWrapper(
            base_env=self.env,
            perceiver=components.perceiver,
            step_penalty=-0.5,
            achievement_bonus=10.0,
        )

    @classmethod
    def create_default(
        cls,
        num_obstructions: int = 2,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> "PRBenchSLAPSystem":
        """Factory method for creating improvisational system with PRBench components."""
        planning_components = cls._create_planning_components(num_obstructions)
        return cls(
            planning_components,
            num_obstructions=num_obstructions,
            seed=seed,
            render_mode=render_mode,
        )
