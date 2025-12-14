"""ClutteredStorage2D environment graph-based implementation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from prbench.envs.geom2d.clutteredstorage2d import (
    ObjectCentricClutteredStorage2DEnv,
    ShelfType,
    TargetBlockType,
)
from prbench.envs.geom2d.object_types import CRVRobotType
from prbench.envs.geom2d.utils import (
    get_suctioned_objects,
    is_inside_shelf,
)
from relational_structs import (
    GroundAtom,
    LiftedOperator,
    Object,
    PDDLDomain,
    Predicate,
    Type,
    Variable,
)
from relational_structs import ObjectCentricState
from task_then_motion_planning.structs import Perceiver

from tamp_improv.benchmarks.base import (
    BaseTAMPSystem,
    ImprovisationalTAMPSystem,
    PlanningComponents,
)
from tamp_improv.benchmarks.clutteredstorage_env_wrapper import (
    ClutteredStorage2DEnvWrapper,
)
from tamp_improv.benchmarks.wrappers import ImprovWrapper


class ClutteredStorage2DTypes:
    """Container for cluttered storage 2D types."""

    def __init__(self) -> None:
        """Initialize types."""
        self.robot = CRVRobotType
        self.block = TargetBlockType
        self.shelf = ShelfType

    def as_set(self) -> set[Type]:
        """Convert to set of types, including all parent types."""
        types = {self.robot, self.block, self.shelf}
        # Add all parent types to satisfy type hierarchy requirements
        all_types = set(types)
        for t in types:
            current = t
            while current.parent is not None:
                all_types.add(current.parent)
                current = current.parent
        return all_types


class ClutteredStorage2DPredicates:
    """Container for cluttered storage 2D predicates."""

    def __init__(self, types: ClutteredStorage2DTypes) -> None:
        """Initialize predicates."""
        self.holding = Predicate("Holding", [types.robot, types.block])
        self.hand_empty = Predicate("HandEmpty", [types.robot])
        self.not_on_shelf = Predicate("NotOnShelf", [types.block, types.shelf])
        self.on_shelf = Predicate("OnShelf", [types.block, types.shelf])

    def __getitem__(self, key: str) -> Predicate:
        """Get predicate by name."""
        return next(p for p in self.as_set() if p.name == key)

    def as_set(self) -> set[Predicate]:
        """Convert to set of predicates."""
        return {
            self.holding,
            self.hand_empty,
            self.not_on_shelf,
            self.on_shelf,
        }


class GraphClutteredStorage2DPerceiver(Perceiver[NDArray[np.float32]]):
    """Perceiver for cluttered storage 2D environment."""

    def __init__(self, types: ClutteredStorage2DTypes, n_blocks: int = 1) -> None:
        """Initialize with required types."""
        self._robot = Object("robot", types.robot)
        self._blocks = [Object(f"block{i}", types.block) for i in range(n_blocks)]
        self._shelf = Object("shelf", types.shelf)
        self._predicates: ClutteredStorage2DPredicates | None = None
        self._types = types
        self.n_blocks = n_blocks
        # Will be set during initialization with environment info
        self._observation_objects: list[Object] | None = None
        self._type_features: dict | None = None

    def initialize(
        self,
        predicates: ClutteredStorage2DPredicates,
        observation_objects: list[Object] | None = None,
        type_features: dict | None = None,
    ) -> None:
        """Initialize predicates after environment creation."""
        self._predicates = predicates
        self._observation_objects = observation_objects
        self._type_features = type_features

    @property
    def predicates(self) -> ClutteredStorage2DPredicates:
        """Get predicates, ensuring they're initialized."""
        if self._predicates is None:
            raise RuntimeError("Predicates not initialized. Call initialize() first.")
        return self._predicates

    def reset(
        self,
        obs: NDArray[np.float32],
        _info: dict[str, Any],
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Reset perceiver with observation and info."""
        # Convert vector observation to ObjectCentricState
        state = self._vec_to_state(obs)

        objects = {
            self._robot,
            self._shelf,
        }
        objects.update(self._blocks)
        atoms = self._get_atoms(state)

        # Goal: all blocks on shelf and hand empty
        goal = set()
        for block in self._blocks:
            goal.add(self.predicates["OnShelf"]([block, self._shelf]))
        goal.add(self.predicates["HandEmpty"]([self._robot]))

        return objects, atoms, goal

    def step(self, obs: NDArray[np.float32]) -> set[GroundAtom]:
        """Step perceiver with observation."""
        # Convert vector observation to ObjectCentricState
        state = self._vec_to_state(obs)
        return self._get_atoms(state)

    def _vec_to_state(self, vec: NDArray[np.float32]) -> ObjectCentricState:
        """Convert vector observation to ObjectCentricState."""
        if self._observation_objects is None or self._type_features is None:
            raise RuntimeError("Perceiver not fully initialized with observation objects and type features")

        return ObjectCentricState.from_vec(
            vec,
            constant_objects=self._observation_objects,
            type_features=self._type_features
        )

    def _get_atoms(self, obs: ObjectCentricState) -> set[GroundAtom]:
        """Convert observation to ground atoms."""
        atoms = set()

        # Get robot from observation
        robot_obj = obs.get_objects(self._types.robot)[0]
        target_blocks = obs.get_objects(self._types.block)
        shelf_obj = obs.get_objects(self._types.shelf)[0]

        # Add holding / hand empty atoms
        suctioned_objs = {o for o, _ in get_suctioned_objects(obs, robot_obj)}

        for i, block in enumerate(self._blocks):
            # Find corresponding block in observation
            target_block = next((b for b in target_blocks if b.name == block.name), None)
            if target_block and target_block in suctioned_objs:
                atoms.add(GroundAtom(self.predicates.holding, [self._robot, block]))

        if not suctioned_objs:
            atoms.add(GroundAtom(self.predicates.hand_empty, [self._robot]))

        # Add on_shelf / not_on_shelf atoms
        for i, block in enumerate(self._blocks):
            target_block = next((b for b in target_blocks if b.name == block.name), None)

            if target_block:
                if is_inside_shelf(obs, target_block, shelf_obj, {}):
                    atoms.add(GroundAtom(self.predicates.on_shelf, [block, self._shelf]))
                else:
                    atoms.add(GroundAtom(self.predicates.not_on_shelf, [block, self._shelf]))

        return atoms

    def encode_atoms_to_vector(self, atoms: set[GroundAtom]) -> NDArray[np.float32]:
        """Encode a set of atoms as a binary vector (one-hot style).

        This creates a fixed-length binary vector where each dimension corresponds
        to a possible atom. Used for goal-conditioned distance heuristic learning.

        Args:
            atoms: Set of ground atoms to encode

        Returns:
            Binary vector representation of the atoms
        """
        if not hasattr(self, '_atom_vocabulary'):
            self._build_atom_vocabulary()

        if len(self._atom_vocabulary) == 0:
            raise RuntimeError(
                f"Atom vocabulary is empty! Predicates initialized: {self._predicates is not None}, "
                f"n_blocks: {self.n_blocks}"
            )

        # Create binary vector
        vector = np.zeros(len(self._atom_vocabulary), dtype=np.float32)
        for atom in atoms:
            atom_str = str(atom)
            if atom_str in self._atom_to_idx:
                vector[self._atom_to_idx[atom_str]] = 1.0

        return vector

    def _build_atom_vocabulary(self) -> None:
        """Build vocabulary of all possible atoms for this domain.

        Creates a sorted list of all possible ground atoms and a mapping
        from atom strings to indices for efficient encoding.
        """
        if self._predicates is None:
            raise RuntimeError("Predicates not initialized. Call initialize() first.")

        all_atoms = []

        # HandEmpty(robot) - only applies to robot
        all_atoms.append(str(self._predicates.hand_empty([self._robot])))

        # Holding(robot, block) - robot holds each block
        for block in self._blocks:
            all_atoms.append(str(self._predicates.holding([self._robot, block])))

        # OnShelf(block, shelf) and NotOnShelf(block, shelf) - each block on/not on shelf
        for block in self._blocks:
            all_atoms.append(str(self._predicates.on_shelf([block, self._shelf])))
            all_atoms.append(str(self._predicates.not_on_shelf([block, self._shelf])))

        # Create sorted vocabulary and index mapping
        self._atom_vocabulary = sorted(all_atoms)
        self._atom_to_idx = {atom: idx for idx, atom in enumerate(self._atom_vocabulary)}


class BaseGraphClutteredStorage2DTAMPSystem(
    BaseTAMPSystem[NDArray[np.float32], NDArray[np.float32]]
):
    """Base TAMP system for cluttered storage 2D graph-based environment."""

    def __init__(
        self,
        planning_components: PlanningComponents[NDArray[np.float32]],
        n_blocks: int = 1,
        seed: int | None = None,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize graph-based ClutteredStorage2D TAMP system."""
        self._render_mode = render_mode
        # Only set n_blocks if not already set (to support cooperative inheritance)
        if not hasattr(self, 'n_blocks'):
            self.n_blocks = n_blocks
        super().__init__(
            planning_components,
            name="GraphClutteredStorage2DTAMPSystem",
            seed=seed,
            render_mode=render_mode,
            **kwargs,
        )

    def _create_env(self) -> gym.Env:
        """Create base environment."""
        base_env = ObjectCentricClutteredStorage2DEnv(
            num_blocks=self.n_blocks, render_mode=self._render_mode
        )

        base_env._num_init_shelf_blocks = 0
        base_env._num_init_outside_blocks = self.n_blocks
        # Wrap with ClutteredStorage2DEnvWrapper to add reset_from_state
        return ClutteredStorage2DEnvWrapper(base_env)

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "graph-clutteredstorage2d-domain"

    def get_domain(self) -> PDDLDomain:
        """Get domain."""
        return PDDLDomain(
            self._get_domain_name(),
            self.components.operators,
            self.components.predicate_container.as_set(),
            self.components.types,
        )

    @classmethod
    def _create_planning_components(
        cls, n_blocks: int = 1
    ) -> PlanningComponents[NDArray[np.float32]]:
        """Create planning components for graph-based ClutteredStorage2D system."""
        types_container = ClutteredStorage2DTypes()
        types_set = types_container.as_set()

        predicates = ClutteredStorage2DPredicates(types_container)

        perceiver = GraphClutteredStorage2DPerceiver(types_container, n_blocks)
        perceiver.initialize(predicates)

        robot = Variable("?robot", types_container.robot)
        block = Variable("?block", types_container.block)
        shelf = Variable("?shelf", types_container.shelf)

        operators = {
            LiftedOperator(
                "PickBlockNotOnShelf",
                [robot, block, shelf],
                preconditions={
                    predicates["HandEmpty"]([robot]),
                    predicates["NotOnShelf"]([block, shelf]),
                },
                add_effects={
                    predicates["Holding"]([robot, block]),
                },
                delete_effects={
                    predicates["HandEmpty"]([robot]),
                },
            ),
            LiftedOperator(
                "PickBlockOnShelf",
                [robot, block, shelf],
                preconditions={
                    predicates["HandEmpty"]([robot]),
                    predicates["OnShelf"]([block, shelf]),
                },
                add_effects={
                    predicates["Holding"]([robot, block]),
                },
                delete_effects={
                    predicates["HandEmpty"]([robot]),
                },
            ),
            LiftedOperator(
                "PlaceBlockNotOnShelf",
                [robot, block, shelf],
                preconditions={
                    predicates["Holding"]([robot, block]),
                    predicates["OnShelf"]([block, shelf]),
                },
                add_effects={
                    predicates["HandEmpty"]([robot]),
                    predicates["NotOnShelf"]([block, shelf]),
                },
                delete_effects={
                    predicates["Holding"]([robot, block]),
                    predicates["OnShelf"]([block, shelf]),
                },
            ),
            LiftedOperator(
                "PlaceBlockOnShelf",
                [robot, block, shelf],
                preconditions={
                    predicates["Holding"]([robot, block]),
                    predicates["NotOnShelf"]([block, shelf]),
                },
                add_effects={
                    predicates["HandEmpty"]([robot]),
                    predicates["OnShelf"]([block, shelf]),
                },
                delete_effects={
                    predicates["Holding"]([robot, block]),
                    predicates["NotOnShelf"]([block, shelf]),
                },
            ),
        }

        return PlanningComponents(
            types=types_set,
            predicate_container=predicates,
            operators=operators,
            skills=set(),
            perceiver=perceiver,
        )

    @classmethod
    def create_default(
        cls,
        n_blocks: int = 1,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> BaseGraphClutteredStorage2DTAMPSystem:
        """Factory method for creating system with default components."""
        from tamp_improv.benchmarks.clutteredstorage_skill import (
            GraphPickBlockNotOnShelfSkill,
            GraphPickBlockOnShelfSkill,
            GraphPlaceBlockNotOnShelfSkill,
            GraphPlaceBlockOnShelfSkill,
        )

        planning_components = cls._create_planning_components(n_blocks=n_blocks)
        system = cls(
            planning_components,
            n_blocks=n_blocks,
            seed=seed,
            render_mode=render_mode,
        )

        # Create environment to get action space and initial constant state
        env = system.env
        assert isinstance(env, ClutteredStorage2DEnvWrapper)

        # Fully initialize perceiver with observation objects and type features
        system.perceiver.initialize(
            predicates=planning_components.predicate_container,
            observation_objects=env.observation_objects,
            type_features=env.unwrapped_env.type_features
        )

        skills = {
            GraphPickBlockNotOnShelfSkill(
                system.components, env.unwrapped_env.action_space, env.unwrapped_env.initial_constant_state
            ),
            GraphPickBlockOnShelfSkill(
                system.components, env.unwrapped_env.action_space, env.unwrapped_env.initial_constant_state
            ),
            GraphPlaceBlockNotOnShelfSkill(
                system.components, env.unwrapped_env.action_space, env.unwrapped_env.initial_constant_state
            ),
            GraphPlaceBlockOnShelfSkill(
                system.components, env.unwrapped_env.action_space, env.unwrapped_env.initial_constant_state
            ),
        }

        # Initialize all skills with observation info for vec to state conversion
        for skill in skills:
            skill.set_observation_info(env.observation_objects, env.unwrapped_env.type_features)

        system.components.skills.update(skills)
        return system


class GraphClutteredStorage2DTAMPSystem(
    ImprovisationalTAMPSystem[NDArray[np.float32], NDArray[np.float32]],
    BaseGraphClutteredStorage2DTAMPSystem,
):
    """TAMP system for cluttered storage 2D graph-based environment with
    improvisational policy learning enabled."""

    def __init__(
        self,
        planning_components: PlanningComponents[NDArray[np.float32]],
        n_blocks: int = 1,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize graph-based ClutteredStorage2D TAMP system."""
        # Set attributes BEFORE calling super().__init__()
        # so they're available when _create_env() is called
        self.n_blocks = n_blocks
        self._render_mode = render_mode

        # Use cooperative multiple inheritance - let Python's MRO handle it
        # ImprovisationalTAMPSystem.__init__ will be called first (per MRO),
        # which will call its super().__init__() leading to
        # BaseGraphClutteredStorage2DTAMPSystem.__init__, then BaseTAMPSystem.__init__
        # This ensures _create_env() is only called ONCE
        super().__init__(
            planning_components,
            seed=seed,
            render_mode=render_mode,
        )

    def _create_wrapped_env(
        self, components: PlanningComponents[NDArray[np.float32]]
    ) -> gym.Env:
        """Create wrapped environment for training."""
        import ipdb
        
        return ImprovWrapper(
            base_env=self.env,
            perceiver=components.perceiver,
            step_penalty=-0.5,
            achievement_bonus=10.0,
        )

    @classmethod
    def create_default(
        cls,
        n_blocks: int = 1,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> GraphClutteredStorage2DTAMPSystem:
        """Factory method for creating improvisational system with default
        components."""
        from tamp_improv.benchmarks.clutteredstorage_skill import (
            GraphPickBlockNotOnShelfSkill,
            GraphPickBlockOnShelfSkill,
            GraphPlaceBlockNotOnShelfSkill,
            GraphPlaceBlockOnShelfSkill,
        )

        planning_components = cls._create_planning_components(n_blocks=n_blocks)
        system = GraphClutteredStorage2DTAMPSystem(
            planning_components,
            n_blocks=n_blocks,
            seed=seed,
            render_mode=render_mode,
        )

        # Create environment to get action space and initial constant state
        env = system.env
        assert isinstance(env, ClutteredStorage2DEnvWrapper)

        # Fully initialize perceiver with observation objects and type features
        system.perceiver.initialize(
            predicates=planning_components.predicate_container,
            observation_objects=env.observation_objects,
            type_features=env.unwrapped_env.type_features
        )

        skills = {
            GraphPickBlockNotOnShelfSkill(
                system.components, env.unwrapped_env.action_space, env.unwrapped_env.initial_constant_state
            ),
            GraphPickBlockOnShelfSkill(
                system.components, env.unwrapped_env.action_space, env.unwrapped_env.initial_constant_state
            ),
            GraphPlaceBlockNotOnShelfSkill(
                system.components, env.unwrapped_env.action_space, env.unwrapped_env.initial_constant_state
            ),
            GraphPlaceBlockOnShelfSkill(
                system.components, env.unwrapped_env.action_space, env.unwrapped_env.initial_constant_state
            ),
        }

        # Initialize all skills with observation info for vec to state conversion
        for skill in skills:
            skill.set_observation_info(env.observation_objects, env.unwrapped_env.type_features)

        system.components.skills.update(skills)
        return system
