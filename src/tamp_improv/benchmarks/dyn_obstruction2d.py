"""DynObstruction2D environment implementation with physics-based manipulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from relational_structs import (
    GroundAtom,
    LiftedOperator,
    Object,
    PDDLDomain,
    Predicate,
    Type,
    Variable,
)
from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver

from tamp_improv.benchmarks.base import (
    BaseTAMPSystem,
    ImprovisationalTAMPSystem,
    PlanningComponents,
)
from tamp_improv.benchmarks.wrappers import ImprovWrapper


@dataclass
class DynObstruction2DTypes:
    """Container for DynObstruction2D types."""

    def __init__(self) -> None:
        """Initialize types."""
        self.robot = Type("robot")
        self.block = Type("block")  # target block
        self.obstruction = Type("obstruction")  # obstruction blocks
        self.surface = Type("surface")  # target surface

    def as_set(self) -> set[Type]:
        """Convert to set of types."""
        return {self.robot, self.block, self.obstruction, self.surface}


@dataclass
class DynObstruction2DPredicates:
    """Container for DynObstruction2D predicates."""

    def __init__(self, types: DynObstruction2DTypes) -> None:
        """Initialize predicates."""
        self.on = Predicate("On", [types.block, types.surface])
        self.clear = Predicate("Clear", [types.surface])
        self.holding = Predicate("Holding", [types.robot, types.block])
        self.gripper_empty = Predicate("GripperEmpty", [types.robot])
        self.obstructing = Predicate("Obstructing", [types.obstruction, types.surface])
        self.obstruction_clear = Predicate(
            "ObstructionClear", [types.obstruction, types.surface]
        )

    def __getitem__(self, key: str) -> Predicate:
        """Get predicate by name."""
        return next(p for p in self.as_set() if p.name == key)

    def as_set(self) -> set[Predicate]:
        """Convert to set of predicates."""
        return {
            self.on,
            self.clear,
            self.holding,
            self.gripper_empty,
            self.obstructing,
            self.obstruction_clear,
        }


class BaseDynObstruction2DSkill(LiftedOperatorSkill[NDArray[np.float32], NDArray[np.float32]]):
    """Base class for DynObstruction2D environment skills."""

    def __init__(self, components: PlanningComponents[NDArray[np.float32]]) -> None:
        """Initialize skill."""
        super().__init__()
        self._components = components
        self._lifted_operator = self._get_lifted_operator()

    def _get_lifted_operator(self) -> LiftedOperator:
        """Get the operator this skill implements."""
        return next(
            op
            for op in self._components.operators
            if op.name == self._get_operator_name()
        )

    def _get_operator_name(self) -> str:
        """Get the name of the operator this skill implements."""
        raise NotImplementedError


class PickUpSkill(BaseDynObstruction2DSkill):
    """Skill for picking up the target block."""

    def _get_operator_name(self) -> str:
        return "PickUp"

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: NDArray[np.float32],  # type: ignore[override]
    ) -> NDArray[np.float32]:
        """Generate action to pick up target block.

        Args:
            objects: [robot, target_block, target_surface]
            obs: Flat observation array
        """
        # TODO: Parse observation to extract positions
        # For now, return a placeholder action
        # Action format: [dx, dy, dtheta, darm, dgripper]
        return np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


class PlaceOnTargetSkill(BaseDynObstruction2DSkill):
    """Skill for placing target block on target surface."""

    def _get_operator_name(self) -> str:
        return "PlaceOnTarget"

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: NDArray[np.float32],  # type: ignore[override]
    ) -> NDArray[np.float32]:
        """Generate action to place block on target surface.

        Args:
            objects: [robot, target_block, target_surface]
            obs: Flat observation array
        """
        # TODO: Parse observation and compute placement action
        # Action format: [dx, dy, dtheta, darm, dgripper]
        return np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


class PushSkill(BaseDynObstruction2DSkill):
    """Skill for pushing obstruction blocks off target surface."""

    def _get_operator_name(self) -> str:
        return "Push"

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: NDArray[np.float32],  # type: ignore[override]
    ) -> NDArray[np.float32]:
        """Generate action to push obstruction block.

        Args:
            objects: [robot, obstruction, target_surface]
            obs: Flat observation array
        """
        # TODO: Parse observation and compute push direction
        # Action format: [dx, dy, dtheta, darm, dgripper]
        return np.array([0.05, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


class DynObstruction2DPerceiver(Perceiver[NDArray[np.float32]]):
    """Perceiver for DynObstruction2D environment."""

    def __init__(
        self, types: DynObstruction2DTypes, num_obstructions: int = 2
    ) -> None:
        """Initialize with required types."""
        self._robot = Object("robot", types.robot)
        self._target_block = Object("target_block", types.block)
        self._target_surface = Object("target_surface", types.surface)
        self._obstructions = [
            Object(f"obstruction{i}", types.obstruction)
            for i in range(num_obstructions)
        ]
        self._predicates: DynObstruction2DPredicates | None = None
        self._types = types
        self._num_obstructions = num_obstructions

    def initialize(self, predicates: DynObstruction2DPredicates) -> None:
        """Initialize predicates after environment creation."""
        self._predicates = predicates

    @property
    def predicates(self) -> DynObstruction2DPredicates:
        """Get predicates, ensuring they're initialized."""
        if self._predicates is None:
            raise RuntimeError("Predicates not initialized. Call initialize() first.")
        return self._predicates

    def get_objects(self):
        """Get all objects in the environment."""
        return set([self._robot, self._target_block, self._target_surface] + self._obstructions)

    def reset(
        self,
        obs: NDArray[np.float32],
        _info: dict[str, Any],
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Reset perceiver with observation and info."""
        objects = self.get_objects()
        atoms = self._get_atoms(obs)
        goal = {
            self.predicates["On"]([self._target_block, self._target_surface]),
            self.predicates["GripperEmpty"]([self._robot]),
        }
        return objects, atoms, goal

    def step(self, obs: NDArray[np.float32]) -> set[GroundAtom]:
        """Step perceiver with observation."""
        return self._get_atoms(obs)

    def _get_atoms(self, obs: NDArray[np.float32]) -> set[GroundAtom]:
        """Extract PDDL atoms from observation.

        Observation structure (for num_obstructions=2, ~81 dims):
        - target_surface: features[0:14]  (x, y, theta, vx, vy, omega, static, held, color_r, color_g, color_b, z_order, width, height)
        - target_block: features[14:29]   (x, y, theta, vx, vy, omega, static, held, color_r, color_g, color_b, z_order, width, height, mass)
        - obstruction0: features[29:44]   (same as target_block)
        - obstruction1: features[44:59]   (same as obstruction0)
        - robot: features[59:81]          (x, y, theta, vx_base, vy_base, omega_base, vx_arm, vy_arm, omega_arm, vx_gripper, vy_gripper, omega_gripper, static, base_radius, arm_joint, arm_length, gripper_base_width, gripper_base_height, finger_gap, finger_height, finger_width)
        """
        atoms = set()

        # Parse observations
        # Target surface (14 features)
        target_surface_x = obs[0]
        target_surface_y = obs[1]
        target_surface_width = obs[12]
        target_surface_height = obs[13]

        # Target block (15 features)
        target_block_x = obs[14]
        target_block_y = obs[15]
        target_block_width = obs[26]
        target_block_height = obs[27]
        target_block_held = obs[21] > 0.5  # "held" feature

        # Obstructions (15 features each)
        obstruction_positions = []
        obstruction_held_status = []
        for i in range(self._num_obstructions):
            offset = 29 + i * 15
            obs_x = obs[offset]
            obs_y = obs[offset + 1]
            obs_held = obs[offset + 7] > 0.5
            obstruction_positions.append((obs_x, obs_y))
            obstruction_held_status.append(obs_held)

        # Robot (22 features, starts at 59 for 2 obstructions)
        robot_offset = 29 + self._num_obstructions * 15
        robot_x = obs[robot_offset]
        robot_y = obs[robot_offset + 1]
        finger_gap = obs[robot_offset + 18]  # gripper finger_gap

        # Determine gripper status
        gripper_closed = finger_gap < 0.1  # Threshold for closed gripper

        # Check if robot is holding target block
        if target_block_held and gripper_closed:
            atoms.add(self.predicates["Holding"]([self._robot, self._target_block]))
        else:
            atoms.add(self.predicates["GripperEmpty"]([self._robot]))

        # Check if target block is on target surface
        if self._is_on_surface(
            target_block_x,
            target_block_y,
            target_block_width,
            target_block_height,
            target_surface_x,
            target_surface_y,
            target_surface_width,
            target_surface_height,
        ):
            atoms.add(self.predicates["On"]([self._target_block, self._target_surface]))

        # Check obstruction status
        surface_obstructed = False
        for i, (obs_x, obs_y) in enumerate(obstruction_positions):
            if self._is_obstructing(
                obs_x,
                obs_y,
                target_surface_x,
                target_surface_y,
                target_surface_width,
                target_surface_height,
            ):
                atoms.add(
                    self.predicates["Obstructing"](
                        [self._obstructions[i], self._target_surface]
                    )
                )
                surface_obstructed = True
            else:
                atoms.add(
                    self.predicates["ObstructionClear"](
                        [self._obstructions[i], self._target_surface]
                    )
                )

        # Surface is clear if no obstructions
        if not surface_obstructed and not target_block_held:
            atoms.add(self.predicates["Clear"]([self._target_surface]))

        return atoms

    def _is_on_surface(
        self,
        block_x: float,
        block_y: float,
        block_width: float,
        block_height: float,
        surface_x: float,
        surface_y: float,
        surface_width: float,
        surface_height: float,
    ) -> bool:
        """Check if block is completely on surface."""
        # Block bottom must be touching surface top
        block_bottom = block_y - block_height / 2
        surface_top = surface_y + surface_height / 2

        # Check vertical alignment (block sitting on surface)
        if not np.isclose(block_bottom, surface_top, atol=0.05):
            return False

        # Check horizontal containment
        block_left = block_x - block_width / 2
        block_right = block_x + block_width / 2
        surface_left = surface_x - surface_width / 2
        surface_right = surface_x + surface_width / 2

        return block_left >= surface_left and block_right <= surface_right

    def _is_obstructing(
        self,
        obs_x: float,
        obs_y: float,
        surface_x: float,
        surface_y: float,
        surface_width: float,
        surface_height: float,
    ) -> bool:
        """Check if obstruction is blocking the target surface."""
        # Simple check: obstruction overlaps with surface horizontally
        surface_left = surface_x - surface_width / 2
        surface_right = surface_x + surface_width / 2

        # Obstruction is on surface if its x coordinate is within surface bounds
        # and it's at roughly the same y level
        surface_y_level = surface_y + surface_height / 2
        return surface_left <= obs_x <= surface_right and np.isclose(
            obs_y, surface_y_level, atol=0.1
        )


class BaseDynObstruction2DTAMPSystem(
    BaseTAMPSystem[NDArray[np.float32], NDArray[np.float32]]
):
    """Base TAMP system for DynObstruction2D environment."""

    def __init__(
        self,
        planning_components: PlanningComponents[NDArray[np.float32]],
        num_obstructions: int = 2,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize DynObstruction2D TAMP system."""
        self._render_mode = render_mode
        self._num_obstructions = num_obstructions
        super().__init__(
            planning_components, name="DynObstruction2DTAMPSystem", seed=seed
        )

    def _create_env(self) -> gym.Env:
        """Create base environment.

        NOTE: Requires prbench to be installed.
        Install with: pip install -e 'path/to/prbench[dynamic2d]'
        """
        try:
            from prbench.envs.dynamic2d.dyn_obstruction2d import DynObstruction2DEnv
        except ImportError as e:
            raise ImportError(
                "DynObstruction2DEnv requires prbench to be installed. "
                "Install with: pip install -e 'path/to/prbench[dynamic2d]'"
            ) from e

        return DynObstruction2DEnv(
            num_obstructions=self._num_obstructions, render_mode=self._render_mode
        )

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "dyn-obstruction2d-domain"

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
        cls, num_obstructions: int = 2
    ) -> PlanningComponents[NDArray[np.float32]]:
        """Create planning components for DynObstruction2D system."""
        types_container = DynObstruction2DTypes()
        types_set = types_container.as_set()
        predicates = DynObstruction2DPredicates(types_container)
        perceiver = DynObstruction2DPerceiver(types_container, num_obstructions)
        perceiver.initialize(predicates)

        robot = Variable("?robot", types_container.robot)
        block = Variable("?block", types_container.block)
        obstruction = Variable("?obstruction", types_container.obstruction)
        surface = Variable("?surface", types_container.surface)

        operators = {
            LiftedOperator(
                "PickUp",
                [robot, block, surface],
                preconditions={
                    predicates["GripperEmpty"]([robot]),
                },
                add_effects={
                    predicates["Holding"]([robot, block]),
                },
                delete_effects={
                    predicates["GripperEmpty"]([robot]),
                },
            ),
            LiftedOperator(
                "PlaceOnTarget",
                [robot, block, surface],
                preconditions={
                    predicates["Holding"]([robot, block]),
                    predicates["Clear"]([surface]),
                },
                add_effects={
                    predicates["On"]([block, surface]),
                    predicates["GripperEmpty"]([robot]),
                },
                delete_effects={
                    predicates["Holding"]([robot, block]),
                    predicates["Clear"]([surface]),
                },
            ),
            LiftedOperator(
                "Push",
                [robot, obstruction, surface],
                preconditions={
                    predicates["Obstructing"]([obstruction, surface]),
                    predicates["GripperEmpty"]([robot]),
                },
                add_effects={
                    predicates["ObstructionClear"]([obstruction, surface]),
                },
                delete_effects={
                    predicates["Obstructing"]([obstruction, surface]),
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
        num_obstructions: int = 2,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> BaseDynObstruction2DTAMPSystem:
        """Factory method for creating system with default components."""
        planning_components = cls._create_planning_components(num_obstructions)
        system = cls(
            planning_components,
            num_obstructions=num_obstructions,
            seed=seed,
            render_mode=render_mode,
        )
        skills = {
            PickUpSkill(system.components),  # type: ignore[arg-type]
            PlaceOnTargetSkill(system.components),  # type: ignore[arg-type]
            PushSkill(system.components),  # type: ignore[arg-type]
        }
        system.components.skills.update(skills)
        return system


class DynObstruction2DTAMPSystem(
    ImprovisationalTAMPSystem[NDArray[np.float32], NDArray[np.float32]],
    BaseDynObstruction2DTAMPSystem,
):
    """TAMP system for DynObstruction2D environment with improvisational policy
    learning enabled."""

    def __init__(
        self,
        planning_components: PlanningComponents[NDArray[np.float32]],
        num_obstructions: int = 2,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize DynObstruction2D TAMP system."""
        self._num_obstructions = num_obstructions
        self._render_mode = render_mode
        super().__init__(planning_components, seed=seed, render_mode=render_mode)

    def _create_wrapped_env(
        self, components: PlanningComponents[NDArray[np.float32]]
    ) -> gym.Env:
        """Create wrapped environment for training."""
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
    ) -> DynObstruction2DTAMPSystem:
        """Factory method for creating improvisational system with default
        components."""
        planning_components = cls._create_planning_components(num_obstructions)
        system = cls(
            planning_components,
            num_obstructions=num_obstructions,
            seed=seed,
            render_mode=render_mode,
        )
        skills = {
            PickUpSkill(system.components),  # type: ignore[arg-type]
            PlaceOnTargetSkill(system.components),  # type: ignore[arg-type]
            PushSkill(system.components),  # type: ignore[arg-type]
        }
        system.components.skills.update(skills)
        return system
