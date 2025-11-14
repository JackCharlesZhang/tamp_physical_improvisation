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

    def _parse_obs(self, obs: NDArray[np.float32]) -> dict[str, Any]:
        """Parse observation into structured data.

        Observation structure (80 dims):
        - target_surface: [0:14]
        - target_block: [14:29]
        - obstruction0: [29:44]
        - obstruction1: [44:59]
        - robot: [59:80]
        """
        return {
            "target_surface": {
                "x": float(obs[0]),
                "y": float(obs[1]),
                "width": float(obs[12]),
                "height": float(obs[13]),
            },
            "target_block": {
                "x": float(obs[14]),
                "y": float(obs[15]),
                "width": float(obs[26]),
                "height": float(obs[27]),
                "held": float(obs[21]) > 0.5,
            },
            "obstructions": [
                {
                    "x": float(obs[29]),
                    "y": float(obs[30]),
                    "width": float(obs[41]),
                    "height": float(obs[42]),
                },
                {
                    "x": float(obs[44]),
                    "y": float(obs[45]),
                    "width": float(obs[56]),
                    "height": float(obs[57]),
                },
            ],
            "robot": {
                "x": float(obs[59]),
                "y": float(obs[60]),
                "theta": float(obs[61]),
                "base_radius": float(obs[72]),
                "arm_joint": float(obs[73]),  # Current arm extension
                "arm_length": float(obs[74]),  # Max arm extension
                "finger_gap": float(obs[77]),
            },
        }


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

        Returns:
            Action [dx, dy, dtheta, darm, dgripper]
        """
        parsed = self._parse_obs(obs)
        robot = parsed["robot"]
        target = parsed["target_block"]
        obstructions = parsed["obstructions"]

        # Target position: above the block
        target_x = target["x"]
        target_y = target["y"] + target["height"] / 2 + 0.1

        robot_x, robot_y = robot["x"], robot["y"]

        # COLLISION AVOIDANCE: Check if too close to obstructions
        for obs_block in obstructions:
            obs_x, obs_y = obs_block["x"], obs_block["y"]
            obs_width = obs_block["width"]

            # If at same y-level and too close, move away first
            if (
                np.isclose(robot_y, obs_y, atol=0.05)
                and abs(robot_x - obs_x) < (robot["base_radius"] + obs_width / 2 + 0.1)
                and not np.isclose(robot_x, obs_x, atol=0.05)
            ):
                # Move away from obstruction
                dx = np.clip(robot_x - obs_x, -0.049, 0.049)
                return np.array([dx, 0.0, 0.0, 0.0, -0.02], dtype=np.float32)

        # Calculate angle to target
        delta_x = target_x - robot_x
        delta_y = target_y - robot_y
        target_angle = np.arctan2(delta_y, delta_x)
        angle_error = target_angle - robot["theta"]

        # Normalize angle to [-pi, pi]
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi

        # ROTATION: Orient toward target first
        if abs(angle_error) > 0.1:  # ~5.7 degrees
            dtheta = np.clip(angle_error, -np.pi / 16 + 0.001, np.pi / 16 - 0.001)
            return np.array([0.0, 0.0, dtheta, 0.0, -0.02], dtype=np.float32)

        dtheta = 0.0  # Already aligned

        # Calculate distance to target
        distance = np.sqrt(delta_x**2 + delta_y**2)

        # STAGED APPROACH: Be very gentle to avoid knocking block away

        # Stage 1: If far away (>0.3m), navigate closer with base only
        if distance > 0.3:
            # Move base slowly, no arm extension
            speed = 0.02  # Much slower
            dx = np.clip(delta_x * speed / distance, -0.02, 0.02)
            dy = np.clip(delta_y * speed / distance, -0.02, 0.02)
            darm = 0.0  # Don't extend arm yet
            dgripper = -0.02  # Keep gripper open
            return np.array([dx, dy, dtheta, darm, dgripper], dtype=np.float32)

        # Stage 2: Medium distance (0.15-0.3m), position precisely + start extending
        if distance > 0.15:
            # Very slow base movement + gentle arm extension
            speed = 0.01
            dx = np.clip(delta_x * speed / distance, -0.01, 0.01)
            dy = np.clip(delta_y * speed / distance, -0.01, 0.01)
            # Extend arm slowly to 60% of distance
            desired_arm_extension = min(distance * 0.6, robot["arm_length"])
            arm_error = desired_arm_extension - robot["arm_joint"]
            darm = np.clip(arm_error, -0.03, 0.03)  # Much slower
            dgripper = -0.02  # Keep gripper open
            return np.array([dx, dy, dtheta, darm, dgripper], dtype=np.float32)

        # Stage 3: Close (<0.15m), STOP base, only extend arm gently
        if distance > 0.08 and not target["held"]:
            # Stop moving base, only extend arm very gently
            dx = 0.0
            dy = 0.0
            # Extend arm to reach target
            desired_arm_extension = min(distance * 0.9, robot["arm_length"])
            arm_error = desired_arm_extension - robot["arm_joint"]
            darm = np.clip(arm_error, -0.02, 0.02)  # Very slow
            dgripper = -0.02  # Keep gripper open
            return np.array([dx, dy, dtheta, darm, dgripper], dtype=np.float32)

        # Stage 4: Very close, grasp
        if not target["held"]:
            # Stop everything, just close gripper
            return np.array([0.0, 0.0, 0.0, 0.0, 0.02], dtype=np.float32)
        else:
            # Already holding, maintain position
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


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

        Returns:
            Action [dx, dy, dtheta, darm, dgripper]
        """
        parsed = self._parse_obs(obs)
        robot = parsed["robot"]
        surface = parsed["target_surface"]
        target = parsed["target_block"]

        # Target position: center of target surface, slightly above
        target_x = surface["x"]
        target_y = surface["y"] + surface["height"] / 2 + target["height"] / 2 + 0.1

        robot_x, robot_y = robot["x"], robot["y"]

        # Calculate angle to target
        delta_x = target_x - robot_x
        delta_y = target_y - robot_y
        target_angle = np.arctan2(delta_y, delta_x)
        angle_error = target_angle - robot["theta"]

        # Normalize angle to [-pi, pi]
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi

        # ROTATION: Orient toward target
        if abs(angle_error) > 0.1:
            dtheta = np.clip(angle_error, -np.pi / 16 + 0.001, np.pi / 16 - 0.001)
            return np.array([0.0, 0.0, dtheta, 0.0, 0.0], dtype=np.float32)

        dtheta = 0.0  # Already aligned

        # Calculate distance
        distance = np.sqrt(delta_x**2 + delta_y**2)

        # STAGED GENTLE PLACEMENT

        # Stage 1: Far away, navigate slowly
        if distance > 0.3:
            speed = 0.02
            dx = np.clip(delta_x * speed / distance, -0.02, 0.02)
            dy = np.clip(delta_y * speed / distance, -0.02, 0.02)
            darm = 0.0
            dgripper = 0.0 if target["held"] else -0.02
            return np.array([dx, dy, dtheta, darm, dgripper], dtype=np.float32)

        # Stage 2: Medium distance, slow approach
        if distance > 0.15:
            speed = 0.01
            dx = np.clip(delta_x * speed / distance, -0.01, 0.01)
            dy = np.clip(delta_y * speed / distance, -0.01, 0.01)
            darm = 0.0
            dgripper = 0.0 if target["held"] else -0.02
            return np.array([dx, dy, dtheta, darm, dgripper], dtype=np.float32)

        # Stage 3: Close, position precisely above surface
        if distance > 0.08:
            speed = 0.005  # Very slow
            dx = np.clip(delta_x * speed / distance, -0.005, 0.005)
            dy = np.clip(delta_y * speed / distance, -0.005, 0.005)
            darm = 0.0
            dgripper = 0.0 if target["held"] else -0.02
            return np.array([dx, dy, dtheta, darm, dgripper], dtype=np.float32)

        # Stage 4: In position, release gently
        if target["held"]:
            # Release gripper
            return np.array([0.0, 0.0, 0.0, 0.0, -0.02], dtype=np.float32)
        else:
            # Already placed
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


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

        Returns:
            Action [dx, dy, dtheta, darm, dgripper]
        """
        parsed = self._parse_obs(obs)
        robot = parsed["robot"]
        surface = parsed["target_surface"]

        # Determine which obstruction to push
        obstruction_obj = objects[1]  # Second object is the obstruction
        obstruction_idx = int(obstruction_obj.name[-1])  # Extract index from "obstructionX"
        obstruction = parsed["obstructions"][obstruction_idx]

        robot_x, robot_y = robot["x"], robot["y"]

        # Calculate push direction: away from target surface center
        push_dir_x = obstruction["x"] - surface["x"]
        if abs(push_dir_x) < 0.01:
            # If directly above, push to the right
            push_dir_x = 1.0

        # Normalize direction
        push_dir_x = push_dir_x / abs(push_dir_x) if push_dir_x != 0 else 1.0

        # Position behind obstruction (opposite to push direction)
        approach_offset = 0.2
        target_x = obstruction["x"] - push_dir_x * approach_offset
        target_y = obstruction["y"]

        # Calculate angle to approach position
        delta_x = target_x - robot_x
        delta_y = target_y - robot_y
        distance_to_position = np.sqrt(delta_x**2 + delta_y**2)

        # If far from position, navigate there
        if distance_to_position > 0.1:
            # Calculate angle to approach position
            target_angle = np.arctan2(delta_y, delta_x)
            angle_error = target_angle - robot["theta"]

            # Normalize angle
            while angle_error > np.pi:
                angle_error -= 2 * np.pi
            while angle_error < -np.pi:
                angle_error += 2 * np.pi

            # ROTATION: Orient toward approach position
            if abs(angle_error) > 0.1:
                dtheta = np.clip(angle_error, -np.pi / 16 + 0.001, np.pi / 16 - 0.001)
                return np.array([0.0, 0.0, dtheta, 0.0, 0.0], dtype=np.float32)

            # NAVIGATION: Move to approach position SLOWLY
            speed = 0.02  # Much slower
            dx = np.clip(delta_x * speed / distance_to_position, -0.02, 0.02)
            dy = np.clip(delta_y * speed / distance_to_position, -0.02, 0.02)
            return np.array([dx, dy, 0.0, 0.0, 0.0], dtype=np.float32)

        # In position - now push GENTLY!
        # Orient toward push direction (toward obstruction)
        push_angle = np.arctan2(0.0, push_dir_x)  # Pushing horizontally
        angle_error = push_angle - robot["theta"]

        # Normalize angle
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi

        # ROTATION: Orient in push direction
        if abs(angle_error) > 0.05:
            dtheta = np.clip(angle_error, -np.pi / 16 + 0.001, np.pi / 16 - 0.001)
            return np.array([0.0, 0.0, dtheta, 0.0, 0.0], dtype=np.float32)

        # ARM EXTENSION: Extend arm SLOWLY to push
        distance_to_obs = abs(robot_x - obstruction["x"])
        desired_arm = min(distance_to_obs * 0.7, robot["arm_length"])
        darm = np.clip(desired_arm - robot["arm_joint"], -0.03, 0.03)  # Much slower

        # PUSH: Move forward GENTLY to nudge, not slam
        dx = np.clip(push_dir_x * 0.01, -0.01, 0.01)  # 5x slower!

        return np.array([dx, 0.0, 0.0, darm, 0.0], dtype=np.float32)


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

        Observation structure (for num_obstructions=2, 80 dims):
        - target_surface: features[0:14]  (14 features)
        - target_block: features[14:29]   (15 features)
        - obstruction0: features[29:44]   (15 features)
        - obstruction1: features[44:59]   (15 features)
        - robot: features[59:80]          (21 features)
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
        """Check if block is on surface.

        Exactly matches prbench's is_on logic from geom2d/utils.py:
        - Gets bottom 2 vertices of block
        - Offsets y by -tol
        - Checks if offset points are contained in surface rectangle
        """
        tol = 0.025  # Matches prbench default

        # Bottom 2 vertices of block (assuming axis-aligned)
        # These are bottom-left and bottom-right corners
        bottom_left = (block_x - block_width / 2, block_y - block_height / 2)
        bottom_right = (block_x + block_width / 2, block_y - block_height / 2)

        # Surface rectangle bounds (Rectangle origin is bottom-left corner)
        surface_left = surface_x - surface_width / 2
        surface_right = surface_x + surface_width / 2
        surface_bottom = surface_y - surface_height / 2
        surface_top = surface_y + surface_height / 2

        # Check both bottom vertices with y offset
        for vertex_x, vertex_y in [bottom_left, bottom_right]:
            offset_y = vertex_y - tol
            # Check if point (vertex_x, offset_y) is contained in surface rectangle
            # For Rectangle.contains_point with theta=0:
            # point is in rectangle if x in [rect.x, rect.x + width] and y in [rect.y, rect.y + height]
            if not (surface_left <= vertex_x <= surface_right):
                return False
            if not (surface_bottom <= offset_y <= surface_top):
                return False

        return True

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
