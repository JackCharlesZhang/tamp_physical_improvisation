"""DynObstruction2D environment implementation with physics-based manipulation."""

from __future__ import annotations

# Debug flag to control skill phase logging
DEBUG_SKILL_PHASES = False

import abc
from dataclasses import dataclass
from typing import Any, Sequence, Union

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray
from relational_structs import (
    GroundAtom,
    LiftedOperator,
    Object,
    ObjectCentricState,
    PDDLDomain,
    Predicate,
    Type,
    Variable,
)
from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver
from tomsgeoms2d.structs import Rectangle

from tamp_improv.benchmarks.base import (
    BaseTAMPSystem,
    ImprovisationalTAMPSystem,
    PlanningComponents,
)
from tamp_improv.benchmarks.wrappers import ImprovWrapper


# Monkey-patch Tobject into tomsgeoms2d if it doesn't exist
# (prbench imports it but dyn_obstruction2d doesn't actually use it)
# MUST happen BEFORE importing prbench!
import tomsgeoms2d.structs
if not hasattr(tomsgeoms2d.structs, "Tobject"):
    # Create a dummy Tobject class to satisfy prbench imports
    from tomsgeoms2d.structs import Lobject
    tomsgeoms2d.structs.Tobject = Lobject  # Use Lobject as a stand-in

from prbench.envs.dynamic2d.dyn_obstruction2d import DynObstruction2DEnv, DynObstruction2DEnvConfig


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_horizontal_overlap_percentage(
    block_x: float, block_y: float, block_width: float, block_height: float, block_theta: float,
    obs_x: float, obs_y: float, obs_width: float, obs_height: float, obs_theta: float
) -> float:
    """Compute what percentage of the held block's bottom face is covered by an obstruction's top face.

    Uses tomsgeoms2d Rectangle to handle rotations properly.

    Args:
        block_x, block_y: Center position of held block
        block_width, block_height: Dimensions of held block
        block_theta: Rotation of held block
        obs_x, obs_y: Center position of obstruction
        obs_width, obs_height: Dimensions of obstruction
        obs_theta: Rotation of obstruction

    Returns:
        Percentage (0.0 to 1.0) of held block's bottom face covered by obstruction's top face
    """
    # Create rectangles for both blocks
    block_rect = Rectangle.from_center(block_x, block_y, block_width, block_height, block_theta)
    obs_rect = Rectangle.from_center(obs_x, obs_y, obs_width, obs_height, obs_theta)

    # Check if obstruction is below the held block (within reasonable vertical range)
    if obs_y > block_y:
        return 0.0  # Obstruction is above, not below

    # Compute horizontal overlap by checking if bottom vertices of held block
    # project onto the top face of the obstruction
    # Get bottom two vertices of held block (sorted by y-coordinate)
    block_vertices = sorted(block_rect.vertices, key=lambda v: v[1])
    bottom_vertices = block_vertices[:2]  # Two lowest vertices

    # Check how many bottom vertices are contained in the obstruction's horizontal extent
    overlap_count = 0
    for vx, vy in bottom_vertices:
        # Project vertex down to obstruction's top surface level
        # Check if this projection falls within obstruction's bounds
        if obs_rect.contains_point(vx, obs_y + obs_height/2):
            overlap_count += 1

    # Return percentage: 0.0 (no overlap), 0.5 (one vertex), 1.0 (both vertices)
    return overlap_count / 2.0


# ==============================================================================
# TYPES AND PREDICATES
# ==============================================================================

@dataclass
class DynObstruction2DTypes:
    """Container for DynObstruction2D types.

    Uses the same types as prbench's DynObstruction2DEnv for compatibility.
    """

    def __init__(self) -> None:
        """Initialize types using prbench types."""
        # Import prbench types
        from prbench.envs.dynamic2d.dyn_obstruction2d import TargetSurfaceType
        from prbench.envs.dynamic2d.object_types import (
            Dynamic2DType,
            DynRectangleType,
            KinRobotType,
        )

        # Use prbench's type system for physics, but create distinct types for planning
        # This allows the PDDL planner to distinguish target block from obstructions
        self.robot = KinRobotType
        # Create a common parent type for things that can be picked up
        self.pickable = Type("pickable", parent=DynRectangleType)
        self.block = Type("target_block", parent=self.pickable)  # target block (dynamic)
        self.obstruction = Type("obstruction", parent=self.pickable)  # obstruction blocks (dynamic)
        self.surface = TargetSurfaceType  # target surface (kinematic, special type)

        # Store parent type for PDDL domain
        self.dynamic2d = Dynamic2DType

    def as_set(self) -> set[Type]:
        """Convert to set of types, including parent types for PDDL."""
        types = {self.robot, self.block, self.obstruction, self.surface}
        # Add all parent types to satisfy PDDL type hierarchy
        all_types = set(types)
        for t in types:
            current = t
            while current.parent is not None:
                all_types.add(current.parent)
                current = current.parent
        return all_types


@dataclass
class DynObstruction2DPredicates:
    """Container for DynObstruction2D predicates."""

    def __init__(self, types: DynObstruction2DTypes) -> None:
        """Initialize predicates."""
        self.on = Predicate("On", [types.block, types.surface])
        self.clear = Predicate("Clear", [types.surface])
        self.blocking = Predicate("Blocking", [types.obstruction, types.surface])  # Obstruction blocking surface
        self.holding = Predicate("Holding", [types.robot, types.pickable])  # Use pickable parent type
        self.gripper_empty = Predicate("GripperEmpty", [types.robot])

    def __getitem__(self, key: str) -> Predicate:
        """Get predicate by name."""
        return next(p for p in self.as_set() if p.name == key)

    def as_set(self) -> set[Predicate]:
        """Convert to set of predicates."""
        return {
            self.on,
            self.clear,
            self.blocking,
            self.holding,
            self.gripper_empty,
        }


def _obs_to_state(obs: NDArray[np.float32], objects: Sequence[Object]) -> ObjectCentricState:
    """Convert flat observation array to ObjectCentricState.

    This allows us to use the same controller interface as other environments.
    Observation structure (80 dims):
    - target_surface: [0:14]
    - target_block: [14:29]
    - obstruction0: [29:44]
    - obstruction1: [44:59]
    - robot: [59:80]
    """
    # Create mapping from object names to features
    state_dict: dict[Object, dict[str, float]] = {}

    # Find objects by type
    robot = None
    target_block = None
    target_surface = None
    obstructions = []

    for obj in objects:
        if obj.name == "robot":
            robot = obj
        elif obj.name == "target_block":
            target_block = obj
        elif obj.name == "target_surface":
            target_surface = obj
        elif obj.name.startswith("obstruction"):
            obstructions.append(obj)

    # Parse target surface (14 features)
    if target_surface is not None:
        state_dict[target_surface] = {
            "x": float(obs[0]),
            "y": float(obs[1]),
            "theta": float(obs[2]),
            "vx": float(obs[3]),
            "vy": float(obs[4]),
            "omega": float(obs[5]),
            "static": float(obs[6]),
            "held": float(obs[7]),
            "color_r": float(obs[8]),
            "color_g": float(obs[9]),
            "color_b": float(obs[10]),
            "z_order": float(obs[11]),
            "width": float(obs[12]),
            "height": float(obs[13]),
        }

    # Parse target block (15 features)
    if target_block is not None:
        state_dict[target_block] = {
            "x": float(obs[14]),
            "y": float(obs[15]),
            "theta": float(obs[16]),
            "vx": float(obs[17]),
            "vy": float(obs[18]),
            "omega": float(obs[19]),
            "static": float(obs[20]),
            "held": float(obs[21]),
            "color_r": float(obs[22]),
            "color_g": float(obs[23]),
            "color_b": float(obs[24]),
            "z_order": float(obs[25]),
            "width": float(obs[26]),
            "height": float(obs[27]),
            "mass": float(obs[28]),
        }

    # Parse obstructions (15 features each)
    for i, obstruction_obj in enumerate(sorted(obstructions, key=lambda o: o.name)):
        offset = 29 + i * 15
        state_dict[obstruction_obj] = {
            "x": float(obs[offset]),
            "y": float(obs[offset + 1]),
            "theta": float(obs[offset + 2]),
            "vx": float(obs[offset + 3]),
            "vy": float(obs[offset + 4]),
            "omega": float(obs[offset + 5]),
            "static": float(obs[offset + 6]),
            "held": float(obs[offset + 7]),
            "color_r": float(obs[offset + 8]),
            "color_g": float(obs[offset + 9]),
            "color_b": float(obs[offset + 10]),
            "z_order": float(obs[offset + 11]),
            "width": float(obs[offset + 12]),
            "height": float(obs[offset + 13]),
            "mass": float(obs[offset + 14]),
        }

    # Parse robot (21 features)
    if robot is not None:
        robot_offset = 29 + len(obstructions) * 15
        state_dict[robot] = {
            "x": float(obs[robot_offset]),
            "y": float(obs[robot_offset + 1]),
            "theta": float(obs[robot_offset + 2]),
            "vx_base": float(obs[robot_offset + 3]),
            "vy_base": float(obs[robot_offset + 4]),
            "omega_base": float(obs[robot_offset + 5]),
            "vx_arm": float(obs[robot_offset + 6]),
            "vy_arm": float(obs[robot_offset + 7]),
            "omega_arm": float(obs[robot_offset + 8]),
            "vx_gripper": float(obs[robot_offset + 9]),
            "vy_gripper": float(obs[robot_offset + 10]),
            "omega_gripper": float(obs[robot_offset + 11]),
            "static": float(obs[robot_offset + 12]),
            "base_radius": float(obs[robot_offset + 13]),
            "arm_joint": float(obs[robot_offset + 14]),
            "arm_length": float(obs[robot_offset + 15]),
            "gripper_base_width": float(obs[robot_offset + 16]),
            "gripper_base_height": float(obs[robot_offset + 17]),
            "finger_gap": float(obs[robot_offset + 18]),
            "finger_height": float(obs[robot_offset + 19]),
            "finger_width": float(obs[robot_offset + 20]),
        }

    # Use prbench's type features (already defined globally)
    from prbench.envs.dynamic2d.object_types import Dynamic2DRobotEnvTypeFeatures

    # Convert state_dict values from dicts to numpy arrays
    data_arrays = {obj: np.array(list(features.values()), dtype=np.float32)
                   for obj, features in state_dict.items()}

    return ObjectCentricState(data_arrays, Dynamic2DRobotEnvTypeFeatures)


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
        # Target surface (14 features): x, y, theta, vx, vy, omega, static, held, r, g, b, z_order, width, height
        target_surface_x = obs[0]
        target_surface_y = obs[1]
        target_surface_theta = obs[2]
        target_surface_width = obs[12]
        target_surface_height = obs[13]

        # Target block (15 features): x, y, theta, vx, vy, omega, static, held, r, g, b, z_order, width, height, mass
        target_block_x = obs[14]
        target_block_y = obs[15]
        target_block_theta = obs[16]
        target_block_width = obs[26]
        target_block_height = obs[27]
        target_block_held = obs[21] > 0.5  # "held" feature

        # Obstructions (15 features each)
        obstruction_data = []  # List of (x, y, width, height)
        obstruction_held_status = []
        for i in range(self._num_obstructions):
            offset = 29 + i * 15
            obs_x = obs[offset]
            obs_y = obs[offset + 1]
            obs_held = obs[offset + 7] > 0.5
            obs_width = obs[offset + 12]  # width feature
            obs_height = obs[offset + 13]  # height feature
            obstruction_data.append((obs_x, obs_y, obs_width, obs_height))
            obstruction_held_status.append(obs_held)

        # Robot (22 features, starts at 59 for 2 obstructions)
        robot_offset = 29 + self._num_obstructions * 15
        robot_x = obs[robot_offset]
        robot_y = obs[robot_offset + 1]
        finger_gap = obs[robot_offset + 18]  # gripper finger_gap

        # DEBUG: Log held status and gripper state
        from tamp_improv.benchmarks.debug_physics import DEBUG_SKILL_PHASES
        # if DEBUG_SKILL_PHASES:
        #     print(f"[Perceiver] target_block_held={target_block_held}, finger_gap={finger_gap:.3f}")

        # Check if robot is holding anything (target block or obstructions)
        # If held is True (physics says it's in hand), then it's holding it
        # regardless of finger gap (the gripper grasp callback already verified proper grasping)
        holding_something = False
        if target_block_held:
            # if DEBUG_SKILL_PHASES:
            #     print(f"[Perceiver] Robot holding target_block")
            atoms.add(self.predicates["Holding"]([self._robot, self._target_block]))
            holding_something = True
        else:
            # Check if holding any obstruction
            for i, obs_held in enumerate(obstruction_held_status):
                if obs_held:
                    print(f"[Perceiver] Robot holding obstruction{i}")
                    atoms.add(self.predicates["Holding"]([self._robot, self._obstructions[i]]))
                    holding_something = True
                    break  # Can only hold one object at a time

        if not holding_something:
            print(f"[Perceiver] Adding GripperEmpty predicate")
            atoms.add(self.predicates["GripperEmpty"]([self._robot]))

        # Check if target block is on target surface
        if self._is_on_surface(
            target_block_x,
            target_block_y,
            target_block_theta,
            target_block_width,
            target_block_height,
            target_surface_x,
            target_surface_y,
            target_surface_theta,
            target_surface_width,
            target_surface_height,
        ):
            atoms.add(self.predicates["On"]([self._target_block, self._target_surface]))

        # Check each obstruction to see if it's blocking the surface
        # Add Blocking(obstruction, surface) atoms for each obstruction that's blocking
        for i, (obs_x, obs_y, obs_width, obs_height) in enumerate(obstruction_data):
            if self._is_obstructing(
                obs_x,
                obs_y,
                obs_width,
                obs_height,
                target_surface_x,
                target_surface_y,
                target_surface_width,
                target_surface_height,
            ):
                # This obstruction is blocking the surface
                atoms.add(self.predicates["Blocking"]([self._obstructions[i], self._target_surface]))

        # Surface is clear if no Blocking atoms exist for it (derived predicate)
        any_blocking = any(
            atom.predicate.name == "Blocking" and atom.entities[1] == self._target_surface
            for atom in atoms
        )
        if not any_blocking:
            # print(f"[Perceiver] Adding Clear predicate (no blocking atoms)")
            atoms.add(self.predicates["Clear"]([self._target_surface]))
        # else:
            # print(f"[Perceiver] NOT adding Clear predicate (surface is obstructed)")

        return atoms

    def _is_on_surface(
        self,
        block_x: float,
        block_y: float,
        block_theta: float,
        block_width: float,
        block_height: float,
        surface_x: float,
        surface_y: float,
        surface_theta: float,
        surface_width: float,
        surface_height: float,
    ) -> bool:
        """Check if block is on surface.

        Exactly matches prbench's is_on logic from geom2d/utils.py using tomsgeoms2d.
        """
        tol = 0.025  # Matches prbench default

        # Create Rectangle objects using tomsgeoms2d (same as prbench)
        block_geom = Rectangle.from_center(
            block_x, block_y, block_width, block_height, block_theta
        )
        surface_geom = Rectangle.from_center(
            surface_x, surface_y, surface_width, surface_height, surface_theta
        )

        # Get bottom 2 vertices of block (sorted by y coordinate)
        sorted_vertices = sorted(block_geom.vertices, key=lambda v: v[1])
        bottom_two = sorted_vertices[:2]

        # Check if both bottom vertices (with offset) are contained in surface
        for x, y in bottom_two:
            offset_y = y - tol
            if not surface_geom.contains_point(x, offset_y):
                return False

        return True

    def _is_obstructing(
        self,
        obs_x: float,
        obs_y: float,
        obs_width: float,
        obs_height: float,
        surface_x: float,
        surface_y: float,
        surface_width: float,
        surface_height: float,
    ) -> bool:
        """Check if obstruction is blocking the target surface.

        Uses the same overlap logic as PlaceOnTarget skill to ensure consistency.
        """
        overlap = compute_horizontal_overlap_percentage(
            block_x=obs_x, block_y=obs_y,
            block_width=obs_width, block_height=obs_height, block_theta=0.0,
            obs_x=surface_x, obs_y=surface_y,
            obs_width=surface_width, obs_height=surface_height, obs_theta=0.0
        )

        # Match the threshold used in PlaceOnTarget skill
        return overlap > 0.1


# ==============================================================================
# SLAP-COMPATIBLE PHASE-BASED SKILLS
# ==============================================================================

class BaseDynObstruction2DSkill(
    LiftedOperatorSkill[NDArray[np.float32], NDArray[np.float32]]
):
    """Base class for DynObstruction2D SLAP skills."""

    # Constants
    POSITION_TOL = 5e-2  # Looser tolerance for physics
    SAFE_Y = 1.5  # Higher safe navigation height for better clearance
    GARBAGE_X = 0.3  # X-position for garbage/disposal placement
    TARGET_THETA = -np.pi / 2  # -90° = pointing down (theta=0 is pointing right)
    MAX_DX = MAX_DY = 5e-2
    MAX_DTHETA = np.pi / 16
    MAX_DARM, MAX_DGRIPPER = 1e-1, 2e-2

    def __init__(self, components: PlanningComponents[NDArray[np.float32]]) -> None:
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

    def _parse_obs(self, obs: NDArray[np.float32]) -> dict[str, float]:
        """Parse observation (80 dims: surface[0:14], block[14:29], obs0[29:44], obs1[44:59], robot[59:80])."""
        return {
            'surface_x': obs[0], 'surface_y': obs[1], 'surface_width': obs[12], 'surface_height': obs[13],
            'block_x': obs[14], 'block_y': obs[15], 'block_width': obs[26], 'block_height': obs[27], 'target_block_held': bool(obs[21]),
            'obs0_x': obs[29], 'obs0_y': obs[30], 'obs0_width': obs[41], 'obs0_height': obs[42],
            'obs1_x': obs[44], 'obs1_y': obs[45], 'obs1_width': obs[56], 'obs1_height': obs[57],
            'robot_x': obs[59], 'robot_y': obs[60], 'robot_theta': obs[61],
            'arm_joint': obs[73], 'arm_length_max': obs[74], 'gripper_base_height': obs[76], 'finger_gap': obs[77]
        }

    @staticmethod
    def _angle_diff(target: float, current: float) -> float:
        """Compute shortest angular distance from current to target, handling wrapping."""
        diff = target - current
        # Normalize to [-π, π]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff

    @staticmethod
    def _calculate_placement_height(
        surface_y: float,
        surface_height: float,
        block_height: float,
        arm_length_max: float
    ) -> float:
        """Calculate robot y-position needed to place block on a surface.

        Formula: Block bottom should touch surface top.
        - Block center = surface_y + surface_height + block_height/2
        - Robot gripper is at: robot_y - arm_length_max = block_center
        - Therefore: robot_y = surface_y + surface_height + block_height/2 + arm_length_max

        Args:
            surface_y: Y-position of surface center
            surface_height: Height of surface
            block_height: Height of block being placed
            arm_length_max: Maximum arm length (distance from robot to gripper)

        Returns:
            Target robot y-position for placement
        """
        return surface_y + surface_height + block_height/2 + arm_length_max

    @staticmethod
    def _compute_horizontal_overlap_percentage(
        block_x: float, block_y: float, block_width: float, block_height: float, block_theta: float,
        obs_x: float, obs_y: float, obs_width: float, obs_height: float, obs_theta: float
    ) -> float:
        """Compute what percentage of the held block's bottom face is covered by an obstruction's top face.

        Delegates to module-level function to avoid code duplication with Perceiver.
        """
        return compute_horizontal_overlap_percentage(
            block_x, block_y, block_width, block_height, block_theta,
            obs_x, obs_y, obs_width, obs_height, obs_theta
        )


class PickUpSkill(BaseDynObstruction2DSkill):
    """SLAP skill for picking up target block."""

    def _get_operator_name(self) -> str:
        return "PickUp"

    def _get_action_given_objects(self, objects: Sequence[Object], obs: NDArray[np.float32]) -> NDArray[np.float64]:
        # from tamp_improv.benchmarks.debug_physics import log_skill_action

        p = self._parse_obs(obs)

        # Determine which object to pick up from ground operator: PickUp(robot, obj)
        # objects[0] is robot, objects[1] is the object to pick up
        target_obj = objects[1]

        # Map object name to observation indices
        if target_obj.name == "target_block":
            obj_x, obj_y = p['block_x'], p['block_y']
            obj_width, obj_height = p['block_width'], p['block_height']
            obj_held = p['target_block_held']
        elif target_obj.name == "obstruction0":
            obj_x, obj_y = p['obs0_x'], p['obs0_y']
            obj_width, obj_height = p['obs0_width'], p['obs0_height']
            # Need to add obs held status to parser
            obj_held = bool(obs[29 + 7])  # obs0 held feature
        elif target_obj.name == "obstruction1":
            obj_x, obj_y = p['obs1_x'], p['obs1_y']
            obj_width, obj_height = p['obs1_width'], p['obs1_height']
            obj_held = bool(obs[44 + 7])  # obs1 held feature
        else:
            raise ValueError(f"Unknown object to pick up: {target_obj.name}")

        # PRIORITY: If we're holding the object but not at safe height, lift immediately
        # This ensures Phase 7 always executes before the skill completes
        if obj_held and p['robot_y'] < self.SAFE_Y - self.POSITION_TOL:
            action = np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
            # log_skill_action("PickUp", "7-Lift-PRIORITY", action, {
            #     "robot_y": p['robot_y'], "safe_y": self.SAFE_Y
            # })
            return action

        # Determine if we're in the descent/grasp phase
        angle_error = self._angle_diff(self.TARGET_THETA, p['robot_theta'])
        theta_aligned = abs(angle_error) <= self.POSITION_TOL
        arm_extended = abs(p['arm_joint'] - p['arm_length_max'] * 0.95) <= self.POSITION_TOL
        below_safe_height = p['robot_y'] < (self.SAFE_Y - self.POSITION_TOL)
        in_descent_phase = theta_aligned and arm_extended and below_safe_height

        # Phase 0: Move to safe height
        if not in_descent_phase and not np.isclose(p['robot_y'], self.SAFE_Y, atol=self.POSITION_TOL):
            action = np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
            # log_skill_action("PickUp", "0-SafeHeight", action, {
            #     "robot_y": p['robot_y'], "safe_y": self.SAFE_Y
            # })
            return action

        # Phase 1: Rotate gripper
        angle_error = self._angle_diff(self.TARGET_THETA, p['robot_theta'])
        if abs(angle_error) > self.POSITION_TOL:
            action = np.array([0, 0, np.clip(angle_error, -self.MAX_DTHETA, self.MAX_DTHETA), 0, 0], dtype=np.float64)
            # log_skill_action("PickUp", "1-Rotate", action, {
            #     "robot_theta": p['robot_theta'], "target_theta": self.TARGET_THETA, "error": angle_error
            # })
            return action

        # Phase 2: Extend arm
        target_arm = p['arm_length_max'] * 0.95
        if abs(p['arm_joint'] - target_arm) > self.POSITION_TOL:
            action = np.array([0, 0, 0, np.clip(target_arm - p['arm_joint'], -self.MAX_DARM, self.MAX_DARM), 0], dtype=np.float64)
            # log_skill_action("PickUp", "2-ExtendArm", action, {
            #     "arm_joint": p['arm_joint'], "target_arm": target_arm
            # })
            return action

        # Phase 3: Open gripper (only at start, before arm extension)
        # Once arm is extended, we never open gripper again (prevents re-opening after Phase 7 lift)
        at_safe_height = np.isclose(p['robot_y'], self.SAFE_Y, atol=self.POSITION_TOL)
        arm_not_extended = p['arm_joint'] < p['arm_length_max'] * 0.5  # Arm still retracted
        if at_safe_height and arm_not_extended and p['finger_gap'] < p['gripper_base_height'] - self.POSITION_TOL:
            action = np.array([0, 0, 0, 0, self.MAX_DGRIPPER], dtype=np.float64)
            # log_skill_action("PickUp", "3-OpenGripper", action, {
            #     "finger_gap": p['finger_gap'], "target_gap": p['gripper_base_height']
            # })
            return action

        # Phase 4: Move horizontally to object x (only before grasping)
        gripper_is_open = p['finger_gap'] > obj_width * 0.8
        if gripper_is_open and not np.isclose(p['robot_x'], obj_x, atol=self.POSITION_TOL):
            if not np.isclose(p['robot_y'], self.SAFE_Y, atol=self.POSITION_TOL):
                action = np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
                # log_skill_action("PickUp", "4a-ReturnToSafe", action, {
                #     "robot_y": p['robot_y'], "safe_y": self.SAFE_Y
                # })
                return action
            action = np.array([np.clip(obj_x - p['robot_x'], -self.MAX_DX, self.MAX_DX), 0, 0, 0, 0], dtype=np.float64)
            # log_skill_action("PickUp", "4-MoveToBlock", action, {
            #     "robot_x": p['robot_x'], "block_x": obj_x
            # })
            return action

        # Phase 5: Descend to object (only if gripper is still open - haven't grasped yet)
        gripper_is_open = p['finger_gap'] > obj_width * 0.8  # Still open if wider than object
        # Target: gripper fingers at object center height
        # robot_y - arm_length = obj_y + obj_height/2
        # Therefore: robot_y = obj_y + obj_height/2 + arm_length
        target_y = obj_y + obj_height/2 + p['arm_length_max']
        if gripper_is_open and p['robot_y'] > target_y + self.POSITION_TOL:
            # # Debug: Check if block is clipping through table
            # table_top_y = p['surface_y'] + p['surface_height']
            # block_bottom_y = p['block_y'] - p['block_height']/2
            # penetration = table_top_y - block_bottom_y
            # print(f"[PickUp] Phase 5-Descend:")
            # print(f"  robot_y={p['robot_y']:.3f}, target_y={target_y:.3f}")
            # print(f"  block_y={p['block_y']:.3f}, block_height={p['block_height']:.3f}")
            # print(f"  block_bottom_y={block_bottom_y:.3f}, table_top_y={table_top_y:.3f}")
            # if penetration > 0:
            #     print(f"  ⚠️  BLOCK PENETRATING TABLE BY {penetration:.3f}!")
            # else:
            #     print(f"  ✓ Block clearance from table: {-penetration:.3f}")

            action = np.array([0, np.clip(target_y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
            # log_skill_action("PickUp", "5-Descend", action, {
            #     "robot_y": p['robot_y'], "target_y": target_y, "block_y": p['block_y'], "block_height": p['block_height']
            # })
            return action

        # Phase 6: Close gripper (only after we've finished descending)
        target_y = obj_y + obj_height/2 + p['arm_length_max']
        at_grasp_position = abs(p['robot_y'] - target_y) <= self.POSITION_TOL
        if at_grasp_position and p['finger_gap'] > obj_width * 0.7:
            action = np.array([0, 0, 0, 0, -self.MAX_DGRIPPER], dtype=np.float64)
            # log_skill_action("PickUp", "6-CloseGripper", action, {
            #     "finger_gap": p['finger_gap'], "block_width": obj_width, "target": obj_width * 0.7
            # })
            return action

        # Phase 7: Lift with block
        if p['robot_y'] < self.SAFE_Y - self.POSITION_TOL:
            action = np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
            # log_skill_action("PickUp", "7-Lift", action, {
            #     "robot_y": p['robot_y'], "safe_y": self.SAFE_Y
            # })
            return action

        # log_skill_action("PickUp", "DONE", np.zeros(5, dtype=np.float64), {
        #     "robot_y": p['robot_y'], "robot_x": p['robot_x'], "block_x": p['block_x']
        # })
        return np.array([0, 0, 0, 0, 0], dtype=np.float64)


class PickUpFromTargetSkill(PickUpSkill):
    """SLAP skill for picking up blocking obstructions from target surface.

    Uses the same implementation as PickUpSkill since it handles any pickable object.
    """

    def _get_operator_name(self) -> str:
        return "PickUpFromTarget"

    def _get_action_given_objects(self, objects: Sequence[Object], obs: NDArray[np.float32]) -> NDArray[np.float64]:
        # Add debug logging
        target_obj = objects[1]
        print(f"[PickUpFromTarget] ENTRY: target_obj={target_obj.name}, type={target_obj.type.name}")

        # Call parent implementation
        action = super()._get_action_given_objects(objects, obs)

        p = self._parse_obs(obs)
        print(f"[PickUpFromTarget] robot_x={p['robot_x']:.3f}, robot_y={p['robot_y']:.3f}, finger_gap={p['finger_gap']:.3f}")
        if target_obj.name == "obstruction0":
            print(f"[PickUpFromTarget] obs0_x={p['obs0_x']:.3f}, obs0_y={p['obs0_y']:.3f}, obs0_held={bool(obs[29+7])}")
        elif target_obj.name == "obstruction1":
            print(f"[PickUpFromTarget] obs1_x={p['obs1_x']:.3f}, obs1_y={p['obs1_y']:.3f}, obs1_held={bool(obs[44+7])}")
        print(f"[PickUpFromTarget] action={action}")

        return action


class PlaceSkill(BaseDynObstruction2DSkill):
    """SLAP skill for placing to garbage zone."""

    def _get_operator_name(self) -> str:
        return "Place"

    def _get_action_given_objects(self, objects: Sequence[Object], obs: NDArray[np.float32]) -> NDArray[np.float64]:
        p = self._parse_obs(obs)

        # print(f"\n[Place] ENTRY: robot_x={p['robot_x']:.3f}, robot_y={p['robot_y']:.3f}, robot_theta={p['robot_theta']:.3f}")

        # Calculate placement height: use EXACT same surface parameters as PlaceOnTarget
        # since we're placing on the same table surface, just at garbage location
        placement_y = self._calculate_placement_height(
            surface_y=p['surface_y'],
            surface_height=p['surface_height'],
            block_height=p['block_height'],
            arm_length_max=p['arm_length_max']
        )

        # Ensure alignment - use shortest angular path
        angle_error = self._angle_diff(self.TARGET_THETA, p['robot_theta'])
        if abs(angle_error) > self.POSITION_TOL:
            # print(f"[Place] Phase 0: Aligning (angle_error={angle_error:.3f})")
            return np.array([0, 0, np.clip(angle_error, -self.MAX_DTHETA, self.MAX_DTHETA), 0, 0], dtype=np.float64)
        # To safe height (only if we haven't reached garbage x yet)
        not_at_garbage_x = not np.isclose(p['robot_x'], self.GARBAGE_X, atol=self.POSITION_TOL)
        if not_at_garbage_x and not np.isclose(p['robot_y'], self.SAFE_Y, atol=self.POSITION_TOL):
            # print(f"[Place] Phase 1: To safe height (robot_y={p['robot_y']:.3f}, SAFE_Y={self.SAFE_Y:.3f})")
            return np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
        # To garbage x
        if not_at_garbage_x:
            # print(f"[Place] Phase 2: To garbage x (robot_x={p['robot_x']:.3f}, GARBAGE_X={self.GARBAGE_X:.3f})")
            return np.array([np.clip(self.GARBAGE_X - p['robot_x'], -self.MAX_DX, self.MAX_DX), 0, 0, 0, 0], dtype=np.float64)
        # Phase 3: Descend to calculated placement height (only if still holding the block)
        still_holding = p['target_block_held']
        if still_holding and not np.isclose(p['robot_y'], placement_y, atol=self.POSITION_TOL):
            # print(f"[Place] Phase 3: Descending (robot_y={p['robot_y']:.3f}, placement_y={placement_y:.3f})")
            return np.array([0, np.clip(placement_y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
        # Phase 4: Open gripper (use same logic as PlaceOnTarget)
        if p['finger_gap'] < p['gripper_base_height'] * 0.95:
            # print(f"[Place] Phase 4: Opening gripper (finger_gap={p['finger_gap']:.3f}, target={p['gripper_base_height']:.3f})")
            return np.array([0, 0, 0, 0, self.MAX_DGRIPPER], dtype=np.float64)
        # Phase 5: Lift (only after block has been released)
        block_released = not p['target_block_held']
        if block_released and not np.isclose(p['robot_y'], self.SAFE_Y, atol=self.POSITION_TOL):
            # print(f"[Place] Phase 5: Lifting (robot_y={p['robot_y']:.3f}, SAFE_Y={self.SAFE_Y:.3f})")
            return np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
        # print(f"[Place] DONE")
        return np.zeros(5, dtype=np.float64)


class PlaceOnTargetSkill(BaseDynObstruction2DSkill):
    """SLAP skill for placing on target surface."""

    def _get_operator_name(self) -> str:
        return "PlaceOnTarget"

    def _get_action_given_objects(self, objects: Sequence[Object], obs: NDArray[np.float32]) -> NDArray[np.float64]:
        p = self._parse_obs(obs)

        print(f"\n[PlaceOnTarget] ENTRY: target_block_held={p['target_block_held']}, finger_gap={p['finger_gap']:.3f}, robot_y={p['robot_y']:.3f}")

        # Normal placement: Place at surface x-position
        # Trust that Clear(surface) precondition ensures surface is actually clear
        target_x = p['surface_x']
        target_y = self._calculate_placement_height(
            surface_y=p['surface_y'],
            surface_height=p['surface_height'],
            block_height=p['block_height'],
            arm_length_max=p['arm_length_max']
        )
        print(f"[PlaceOnTarget] Placing on target surface")
        print(f"  target_x={target_x:.3f} (surface_x), target_y={target_y:.3f}")
        print(f"  surface_y={p['surface_y']:.3f}, surface_height={p['surface_height']:.3f}")
        print(f"  block_height={p['block_height']:.3f}, arm_length_max={p['arm_length_max']:.3f}")

        # Phase 0: Ensure alignment - use shortest angular path
        angle_error = self._angle_diff(self.TARGET_THETA, p['robot_theta'])
        if abs(angle_error) > self.POSITION_TOL:
            print(f"[PlaceOnTarget] Phase 0: Rotating (error={angle_error:.3f})")
            action = np.array([0, 0, np.clip(angle_error, -self.MAX_DTHETA, self.MAX_DTHETA), 0, 0], dtype=np.float64)
            return action

        # Phase 1: To safe height (only if we haven't reached target x yet)
        # Once we're at target x, we proceed to descend and don't return to safe height
        not_at_target_x = not np.isclose(p['robot_x'], target_x, atol=self.POSITION_TOL)
        if not_at_target_x and not np.isclose(p['robot_y'], self.SAFE_Y, atol=self.POSITION_TOL):
            print(f"[PlaceOnTarget] Phase 1: Moving to safe height (robot_y={p['robot_y']:.3f}, SAFE_Y={self.SAFE_Y:.3f})")
            action = np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
            return action

        # Phase 2: To target x
        if not_at_target_x:
            print(f"[PlaceOnTarget] Phase 2: Moving to target_x (robot_x={p['robot_x']:.3f}, target_x={target_x:.3f})")
            action = np.array([np.clip(target_x - p['robot_x'], -self.MAX_DX, self.MAX_DX), 0, 0, 0, 0], dtype=np.float64)
            return action

        # Phase 3: Descend (only if gripper is still holding the object)
        still_holding = p['target_block_held']
        if still_holding and not np.isclose(p['robot_y'], target_y, atol=self.POSITION_TOL):
            block_center_y = p['robot_y'] - p['arm_length_max']
            block_bottom_y = block_center_y - p['block_height']/2
            table_top_y = p['surface_y'] + p['surface_height']
            actual_block_bottom_y = p['block_y'] - p['block_height']/2
            penetration = table_top_y - actual_block_bottom_y

            print(f"[PlaceOnTarget] Phase 3: Descending")
            print(f"  robot_y={p['robot_y']:.3f}, target_y={target_y:.3f}")
            print(f"  CALCULATED: block_center_y={block_center_y:.3f}, block_bottom_y={block_bottom_y:.3f}")
            print(f"  ACTUAL: block_y={p['block_y']:.3f}, block_bottom_y={actual_block_bottom_y:.3f}")
            print(f"  table_top_y={table_top_y:.3f}")
            if penetration > 0:
                print(f"  ⚠️  BLOCK PENETRATING TABLE BY {penetration:.3f}!")
            else:
                print(f"  ✓ Block clearance from table: {-penetration:.3f}")

            action = np.array([0, np.clip(target_y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
            return action

        # Phase 4: Open gripper (only if we're at target position and still holding)
        at_target_position = np.isclose(p['robot_y'], target_y, atol=self.POSITION_TOL)
        still_holding_at_target = at_target_position and p['target_block_held']
        # Open gripper if not yet fully open (check if finger_gap is less than fully open)
        if still_holding_at_target and p['finger_gap'] < p['gripper_base_height'] * 0.95:
            action = np.array([0, 0, 0, 0, self.MAX_DGRIPPER], dtype=np.float64)
            return action

        # Phase 5: Lift (only after block has been released)
        # Check if block is no longer held (not if gripper is opened)
        block_released = not p['target_block_held']
        if block_released and not np.isclose(p['robot_y'], self.SAFE_Y, atol=self.POSITION_TOL):
            print(f"[PlaceOnTarget] Phase 5: Lifting to safe height (robot_y={p['robot_y']:.3f}, SAFE_Y={self.SAFE_Y:.3f})")
            action = np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
            return action

        print(f"[PlaceOnTarget] DONE - Returning zeros (robot_y={p['robot_y']:.3f}, held={p['target_block_held']})")
        return np.zeros(5, dtype=np.float64)


class PushSkill(BaseDynObstruction2DSkill):
    """SLAP skill for pushing a block to make it accessible for picking."""

    # Push-specific constants
    PUSH_HEIGHT_OFFSET = 0.05  # How high above block center to push
    PUSH_DISTANCE = 0.3  # How far to push the block

    def _get_operator_name(self) -> str:
        return "Push"

    def _get_action_given_objects(self, objects: Sequence[Object], obs: NDArray[np.float32]) -> NDArray[np.float64]:
        """Push block away from wall or obstacle to make it graspable.

        Strategy:
        1. Move to safe height
        2. Rotate gripper down (theta = -π/2)
        3. Extend arm
        4. Move horizontally to block
        5. Descend to block height (slightly above center)
        6. Push horizontally toward target surface
        7. Retract and lift
        """
        p = self._parse_obs(obs)

        # Calculate push target: move block toward target surface
        push_direction = np.sign(p['surface_x'] - p['block_x'])
        if abs(push_direction) < 0.01:
            push_direction = 1.0  # Default to right if directly aligned

        target_push_x = p['block_x'] + push_direction * self.PUSH_DISTANCE

        # Phase 0: Move to safe height
        if not np.isclose(p['robot_y'], self.SAFE_Y, atol=self.POSITION_TOL):
            return np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)

        # Phase 1: Rotate gripper down
        angle_error = self._angle_diff(self.TARGET_THETA, p['robot_theta'])
        if abs(angle_error) > self.POSITION_TOL:
            return np.array([0, 0, np.clip(angle_error, -self.MAX_DTHETA, self.MAX_DTHETA), 0, 0], dtype=np.float64)

        # Phase 2: Extend arm
        target_arm = p['arm_length_max'] * 0.95
        if abs(p['arm_joint'] - target_arm) > self.POSITION_TOL:
            return np.array([0, 0, 0, np.clip(target_arm - p['arm_joint'], -self.MAX_DARM, self.MAX_DARM), 0], dtype=np.float64)

        # Phase 3: Close gripper (to make a solid pushing surface)
        if p['finger_gap'] > self.POSITION_TOL:
            return np.array([0, 0, 0, 0, -self.MAX_DGRIPPER], dtype=np.float64)

        # Phase 4: Move horizontally to approach block from opposite side of target
        approach_x = p['block_x'] - push_direction * 0.15  # Approach from behind
        if not np.isclose(p['robot_x'], approach_x, atol=self.POSITION_TOL):
            return np.array([np.clip(approach_x - p['robot_x'], -self.MAX_DX, self.MAX_DX), 0, 0, 0, 0], dtype=np.float64)

        # Phase 5: Descend to push height (slightly above block center)
        push_height = p['block_y'] + p['block_height']/2 + self.PUSH_HEIGHT_OFFSET + p['arm_length_max']
        if p['robot_y'] > push_height + self.POSITION_TOL:
            return np.array([0, np.clip(push_height - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)

        # Phase 6: Push horizontally toward target
        if not np.isclose(p['robot_x'], target_push_x, atol=self.POSITION_TOL):
            return np.array([np.clip(target_push_x - p['robot_x'], -self.MAX_DX, self.MAX_DX), 0, 0, 0, 0], dtype=np.float64)

        # Phase 7: Lift back to safe height
        if not np.isclose(p['robot_y'], self.SAFE_Y, atol=self.POSITION_TOL):
            return np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)

        # Done
        return np.zeros(5, dtype=np.float64)


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
        env = DynObstruction2DEnv(
            num_obstructions=self._num_obstructions, render_mode=self._render_mode
        )

        # Apply debug patches to track collisions and state transitions
        try:
            from tamp_improv.benchmarks.physics_debug_patch import patch_prbench_for_debugging
            config = DynObstruction2DEnvConfig()
            world_bounds = (config.world_min_x, config.world_max_x, config.world_min_y, config.world_max_y)
            # Patch the environment (handles wrapper types internally)
            patch_prbench_for_debugging(env, world_bounds)
        except Exception as e:
            print(f"[WARNING] Failed to apply debug patches: {e}")

        return env

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
        pickable_obj = Variable("?obj", types_container.pickable)  # Parent type for PickUp
        surface = Variable("?surface", types_container.surface)

        # Create operators as a dict first (for easy pairing with controllers)
        pick_up_operator = LiftedOperator(
            "PickUp",
            [robot, block],  # Pick up target_block only (not obstructions)
            preconditions={
                predicates["GripperEmpty"]([robot]),
            },
            add_effects={
                predicates["Holding"]([robot, block]),
            },
            delete_effects={
                predicates["GripperEmpty"]([robot]),
            },
        )

        pick_up_from_target_operator = LiftedOperator(
            "PickUpFromTarget",
            [robot, obstruction, surface],  # Pick up blocking obstructions from target surface
            preconditions={
                predicates["GripperEmpty"]([robot]),
                predicates["Blocking"]([obstruction, surface]),  # MUST be blocking to pick up
            },
            add_effects={
                predicates["Holding"]([robot, obstruction]),
                predicates["Clear"]([surface]),  # Optimistic: claims surface is now clear
                # If other obstructions still blocking, Perceiver will NOT add Clear,
                # planner expectation ≠ reality, and replanning will be triggered
            },
            delete_effects={
                predicates["GripperEmpty"]([robot]),
                predicates["Blocking"]([obstruction, surface]),  # Remove blocking when picked up
            },
        )

        place_operator = LiftedOperator(
            "Place",
            [robot, obstruction, surface],  # Place obstruction in garbage zone
            preconditions={
                predicates["Holding"]([robot, obstruction]),
            },
            add_effects={
                predicates["GripperEmpty"]([robot]),
                # Note: We don't add Clear here - PickUpFromTarget already deleted Blocking
                # Clear will be computed by Perceiver if no more Blocking atoms exist
            },
            delete_effects={
                predicates["Holding"]([robot, obstruction]),
            },
        )

        place_on_target_operator = LiftedOperator(
            "PlaceOnTarget",
            [robot, block, surface],
            preconditions={
                predicates["Holding"]([robot, block]),
                predicates["Clear"]([surface]),
                # NOTE: VerticalPathClear removed from preconditions
                # The skill will check collision at runtime and fail if blocked
                # This allows the planner to try, fail, then replan with obstacle clearing
            },
            add_effects={
                predicates["On"]([block, surface]),
                predicates["GripperEmpty"]([robot]),
            },
            delete_effects={
                predicates["Holding"]([robot, block]),
                predicates["Clear"]([surface]),
            },
        )

        # push_operator = LiftedOperator(
        #     "Push",
        #     [robot, block],
        #     preconditions={
        #         predicates["GripperEmpty"]([robot]),
        #     },
        #     add_effects=set(),  # No symbolic effects - just changes block position
        #     delete_effects=set(),
        # )

        operators_dict = {
            "PickUp": pick_up_operator,
            "PickUpFromTarget": pick_up_from_target_operator,
            "Place": place_operator,
            "PlaceOnTarget": place_on_target_operator,
            # "Push": push_operator,  # Commented out for now
        }

        return PlanningComponents(
            types=types_set,
            predicate_container=predicates,
            operators=set(operators_dict.values()),
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
        # Add SLAP skills
        system.components.skills.update({
            PickUpSkill(system.components),  # type: ignore
            PickUpFromTargetSkill(system.components),  # type: ignore
            PlaceSkill(system.components),  # type: ignore
            PlaceOnTargetSkill(system.components),  # type: ignore
            # PushSkill(system.components),  # type: ignore  # Commented out for now
        })
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
        # Add SLAP skills
        system.components.skills.update({
            PickUpSkill(system.components),  # type: ignore
            PickUpFromTargetSkill(system.components),  # type: ignore
            PlaceSkill(system.components),  # type: ignore
            PlaceOnTargetSkill(system.components),  # type: ignore
            # PushSkill(system.components),  # type: ignore  # Commented out for now
        })
        return system
