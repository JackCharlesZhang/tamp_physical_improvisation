"""DynObstruction2D environment implementation with physics-based manipulation."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Sequence, Union

import gymnasium as gym
import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
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
import tomsgeoms2d.structs
if not hasattr(tomsgeoms2d.structs, "Tobject"):
    # Create a dummy Tobject class to satisfy prbench imports
    from tomsgeoms2d.structs import Lobject
    tomsgeoms2d.structs.Tobject = Lobject  # Use Lobject as a stand-in


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

        # Use prbench's type system
        self.robot = KinRobotType
        self.block = DynRectangleType  # target block (dynamic)
        self.obstruction = DynRectangleType  # obstruction blocks (dynamic)
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


class Dynamic2dRobotController(GroundParameterizedController, abc.ABC):
    """Base controller for dynamic 2D robot manipulation tasks.

    Similar to Geom2dRobotController but adapted for dynamic physics environments.
    Uses waypoint-based planning to generate action sequences.
    """

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: Box,
        safe_y: float = 1.0,
    ) -> None:
        """Initialize controller.

        Args:
            objects: Sequence of objects involved in the skill
            action_space: Box action space with bounds for [dx, dy, dtheta, darm, dgripper]
            safe_y: Safe height for horizontal navigation
        """
        super().__init__(objects)
        self._robot = objects[0]
        self._action_space = action_space
        self._safe_y = safe_y

        # Extract max deltas from action space bounds
        self._max_delta_x = float(action_space.high[0])
        self._max_delta_y = float(action_space.high[1])
        self._max_delta_theta = float(action_space.high[2])
        self._max_delta_arm = float(action_space.high[3])
        self._max_delta_gripper = float(action_space.high[4])

        # State tracking
        self._current_params: Union[NDArray[np.float32], float] = 0.0
        self._current_plan: Union[list[NDArray[np.float32]], None] = None
        self._current_state: Union[ObjectCentricState, None] = None

    @abc.abstractmethod
    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[float, float, float, float, float]]:
        """Generate waypoints as (x, y, theta, arm, gripper) tuples."""

    def _waypoints_to_plan(
        self,
        state: ObjectCentricState,
        waypoints: list[tuple[float, float, float, float, float]],
    ) -> list[NDArray[np.float32]]:
        """Convert waypoints to discrete action sequence.

        Each waypoint is (x, y, theta, arm, gripper).
        Returns a list of actions [dx, dy, dtheta, darm, dgripper].
        """
        # Get current robot state
        curr_x = state.get(self._robot, "x")
        curr_y = state.get(self._robot, "y")
        curr_theta = state.get(self._robot, "theta")
        curr_arm = state.get(self._robot, "arm_joint")
        curr_gripper = state.get(self._robot, "finger_gap")

        current_pose = (curr_x, curr_y, curr_theta, curr_arm, curr_gripper)
        all_waypoints = [current_pose] + waypoints

        plan: list[NDArray[np.float32]] = []

        for start, end in zip(all_waypoints[:-1], all_waypoints[1:]):
            # Calculate total deltas
            total_dx = end[0] - start[0]
            total_dy = end[1] - start[1]
            total_dtheta = end[2] - start[2]

            # Handle angle wrapping for shortest path
            if abs(total_dtheta) > np.pi:
                if total_dtheta > 0:
                    total_dtheta -= 2 * np.pi
                else:
                    total_dtheta += 2 * np.pi

            total_darm = end[3] - start[3]
            total_dgripper = end[4] - start[4]

            # Skip if already at waypoint
            if np.allclose(
                [total_dx, total_dy, total_dtheta, total_darm, total_dgripper],
                [0, 0, 0, 0, 0],
                atol=1e-4,
            ):
                continue

            # Calculate number of steps needed (based on max deltas)
            num_steps = int(
                max(
                    np.ceil(abs(total_dx) / self._max_delta_x),
                    np.ceil(abs(total_dy) / self._max_delta_y),
                    np.ceil(abs(total_dtheta) / self._max_delta_theta),
                    np.ceil(abs(total_darm) / self._max_delta_arm),
                    np.ceil(abs(total_dgripper) / self._max_delta_gripper),
                    1,  # At least 1 step
                )
            )

            # Create action for each step
            dx = total_dx / num_steps
            dy = total_dy / num_steps
            dtheta = total_dtheta / num_steps
            darm = total_darm / num_steps
            dgripper = total_dgripper / num_steps

            action = np.array([dx, dy, dtheta, darm, dgripper], dtype=np.float64)

            # Debug: Check if action is within bounds
            import logging
            logger = logging.getLogger(__name__)
            if not self._action_space.contains(action):
                logger.error(f"Controller generated invalid action!")
                logger.error(f"  Waypoint: {i} -> {i+1}")
                logger.error(f"  Start: {start}")
                logger.error(f"  End: {end}")
                logger.error(f"  Deltas: dx={total_dx:.4f}, dy={total_dy:.4f}, dtheta={total_dtheta:.4f}")
                logger.error(f"  Num steps: {num_steps}")
                logger.error(f"  Action: {action}")
                logger.error(f"  Max deltas: dx={self._max_delta_x}, dy={self._max_delta_y}")
                logger.error(f"  Action space: {self._action_space}")

            for _ in range(num_steps):
                plan.append(action)

        return plan

    def reset(
        self, x: ObjectCentricState, params: Union[NDArray[np.float32], float]
    ) -> None:
        """Reset the controller with new state and parameters."""
        self._current_params = params
        self._current_plan = None
        self._current_state = x

    def terminated(self) -> bool:
        """Check if the controller has finished executing its plan."""
        return self._current_plan is not None and len(self._current_plan) == 0

    def step(self) -> NDArray[np.float32]:
        """Execute the next action in the controller's plan."""
        assert self._current_state is not None
        if self._current_plan is None:
            self._current_plan = self._generate_plan(self._current_state)
        return self._current_plan.pop(0)

    def observe(self, x: ObjectCentricState) -> None:
        """Update the controller with a new observed state."""
        self._current_state = x

    def _generate_plan(self, x: ObjectCentricState) -> list[NDArray[np.float32]]:
        """Generate full action sequence from waypoints."""
        waypoints = self._generate_waypoints(x)
        return self._waypoints_to_plan(x, waypoints)


class GroundPickController(Dynamic2dRobotController):
    """Controller for picking up blocks.

    Waypoint sequence:
    1. Rotate to theta=0 (gripper alignment with blocks)
    2. Extend arm fully
    3. Move to safe height
    4. Move horizontally above block
    5. Descend to grasp height
    6. Close gripper
    7. Lift block
    """

    def __init__(self, objects: Sequence[Object], action_space: Box) -> None:
        """Initialize pick controller."""
        super().__init__(objects, action_space, safe_y=1.0)
        self._block = objects[1]  # Block to pick

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> float:
        """No parameters needed for basic pick - always grasp at center."""
        return 0.0

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[float, float, float, float, float]]:
        """Generate waypoints for picking up block."""
        # Get current robot state
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_theta = state.get(self._robot, "theta")
        arm_length = state.get(self._robot, "arm_length")
        gripper_base_height = state.get(self._robot, "gripper_base_height")

        # Get block state
        block_x = state.get(self._block, "x")
        block_y = state.get(self._block, "y")
        block_width = state.get(self._block, "width")

        # Target gripper positions
        target_theta = 0.0  # Align with blocks (always at theta=0)
        extended_arm = arm_length * 0.95
        closed_gripper = block_width * 0.7  # Close around block
        open_gripper = gripper_base_height * 1.0  # Fully open

        waypoints = [
            # Waypoint 1: Rotate to align gripper, extend arm, open gripper
            (robot_x, robot_y, target_theta, extended_arm, open_gripper),
            # Waypoint 2: Move to safe height (maintain orientation)
            (robot_x, self._safe_y, target_theta, extended_arm, open_gripper),
            # Waypoint 3: Move horizontally above block
            (block_x, self._safe_y, target_theta, extended_arm, open_gripper),
            # Waypoint 4: Descend to grasp height
            (block_x, block_y, target_theta, extended_arm, open_gripper),
            # Waypoint 5: Close gripper
            (block_x, block_y, target_theta, extended_arm, closed_gripper),
            # Waypoint 6: Lift block to safe height
            (block_x, self._safe_y, target_theta, extended_arm, closed_gripper),
        ]

        return waypoints


class GroundPlaceController(Dynamic2dRobotController):
    """Controller for placing blocks at arbitrary (x, y) coordinates.

    Parameters: [target_x, target_y] - where to place the block
    """

    def __init__(self, objects: Sequence[Object], action_space: Box) -> None:
        """Initialize place controller."""
        super().__init__(objects, action_space, safe_y=1.0)
        self._block = objects[1]  # Block being held

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> NDArray[np.float32]:
        """Sample random placement location on table.

        For obstruction clearing, we want to place away from target surface.
        Sample from table area excluding target surface region.
        """
        # Table bounds (hardcoded for dyn_obstruction2d environment)
        table_min_x = -2.0
        table_max_x = 2.0
        table_y = 0.2  # Placement height

        # Sample random x position
        target_x = rng.uniform(table_min_x, table_max_x)
        target_y = table_y

        return np.array([target_x, target_y], dtype=np.float32)

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[float, float, float, float, float]]:
        """Generate waypoints for placing block at parameterized location."""
        # Get target placement from parameters
        target_x, target_y = self._current_params  # type: ignore

        # Get current robot state
        robot_theta = state.get(self._robot, "theta")
        arm_length = state.get(self._robot, "arm_length")
        gripper_base_height = state.get(self._robot, "gripper_base_height")
        finger_gap = state.get(self._robot, "finger_gap")

        # Gripper positions
        target_theta = 0.0
        extended_arm = arm_length * 0.95
        open_gripper = gripper_base_height * 1.0

        waypoints = [
            # Waypoint 1: Move to safe height (maintain current x)
            (state.get(self._robot, "x"), self._safe_y, target_theta, extended_arm, finger_gap),
            # Waypoint 2: Move horizontally to target x
            (target_x, self._safe_y, target_theta, extended_arm, finger_gap),
            # Waypoint 3: Descend to placement height
            (target_x, target_y, target_theta, extended_arm, finger_gap),
            # Waypoint 4: Open gripper to release
            (target_x, target_y, target_theta, extended_arm, open_gripper),
            # Waypoint 5: Lift away
            (target_x, self._safe_y, target_theta, extended_arm, open_gripper),
        ]

        return waypoints


class GroundPlaceOnTargetController(Dynamic2dRobotController):
    """Controller for placing blocks on the target surface.

    Parameters: float in [0, 1] - relative x position on target surface
    """

    def __init__(self, objects: Sequence[Object], action_space: Box) -> None:
        """Initialize place-on-target controller."""
        super().__init__(objects, action_space, safe_y=1.0)
        self._block = objects[1]  # Block being held
        self._surface = objects[2]  # Target surface

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> float:
        """Sample random relative x position on target surface [0, 1]."""
        return float(rng.uniform(0.2, 0.8))  # Avoid edges

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[float, float, float, float, float]]:
        """Generate waypoints for placing block on target surface."""
        # Get surface state
        surface_x = state.get(self._surface, "x")
        surface_y = state.get(self._surface, "y")
        surface_width = state.get(self._surface, "width")
        surface_height = state.get(self._surface, "height")

        # Get block state
        block_height = state.get(self._block, "height")

        # Calculate target position from parameter (relative position on surface)
        relative_x = self._current_params  # type: ignore # float in [0, 1]
        target_x = surface_x - surface_width / 2 + relative_x * surface_width

        # Place block on top of surface
        target_y = surface_y + surface_height / 2 + block_height / 2

        # Get current robot state
        arm_length = state.get(self._robot, "arm_length")
        gripper_base_height = state.get(self._robot, "gripper_base_height")
        finger_gap = state.get(self._robot, "finger_gap")

        # Gripper positions
        target_theta = 0.0
        extended_arm = arm_length * 0.95
        open_gripper = gripper_base_height * 1.0

        waypoints = [
            # Waypoint 1: Move to safe height
            (state.get(self._robot, "x"), self._safe_y, target_theta, extended_arm, finger_gap),
            # Waypoint 2: Move horizontally to surface
            (target_x, self._safe_y, target_theta, extended_arm, finger_gap),
            # Waypoint 3: Descend to placement height
            (target_x, target_y, target_theta, extended_arm, finger_gap),
            # Waypoint 4: Open gripper to release
            (target_x, target_y, target_theta, extended_arm, open_gripper),
            # Waypoint 5: Lift away
            (target_x, self._safe_y, target_theta, extended_arm, open_gripper),
        ]

        return waypoints


def create_parameterized_skills(
    types_container: DynObstruction2DTypes,
    operators: dict[str, LiftedOperator],
    action_space: Box,
) -> dict[str, LiftedSkill]:
    """Create lifted skills (operator + controller pairs) for DynObstruction2D.

    Args:
        types_container: Container with PDDL types
        operators: Dictionary mapping operator names to LiftedOperator instances
        action_space: The environment's action space (must match env.action_space)

    Returns:
        Dictionary mapping skill names to LiftedSkill instances
    """

    # Define parameter spaces
    pick_params_space = Box(
        low=np.array([0.0]),
        high=np.array([1.0]),
        dtype=np.float32,
    )

    place_params_space = Box(
        low=np.array([-2.0, 0.0]),  # [min_x, min_y]
        high=np.array([2.0, 2.0]),  # [max_x, max_y]
        dtype=np.float32,
    )

    place_on_target_params_space = Box(
        low=np.array([0.0]),
        high=np.array([1.0]),
        dtype=np.float32,
    )

    # Create partial controller classes with pre-configured action space
    class PickController(GroundPickController):
        """Pick controller with pre-configured action space."""

        def __init__(self, objects: Sequence[Object]) -> None:
            super().__init__(objects, action_space)

    class PlaceController(GroundPlaceController):
        """Place controller with pre-configured action space."""

        def __init__(self, objects: Sequence[Object]) -> None:
            super().__init__(objects, action_space)

    class PlaceOnTargetController(GroundPlaceOnTargetController):
        """Place on target controller with pre-configured action space."""

        def __init__(self, objects: Sequence[Object]) -> None:
            super().__init__(objects, action_space)

    # Create variables for lifted controllers
    robot = Variable("?robot", types_container.robot)
    block = Variable("?block", types_container.block)
    obstruction = Variable("?obstruction", types_container.obstruction)
    surface = Variable("?surface", types_container.surface)

    # Create lifted controllers
    # Note: params_space is not passed to LiftedParameterizedController
    # It's defined in the ground controller's sample_parameters() method

    # PickUp can work on both block and obstruction types
    pick_controller = LiftedParameterizedController(
        variables=[robot, block],
        controller_cls=PickController,
    )

    # Place can work on any block, places at arbitrary location
    place_controller = LiftedParameterizedController(
        variables=[robot, block],
        controller_cls=PlaceController,
    )

    # PlaceOnTarget specifically places on target surface
    place_on_target_controller = LiftedParameterizedController(
        variables=[robot, block, surface],
        controller_cls=PlaceOnTargetController,
    )

    # Create LiftedSkill objects (operator + controller pairs)
    # The operator parameters must match the controller variables
    lifted_skills = {
        "PickUp": LiftedSkill(
            operator=operators["PickUp"],
            controller=pick_controller,
        ),
        "Place": LiftedSkill(
            operator=operators["Place"],
            controller=place_controller,
        ),
        "PlaceOnTarget": LiftedSkill(
            operator=operators["PlaceOnTarget"],
            controller=place_on_target_controller,
        ),
    }

    return lifted_skills


def create_state_abstractor(
    perceiver: DynObstruction2DPerceiver,
) -> callable:
    """Create state abstractor function for SESAME planner.

    Converts ObjectCentricState to RelationalAbstractState (PDDL atoms).
    """
    def state_abstractor(state: ObjectCentricState) -> RelationalAbstractState:
        """Convert object-centric state to relational abstract state."""
        # Extract atoms from the state using the perceiver
        # The perceiver expects observations, but we need to work with ObjectCentricState
        # For now, we'll extract atoms directly from the state

        # Get all objects from the state
        objects = set(state.data.keys())

        # Extract PDDL atoms by examining the state
        # This requires accessing the perceiver's predicate extraction logic
        atoms: set[GroundAtom] = set()

        # Get predicates from perceiver
        predicates = perceiver.predicates

        # Check gripper status (Holding vs GripperEmpty)
        robot = perceiver._robot
        target_block = perceiver._target_block

        # Helper to get feature value from state
        def get_feature(obj: Object, feature_name: str) -> float:
            """Get a specific feature value from the state."""
            if obj not in state.data:
                return 0.0
            feature_idx = type_features[obj.type].index(feature_name)
            return float(state.data[obj][feature_idx])

        # Get type_features from the state (assumes it's already set)
        type_features = state.type_features

        # Check gripper status
        if target_block in state.data:
            held = get_feature(target_block, "held")
            finger_gap = get_feature(robot, "finger_gap")

            if held > 0.5 and finger_gap < 0.1:
                atoms.add(predicates["Holding"]([robot, target_block]))
            else:
                atoms.add(predicates["GripperEmpty"]([robot]))
        else:
            atoms.add(predicates["GripperEmpty"]([robot]))

        # Check On predicate (target_block on target_surface)
        target_surface = perceiver._target_surface
        if target_block in state.data and target_surface in state.data:
            # Extract geometric features
            block_x = get_feature(target_block, "x")
            block_y = get_feature(target_block, "y")
            block_theta = get_feature(target_block, "theta")
            block_width = get_feature(target_block, "width")
            block_height = get_feature(target_block, "height")

            surface_x = get_feature(target_surface, "x")
            surface_y = get_feature(target_surface, "y")
            surface_theta = get_feature(target_surface, "theta")
            surface_width = get_feature(target_surface, "width")
            surface_height = get_feature(target_surface, "height")

            # Use perceiver's geometric checking method
            if perceiver._is_on_surface(
                block_x, block_y, block_theta, block_width, block_height,
                surface_x, surface_y, surface_theta, surface_width, surface_height,
            ):
                atoms.add(predicates["On"]([target_block, target_surface]))

        # Check obstruction predicates
        for obstruction in perceiver._obstructions:
            if obstruction in state.data and target_surface in state.data:
                obs_x = get_feature(obstruction, "x")
                obs_y = get_feature(obstruction, "y")

                surface_x = get_feature(target_surface, "x")
                surface_y = get_feature(target_surface, "y")
                surface_width = get_feature(target_surface, "width")
                surface_height = get_feature(target_surface, "height")

                # Use perceiver's geometric checking method
                if perceiver._is_obstructing(
                    obs_x, obs_y,
                    surface_x, surface_y, surface_width, surface_height,
                ):
                    atoms.add(predicates["Obstructing"]([obstruction, target_surface]))
                else:
                    atoms.add(predicates["ObstructionClear"]([obstruction, target_surface]))

        # Check Clear predicate for surface
        if target_surface in state.data:
            # Surface is Clear if no obstructions are blocking it
            surface_obstructed = any(
                predicates["Obstructing"]([obs, target_surface]) in atoms
                for obs in perceiver._obstructions
            )
            if not surface_obstructed:
                atoms.add(predicates["Clear"]([target_surface]))

        return RelationalAbstractState(atoms=atoms, objects=objects)

    return state_abstractor


def create_goal_deriver(
    perceiver: DynObstruction2DPerceiver,
    state_abstractor_fn: callable,
) -> callable:
    """Create goal deriver function for SESAME planner.

    Returns a function that extracts the goal from any state.
    """
    def goal_deriver(state: ObjectCentricState) -> RelationalAbstractGoal:
        """Derive relational goal from state.

        Goal: Place target block on target surface with empty gripper.
        """
        predicates = perceiver.predicates
        robot = perceiver._robot
        target_block = perceiver._target_block
        target_surface = perceiver._target_surface

        # Define goal atoms
        goal_atoms = {
            predicates["On"]([target_block, target_surface]),
            predicates["GripperEmpty"]([robot]),
        }

        return RelationalAbstractGoal(
            atoms=goal_atoms,
            state_abstractor=state_abstractor_fn,
        )

    return goal_deriver


def create_transition_fn(
    env: gym.Env,
    observation_to_state: callable,
) -> callable:
    """Create transition function for SESAME planner.

    This function simulates state transitions by resetting the environment
    to a given state and stepping forward.

    Args:
        env: The gymnasium environment
        observation_to_state: Function to convert observation (numpy array) to ObjectCentricState
    """
    # Get the environment's observation space to access constant_objects
    from relational_structs.spaces import ObjectCentricBoxSpace
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    env_objects = env.observation_space.constant_objects

    def transition_fn(state: Any, action: Any) -> Any:
        """Simulate state transition.

        Resets the environment to the given state, executes the action,
        and returns the resulting state.

        Args:
            state: State (either ObjectCentricState or vectorized numpy array)
            action: Action to execute

        Returns:
            Next state (same type as input state)
        """
        # Check if state is already ObjectCentricState or needs conversion
        if isinstance(state, np.ndarray):
            # Convert vectorized state to ObjectCentricState
            state_ocs = observation_to_state(state)
            is_vectorized = True
        else:
            # Already an ObjectCentricState
            state_ocs = state
            is_vectorized = False

        # Remap state to use environment's object instances
        # The state might have different object instances with the same names
        remapped_data = {}
        for env_obj in env_objects:
            # Find matching object in state by name
            state_obj = state_ocs.get_object_from_name(env_obj.name)
            remapped_data[env_obj] = state_ocs.data[state_obj].copy()

        # Create new state with environment's objects
        remapped_state = ObjectCentricState(remapped_data, state_ocs.type_features)

        # Reset environment to the remapped state
        env.reset(options={"init_state": remapped_state})

        # Execute action in the simulator
        # Debug: print action details if assertion fails
        import logging
        logger = logging.getLogger(__name__)
        if not env.action_space.contains(action):
            logger.error(f"Action validation failed!")
            logger.error(f"  Action: {action}")
            logger.error(f"  Action type: {type(action)}")
            logger.error(f"  Action dtype: {action.dtype if hasattr(action, 'dtype') else 'N/A'}")
            logger.error(f"  Action shape: {action.shape if hasattr(action, 'shape') else 'N/A'}")
            logger.error(f"  Action space: {env.action_space}")
            logger.error(f"  Action space low: {env.action_space.low}")
            logger.error(f"  Action space high: {env.action_space.high}")
            logger.error(f"  Action space dtype: {env.action_space.dtype}")
        obs, _, _, _, _ = env.step(action)

        # Return the new state in the same format as input
        # obs is already vectorized from env.step()
        if is_vectorized:
            return obs
        else:
            # Convert back to ObjectCentricState if input was ObjectCentricState
            return observation_to_state(obs)

    return transition_fn


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
    GARBAGE_X, GARBAGE_Y = 0.3, 0.15
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
            'block_x': obs[14], 'block_y': obs[15], 'block_width': obs[26], 'block_height': obs[27],
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


class PickUpSkill(BaseDynObstruction2DSkill):
    """SLAP skill for picking up target block."""

    def _get_operator_name(self) -> str:
        return "PickUp"

    def _get_action_given_objects(self, objects: Sequence[Object], obs: NDArray[np.float32]) -> NDArray[np.float64]:
        from tamp_improv.benchmarks.debug_physics import log_skill_action

        p = self._parse_obs(obs)

        # Determine if we're in the descent/grasp phase
        angle_error = self._angle_diff(self.TARGET_THETA, p['robot_theta'])
        theta_aligned = abs(angle_error) <= self.POSITION_TOL
        arm_extended = abs(p['arm_joint'] - p['arm_length_max'] * 0.95) <= self.POSITION_TOL
        below_safe_height = p['robot_y'] < (self.SAFE_Y - self.POSITION_TOL)
        in_descent_phase = theta_aligned and arm_extended and below_safe_height

        # Phase 0: Move to safe height
        if not in_descent_phase and not np.isclose(p['robot_y'], self.SAFE_Y, atol=self.POSITION_TOL):
            action = np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
            log_skill_action("PickUp", "0-SafeHeight", action, {
                "robot_y": p['robot_y'], "safe_y": self.SAFE_Y
            })
            return action

        # Phase 1: Rotate gripper
        angle_error = self._angle_diff(self.TARGET_THETA, p['robot_theta'])
        if abs(angle_error) > self.POSITION_TOL:
            action = np.array([0, 0, np.clip(angle_error, -self.MAX_DTHETA, self.MAX_DTHETA), 0, 0], dtype=np.float64)
            log_skill_action("PickUp", "1-Rotate", action, {
                "robot_theta": p['robot_theta'], "target_theta": self.TARGET_THETA, "error": angle_error
            })
            return action

        # Phase 2: Extend arm
        target_arm = p['arm_length_max'] * 0.95
        if abs(p['arm_joint'] - target_arm) > self.POSITION_TOL:
            action = np.array([0, 0, 0, np.clip(target_arm - p['arm_joint'], -self.MAX_DARM, self.MAX_DARM), 0], dtype=np.float64)
            log_skill_action("PickUp", "2-ExtendArm", action, {
                "arm_joint": p['arm_joint'], "target_arm": target_arm
            })
            return action

        # Phase 3: Open gripper (only at start, before arm extension)
        # Once arm is extended, we never open gripper again (prevents re-opening after Phase 7 lift)
        at_safe_height = np.isclose(p['robot_y'], self.SAFE_Y, atol=self.POSITION_TOL)
        arm_not_extended = p['arm_joint'] < p['arm_length_max'] * 0.5  # Arm still retracted
        if at_safe_height and arm_not_extended and p['finger_gap'] < p['gripper_base_height'] - self.POSITION_TOL:
            action = np.array([0, 0, 0, 0, self.MAX_DGRIPPER], dtype=np.float64)
            log_skill_action("PickUp", "3-OpenGripper", action, {
                "finger_gap": p['finger_gap'], "target_gap": p['gripper_base_height']
            })
            return action

        # Phase 4: Move horizontally to block x
        if not np.isclose(p['robot_x'], p['block_x'], atol=self.POSITION_TOL):
            if not np.isclose(p['robot_y'], self.SAFE_Y, atol=self.POSITION_TOL):
                action = np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
                log_skill_action("PickUp", "4a-ReturnToSafe", action, {
                    "robot_y": p['robot_y'], "safe_y": self.SAFE_Y
                })
                return action
            action = np.array([np.clip(p['block_x'] - p['robot_x'], -self.MAX_DX, self.MAX_DX), 0, 0, 0, 0], dtype=np.float64)
            log_skill_action("PickUp", "4-MoveToBlock", action, {
                "robot_x": p['robot_x'], "block_x": p['block_x']
            })
            return action

        # Phase 5: Descend to block (only if gripper is still open - haven't grasped yet)
        gripper_is_open = p['finger_gap'] > p['block_width'] * 0.8  # Still open if wider than block
        target_y = p['block_y'] + p['block_height']/2 + p['gripper_base_height']/2 + p['arm_length_max']
        if gripper_is_open and p['robot_y'] > target_y + self.POSITION_TOL:
            action = np.array([0, np.clip(target_y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
            log_skill_action("PickUp", "5-Descend", action, {
                "robot_y": p['robot_y'], "target_y": target_y, "block_y": p['block_y'], "block_height": p['block_height']
            })
            return action

        # Phase 6: Close gripper
        if p['finger_gap'] > p['block_width'] * 0.7:
            action = np.array([0, 0, 0, 0, -self.MAX_DGRIPPER], dtype=np.float64)
            log_skill_action("PickUp", "6-CloseGripper", action, {
                "finger_gap": p['finger_gap'], "block_width": p['block_width'], "target": p['block_width'] * 0.7
            })
            return action

        # Phase 7: Lift with block
        if p['robot_y'] < self.SAFE_Y - self.POSITION_TOL:
            action = np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
            log_skill_action("PickUp", "7-Lift", action, {
                "robot_y": p['robot_y'], "safe_y": self.SAFE_Y
            })
            return action

        log_skill_action("PickUp", "DONE", np.zeros(5, dtype=np.float64), {
            "robot_y": p['robot_y'], "robot_x": p['robot_x'], "block_x": p['block_x']
        })
        return np.array([0, 0, 0, 0, 0], dtype=np.float64)


class PlaceSkill(BaseDynObstruction2DSkill):
    """SLAP skill for placing to garbage zone."""

    def _get_operator_name(self) -> str:
        return "Place"

    def _get_action_given_objects(self, objects: Sequence[Object], obs: NDArray[np.float32]) -> NDArray[np.float64]:
        p = self._parse_obs(obs)
        # Ensure alignment - use shortest angular path
        angle_error = self._angle_diff(self.TARGET_THETA, p['robot_theta'])
        if abs(angle_error) > self.POSITION_TOL:
            return np.array([0, 0, np.clip(angle_error, -self.MAX_DTHETA, self.MAX_DTHETA), 0, 0], dtype=np.float64)
        # To safe height
        if not np.isclose(p['robot_y'], self.SAFE_Y, atol=self.POSITION_TOL):
            return np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
        # To garbage x
        if not np.isclose(p['robot_x'], self.GARBAGE_X, atol=self.POSITION_TOL):
            return np.array([np.clip(self.GARBAGE_X - p['robot_x'], -self.MAX_DX, self.MAX_DX), 0, 0, 0, 0], dtype=np.float64)
        # Descend
        if not np.isclose(p['robot_y'], self.GARBAGE_Y, atol=self.POSITION_TOL):
            return np.array([0, np.clip(self.GARBAGE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
        # Open gripper
        if p['finger_gap'] < p['gripper_base_height'] - self.POSITION_TOL:
            return np.array([0, 0, 0, 0, self.MAX_DGRIPPER], dtype=np.float64)
        # Lift
        if not np.isclose(p['robot_y'], self.SAFE_Y, atol=self.POSITION_TOL):
            return np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
        return np.zeros(5, dtype=np.float64)


class PlaceOnTargetSkill(BaseDynObstruction2DSkill):
    """SLAP skill for placing on target surface."""

    def _get_operator_name(self) -> str:
        return "PlaceOnTarget"

    def _get_action_given_objects(self, objects: Sequence[Object], obs: NDArray[np.float32]) -> NDArray[np.float64]:
        p = self._parse_obs(obs)
        # Clamp target_x to stay within safe bounds (table is x=-2.0 to 2.0, add margin for robot)
        WORLD_MIN_X, WORLD_MAX_X = -1.8, 1.8  # Safety margin from edges
        target_x = np.clip(p['surface_x'], WORLD_MIN_X, WORLD_MAX_X)
        target_y = p['surface_y'] + p['surface_height']/2 + p['block_height']/2
        # Ensure alignment - use shortest angular path
        angle_error = self._angle_diff(self.TARGET_THETA, p['robot_theta'])
        if abs(angle_error) > self.POSITION_TOL:
            return np.array([0, 0, np.clip(angle_error, -self.MAX_DTHETA, self.MAX_DTHETA), 0, 0], dtype=np.float64)
        # To safe height
        if not np.isclose(p['robot_y'], self.SAFE_Y, atol=self.POSITION_TOL):
            return np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
        # To target x
        if not np.isclose(p['robot_x'], target_x, atol=self.POSITION_TOL):
            return np.array([np.clip(target_x - p['robot_x'], -self.MAX_DX, self.MAX_DX), 0, 0, 0, 0], dtype=np.float64)
        # Descend
        if not np.isclose(p['robot_y'], target_y, atol=self.POSITION_TOL):
            return np.array([0, np.clip(target_y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
        # Open gripper
        if p['finger_gap'] < p['gripper_base_height'] - self.POSITION_TOL:
            return np.array([0, 0, 0, 0, self.MAX_DGRIPPER], dtype=np.float64)
        # Lift
        if not np.isclose(p['robot_y'], self.SAFE_Y, atol=self.POSITION_TOL):
            return np.array([0, np.clip(self.SAFE_Y - p['robot_y'], -self.MAX_DY, self.MAX_DY), 0, 0, 0], dtype=np.float64)
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
        try:
            from prbench.envs.dynamic2d.dyn_obstruction2d import DynObstruction2DEnv, DynObstruction2DEnvConfig
        except ImportError as e:
            raise ImportError(
                "DynObstruction2DEnv requires prbench to be installed. "
                "Install with: pip install -e 'path/to/prbench[dynamic2d]'"
            ) from e

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

    def create_sesame_models(self) -> SesameModels:
        """Create SesameModels for SESAME planner integration.

        Returns:
            SesameModels container with all necessary components for bilevel planning.
        """
        # Get the perceiver from planning components
        perceiver = self.components.perceiver
        assert isinstance(perceiver, DynObstruction2DPerceiver)

        # Create state abstractor function
        state_abstractor_fn = create_state_abstractor(perceiver)

        # Create goal deriver function
        goal_deriver_fn = create_goal_deriver(perceiver, state_abstractor_fn)

        # Create observation_to_state function
        # Get the objects for state conversion
        objects = [perceiver._robot, perceiver._target_block, perceiver._target_surface]
        objects.extend(perceiver._obstructions)

        def observation_to_state(obs: NDArray[np.float32]) -> ObjectCentricState:
            """Convert observation to ObjectCentricState."""
            return _obs_to_state(obs, objects)

        # Create transition function (needs observation_to_state for conversion)
        transition_fn = create_transition_fn(self.env, observation_to_state)

        # Create skills with the environment's action space
        # (Must happen after env is created to get correct action space)
        types_container = DynObstruction2DTypes()
        operators_dict = {op.name: op for op in self.components.operators}
        lifted_skills = create_parameterized_skills(
            types_container, operators_dict, self.env.action_space
        )

        # Create SesameModels
        sesame_models = SesameModels(
            observation_space=self.env.observation_space,
            state_space=self.env.observation_space,  # Using same space for now
            action_space=self.env.action_space,
            transition_fn=transition_fn,
            types=self.components.types,
            predicates=self.components.predicate_container.as_set(),
            observation_to_state=observation_to_state,
            state_abstractor=state_abstractor_fn,
            goal_deriver=goal_deriver_fn,
            skills=set(lifted_skills.values()),  # Set of LiftedSkill objects
        )

        return sesame_models

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

        # Create operators as a dict first (for easy pairing with controllers)
        pick_up_operator = LiftedOperator(
            "PickUp",
            [robot, block],
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

        place_operator = LiftedOperator(
            "Place",
            [robot, block],
            preconditions={
                predicates["Holding"]([robot, block]),
            },
            add_effects={
                predicates["GripperEmpty"]([robot]),
            },
            delete_effects={
                predicates["Holding"]([robot, block]),
            },
        )

        place_on_target_operator = LiftedOperator(
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
        )

        push_operator = LiftedOperator(
            "Push",
            [robot, block],
            preconditions={
                predicates["GripperEmpty"]([robot]),
            },
            add_effects=set(),  # No symbolic effects - just changes block position
            delete_effects=set(),
        )

        operators_dict = {
            "PickUp": pick_up_operator,
            "Place": place_operator,
            "PlaceOnTarget": place_on_target_operator,
            "Push": push_operator,
        }

        # NOTE: Skills will be created later after environment is initialized
        # so we can use the correct action space from the environment

        return PlanningComponents(
            types=types_set,
            predicate_container=predicates,
            operators=set(operators_dict.values()),
            skills=set(),  # Empty - will be populated after env creation
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
            PlaceSkill(system.components),  # type: ignore
            PlaceOnTargetSkill(system.components),  # type: ignore
            PushSkill(system.components),  # type: ignore
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
            PlaceSkill(system.components),  # type: ignore
            PlaceOnTargetSkill(system.components),  # type: ignore
            PushSkill(system.components),  # type: ignore
        })
        return system
