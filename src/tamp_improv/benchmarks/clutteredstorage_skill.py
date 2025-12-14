"""Skills for the ClutteredStorage2D environment."""

from typing import Optional, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from prbench.envs.geom2d.clutteredstorage2d import (
    ClutteredStorage2DEnvConfig,
    ShelfType,
    TargetBlockType,
)
from prbench.envs.geom2d.object_types import CRVRobotType
from prbench.envs.geom2d.structs import SE2Pose
from prbench.envs.geom2d.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
    get_tool_tip_position,
    run_motion_planning_for_crv_robot,
    snap_suctioned_objects,
    state_2d_has_collision,
)
from relational_structs import (
    GroundOperator,
    LiftedOperator,
    Object,
    ObjectCentricState,
    Variable,
)
from task_then_motion_planning.structs import LiftedOperatorSkill

from prbench_models.geom2d.utils import Geom2dRobotController
from tamp_improv.benchmarks.base import PlanningComponents


# Controllers - kept for internal use by skills
class GroundPickBlockNotOnShelfController(Geom2dRobotController):
    """Controller for picking a block that is initially not on the shelf."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
        max_resampling_attempts: int = 150,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._block = objects[1]
        self._shelf = objects[2]
        self._action_space = action_space
        self._max_resampling_attempts = max_resampling_attempts
        self._rng = np.random.default_rng()

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> tuple[float, float]:
        # Sample grasp ratio on the height of the block
        # <0.0: custom frame dx/dy < 0
        # >0.0: custom frame dx/dy > 0
        while True:
            grasp_ratio = rng.uniform(-1.0, 1.0)
            if grasp_ratio != 0.0:
                break
        max_arm_length = x.get(self._robot, "arm_length")
        min_arm_length = (
            x.get(self._robot, "base_radius")
            + x.get(self._robot, "gripper_width") / 2
            + 1e-4
        )
        arm_length = rng.uniform(min_arm_length, max_arm_length)
        return (grasp_ratio, arm_length)

    def _get_vacuum_actions(self) -> tuple[float, float]:
        return 0.0, 1.0

    def _calculate_grasp_robot_pose(self, state: ObjectCentricState) -> SE2Pose:
        """Calculate the actual grasp point based on ratio parameter."""
        if isinstance(self._current_params, tuple):
            grasp_ratio, arm_length = self._current_params
        else:
            raise ValueError("Expected tuple parameters for grasp ratio and arm length")

        # Get block properties and grasp frame
        block_x = state.get(self._block, "x")
        block_y = state.get(self._block, "y")
        block_theta = state.get(self._block, "theta")
        rel_point_dx = state.get(self._block, "width") / 2
        rel_point = SE2Pose(block_x, block_y, block_theta) * SE2Pose(
            rel_point_dx, 0.0, 0.0
        )

        # Relative SE2 pose w.r.t the grasp frame
        custom_dx = (
            state.get(self._block, "width") / 2
            + arm_length
            + state.get(self._robot, "gripper_width")
        )
        custom_dx *= -1 if grasp_ratio < 0 else 1  # Right or left side grasp
        # Custom dy is always positive.
        custom_dy = abs(grasp_ratio) * state.get(self._block, "height")
        custom_dtheta = 0.0 if grasp_ratio < 0 else np.pi
        custom_pose = SE2Pose(custom_dx, custom_dy, custom_dtheta)

        target_se2_pose = rel_point * custom_pose
        return target_se2_pose

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_theta = state.get(self._robot, "theta")
        robot_radius = state.get(self._robot, "base_radius")

        # Try resampling parameters until we find collision-free waypoints
        for attempt in range(self._max_resampling_attempts):
            # Sample or use provided parameters
            grasp_ratio, arm_length = self.sample_parameters(state, self._rng)
            self._current_params = (grasp_ratio, arm_length)

            # Calculate grasp point and robot target position
            target_se2_pose = self._calculate_grasp_robot_pose(state)
            if isinstance(self._current_params, tuple):
                _, desired_arm_length = self._current_params
            else:
                raise ValueError("Expected tuple parameters for grasp ratio and arm length")

            
            # Plan collision-free waypoints to the target pose
            # We set the arm to be the shortest length during motion planning
            mp_state = state.copy()
            mp_state.set(self._robot, "arm_joint", robot_radius)
            init_constant_state = self._init_constant_state
            if init_constant_state is not None:
                mp_state.data.update(init_constant_state.data)
            assert isinstance(self._action_space, CRVRobotActionSpace)
            collision_free_waypoints = run_motion_planning_for_crv_robot(
                mp_state, self._robot, target_se2_pose, self._action_space
            )
            # Always first make arm shortest to avoid collisions
            final_waypoints: list[tuple[SE2Pose, float]] = [
                (SE2Pose(robot_x, robot_y, robot_theta), robot_radius)
            ]

            if collision_free_waypoints is not None:
                for wp in collision_free_waypoints:
                    final_waypoints.append((wp, robot_radius))
                final_waypoints.append((target_se2_pose, desired_arm_length))

                print("controller found valid plan to pick block!!!!")
                return final_waypoints

        print("controller failed to find plan to pick block")
        return None

class GroundPlaceBlockOnShelfController(Geom2dRobotController):
    """Controller for placing a block onto the shelf."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
        max_resampling_attempts: int = 150,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._block = objects[1]
        self._shelf = objects[2]
        self._action_space = action_space
        self._max_resampling_attempts = max_resampling_attempts
        self._rng = np.random.default_rng()

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> tuple[float, float]:
        del x  # Unused
        # Sample place ratio
        # w.r.t (shelf_width - block_width)
        # and (shelf_height - block_height)
        relative_dx = rng.uniform(0.01, 0.99)
        # Bias towards inside the shelf
        relative_dy = rng.uniform(0.1, 0.95)
        return (relative_dx, relative_dy)

    def _get_vacuum_actions(self) -> tuple[float, float]:
        return 1.0, 0.0

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        """Generate waypoints with parameter resampling until motion planning succeeds."""
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_theta = state.get(self._robot, "theta")
        robot_radius = state.get(self._robot, "base_radius")
        robot_arm_length = state.get(self._robot, "arm_length")
        gripper_height = state.get(self._robot, "gripper_height")
        gripper_width = state.get(self._robot, "gripper_width")
        block_x = state.get(self._block, "x")
        block_y = state.get(self._block, "y")
        block_theta = state.get(self._block, "theta")
        block_width = state.get(self._block, "width")
        block_height = state.get(self._block, "height")
        block_curr_center = SE2Pose(block_x, block_y, block_theta) * SE2Pose(
            block_width / 2, block_height / 2, 0.0
        )
        _, gripper_to_block = get_suctioned_objects(state, self._robot)[0]

        # Shelf position
        shelf_x = state.get(self._shelf, "x1")
        shelf_width = state.get(self._shelf, "width1")
        shelf_y = state.get(self._shelf, "y1")
        shelf_height = state.get(self._shelf, "height1")

        # Determine grasp side
        gripper_x, gripper_y = get_tool_tip_position(state, self._robot)
        gripper_frame = SE2Pose(gripper_x, gripper_y, block_theta)
        relative_frame = block_curr_center.inverse * gripper_frame
        is_left_grasp = relative_frame.x < 0

        current_wp = (
            SE2Pose(robot_x, robot_y, robot_theta),
            state.get(self._robot, "arm_joint"),
        )

        # Try resampling parameters until we find collision-free waypoints
        for attempt in range(self._max_resampling_attempts):
            # Sample or use provided parameters
            relative_dx, relative_dy = self.sample_parameters(state, self._rng)
            
            # Calculate desired block position
            x_min = shelf_x + gripper_height / 2
            x_max = shelf_x + shelf_width - gripper_height / 2
            x_min = min(x_min, x_max)
            x_max = max(x_min, x_max)
            block_desired_x_center = x_min + (x_max - x_min) * relative_dx

            y_min = min(shelf_y + block_width / 2, shelf_y + shelf_height - block_width / 2)
            y_max = max(shelf_y + block_width / 2, shelf_y + shelf_height - block_width / 2)
            block_desired_y_center = y_min + (y_max - y_min) * relative_dy

            # Set block orientation based on grasp side
            if is_left_grasp:
                block_desired_center = SE2Pose(
                    block_desired_x_center, block_desired_y_center, np.pi / 2
                )
            else:
                block_desired_center = SE2Pose(
                    block_desired_x_center, block_desired_y_center, -np.pi / 2
                )

            gripper_final_desired_pose = (
                block_desired_center
                * SE2Pose(-block_width / 2, -block_height / 2, 0.0)
                * gripper_to_block.inverse
            )

            final_robot_y = gripper_final_desired_pose.y - robot_arm_length - gripper_width
            pre_place_robot_x = gripper_final_desired_pose.x
            pre_place_robot_y = final_robot_y - shelf_height
            pre_place_pose_0 = SE2Pose(pre_place_robot_x, pre_place_robot_y, np.pi / 2)

            # Try motion planning for first phase
            final_waypoints: list[tuple[SE2Pose, float]] = []
            mp_state = state.copy()
            init_constant_state = self._init_constant_state
            if init_constant_state is not None:
                mp_state.data.update(init_constant_state.data)
            assert isinstance(self._action_space, CRVRobotActionSpace)

            collision_free_waypoints_0 = run_motion_planning_for_crv_robot(
                mp_state, self._robot, pre_place_pose_0, self._action_space
            )

            if collision_free_waypoints_0 is None:
                continue

            for wp in collision_free_waypoints_0:
                final_waypoints.append((wp, robot_radius))

            # Stretch the arm to the desired position
            if collision_free_waypoints_0:
                last_wp = collision_free_waypoints_0[-1]
                final_waypoints.append((last_wp, robot_arm_length))

            # Try motion planning for second phase
            mp_state.set(self._robot, "x", pre_place_robot_x)
            mp_state.set(self._robot, "y", pre_place_robot_y)
            mp_state.set(self._robot, "theta", np.pi / 2)
            mp_state.set(self._robot, "arm_joint", robot_arm_length)
            if init_constant_state is not None:
                mp_state.data.update(init_constant_state.data)
            pre_place_pose_1 = SE2Pose(pre_place_robot_x, final_robot_y, np.pi / 2)

            collision_free_waypoints_1 = run_motion_planning_for_crv_robot(
                mp_state, self._robot, pre_place_pose_1, self._action_space
            )

            if collision_free_waypoints_1 is None:
                continue

            for wp in collision_free_waypoints_1:
                final_waypoints.append((wp, robot_arm_length))

            # Success! Return the waypoints

            print("controller found valid plan to get to shelf!!")
            return final_waypoints

        print("failed to place block on shelf")
        # All attempts failed
        return [current_wp]


class GroundPickBlockOnShelfController(GroundPickBlockNotOnShelfController):
    """Controller for grasping the block that is on the shelf.

    The grasping point is either on the up or bottom side of the block.
    """

    def _calculate_grasp_robot_pose(self, state: ObjectCentricState) -> SE2Pose:
        """Calculate the actual grasp point based on ratio parameter."""
        if isinstance(self._current_params, tuple):
            grasp_ratio, arm_length = self._current_params
        else:
            raise ValueError("Expected tuple parameters for grasp ratio and arm length")

        # Get block properties and grasp frame
        block_x = state.get(self._block, "x")
        block_y = state.get(self._block, "y")
        block_theta = state.get(self._block, "theta")
        rel_point_dy = state.get(self._block, "height") / 2
        rel_point = SE2Pose(block_x, block_y, block_theta) * SE2Pose(
            0.0, rel_point_dy, 0.0
        )

        # Relative SE2 pose w.r.t the grasp frame
        custom_dy = (
            state.get(self._block, "height") / 2
            + arm_length
            + state.get(self._robot, "gripper_width")
        )
        custom_dy *= -1 if grasp_ratio < 0 else 1  # top or bottom side grasp
        # Custom dx is always positive.
        custom_dx = abs(grasp_ratio) * state.get(self._block, "width")
        custom_dtheta = np.pi / 2 if grasp_ratio < 0 else -np.pi / 2
        custom_pose = SE2Pose(custom_dx, custom_dy, custom_dtheta)

        target_se2_pose = rel_point * custom_pose
        return target_se2_pose


class GroundPlaceBlockNotOnShelfController(Geom2dRobotController):
    """Controller for placing the block not on the shelf."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
        max_resampling_attempts: int = 150,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._block = objects[1]
        self._shelf = objects[2]
        self._action_space = action_space
        env_config = ClutteredStorage2DEnvConfig()
        self.world_x_min = env_config.world_min_x + env_config.robot_base_radius
        self.world_x_max = env_config.world_max_x - env_config.robot_base_radius
        self.world_y_min = env_config.world_min_y + env_config.robot_base_radius
        self.world_y_max = (
            env_config.world_max_y
            - env_config.shelf_height
            - env_config.robot_base_radius
        )
        self._max_resampling_attempts = max_resampling_attempts
        self._rng = np.random.default_rng()

    def _get_vacuum_actions(self) -> tuple[float, float]:
        return 1.0, 0.0

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> tuple[float, float, float]:
        # Sample place ratio
        # w.r.t (shelf_width - block_width)
        # and (shelf_height - block_height)
        abs_x = rng.uniform(self.world_x_min, self.world_x_max)
        abs_y = rng.uniform(self.world_y_min, self.world_y_max)
        abs_theta = rng.uniform(-np.pi, np.pi)
        rel_x = (abs_x - self.world_x_min) / (self.world_x_max - self.world_x_min)
        rel_y = (abs_y - self.world_y_min) / (self.world_y_max - self.world_y_min)
        rel_theta = (abs_theta + np.pi) / (2 * np.pi)

        return (rel_x, rel_y, rel_theta)

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_theta = state.get(self._robot, "theta")
        robot_radius = state.get(self._robot, "base_radius")

        for attempt in range(self._max_resampling_attempts):
            # Sample or use provided parameters
            rel_x, rel_y, rel_theta = self.sample_parameters(state, self._rng) 
            self._current_params = (rel_x, rel_y, rel_theta)
        
            # Calculate place position
            params = cast(tuple[float, ...], self._current_params)
            final_robot_x = (
                self.world_x_min + (self.world_x_max - self.world_x_min) * params[0]
            )
            final_robot_y = (
                self.world_y_min + (self.world_y_max - self.world_y_min) * params[1]
            )
            final_robot_theta = -np.pi + (2 * np.pi) * params[2]
            final_robot_pose = SE2Pose(final_robot_x, final_robot_y, final_robot_theta)

            current_wp = (
                SE2Pose(robot_x, robot_y, robot_theta),
                robot_radius,
            )

            # Plan collision-free waypoints to the target pose
            # We set the arm to be the longest during motion planning
            final_waypoints: list[tuple[SE2Pose, float]] = [current_wp]

            full_state = state.copy()
            init_constant_state = self._init_constant_state
            if init_constant_state is not None:
                full_state.data.update(init_constant_state.data)

            full_state.set(self._robot, "x", params[0])
            full_state.set(self._robot, "y", params[1])
            full_state.set(self._robot, "theta", params[2])
            suctioned_objects = get_suctioned_objects(state, self._robot)
            snap_suctioned_objects(full_state, self._robot, suctioned_objects)
            # Check end-pose collision
            moving_objects = {self._robot} | {o for o, _ in suctioned_objects}
            static_objects = set(full_state) - moving_objects
            if state_2d_has_collision(full_state, moving_objects, static_objects, {}):
                continue

            mp_state = state.copy()
            mp_state.set(self._robot, "arm_joint", robot_radius)
            init_constant_state = self._init_constant_state
            if init_constant_state is not None:
                mp_state.data.update(init_constant_state.data)
            assert isinstance(self._action_space, CRVRobotActionSpace)
            collision_free_waypoints_0 = run_motion_planning_for_crv_robot(
                mp_state, self._robot, final_robot_pose, self._action_space
            )
            if collision_free_waypoints_0 is None:
                continue
            for wp in collision_free_waypoints_0:
                final_waypoints.append((wp, robot_radius))

            return final_waypoints
        return None


# Base skill class
class BaseClutteredStorage2DSkill(
    LiftedOperatorSkill[NDArray[np.float32], NDArray[np.float32]]
):
    """Base class for cluttered storage 2D environment skills."""

    def __init__(
        self,
        components: PlanningComponents[NDArray[np.float32]],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        """Initialize skill."""
        super().__init__()
        self._components = components
        self._action_space = action_space
        self._init_constant_state = init_constant_state
        self._lifted_operator = self._get_lifted_operator()
        self._current_action_plan: list[NDArray[np.float32]] = []
        self._current_ground_operator: Optional[GroundOperator] = None
        # Store observation objects and type features for conversion
        self._observation_objects: list[Object] | None = None
        self._type_features: dict | None = None

    def set_observation_info(
        self, observation_objects: list[Object], type_features: dict
    ) -> None:
        """Set observation objects and type features for vec to state conversion."""
        self._observation_objects = observation_objects
        self._type_features = type_features

    def reset(self, ground_operator: GroundOperator) -> None:
        """Reset the skill with a ground operator."""
        self._current_ground_operator = ground_operator
        self._current_action_plan = []

    def _vec_to_state(self, vec: NDArray[np.float32]) -> ObjectCentricState:
        """Convert vector observation to ObjectCentricState."""
        if self._observation_objects is None or self._type_features is None:
            raise RuntimeError(
                "Skill not initialized with observation objects and type features. "
                "Call set_observation_info() first."
            )
        return ObjectCentricState.from_vec(
            vec, constant_objects=self._observation_objects, type_features=self._type_features
        )

    def get_action(self, obs: NDArray[np.float32]) -> NDArray[np.float32] | None:
        """Get action from vectorized observation."""
        # Convert vector to ObjectCentricState
        state = self._vec_to_state(obs)
        # Call parent's get_action with the converted state
        assert self._current_ground_operator is not None
        objects = self._current_ground_operator.parameters
        return self._get_action_given_objects(objects, state)

    def _get_lifted_operator(self) -> LiftedOperator:
        """Get the operator this skill implements."""
        return next(
            op
            for op in self._components.operators
            if op.name == self._get_operator_name()
        )

    @property
    def lifted_operator(self) -> LiftedOperator:
        """Property to access the lifted operator."""
        return self._lifted_operator

    def _get_operator_name(self) -> str:
        """Get the name of the operator this skill implements."""
        raise NotImplementedError

    def _waypoints_to_actions(
        self,
        controller: Geom2dRobotController,
        waypoints: list[tuple[SE2Pose, float]],
        current_state: ObjectCentricState,
    ) -> list[NDArray[np.float32]]:
        """Convert waypoints to a sequence of actions using the controller's method.

        Args:
            controller: The controller instance with _waypoints_to_plan method
            waypoints: List of (SE2Pose, arm_length) tuples
            current_state: Current state of the environment

        Returns:
            List of actions as numpy arrays
        """
        # Get vacuum actions from the controller
        vacuum_during_plan, vacuum_after_plan = controller._get_vacuum_actions()

        # Use the controller's built-in waypoint-to-plan conversion
        waypoint_plan = controller._waypoints_to_plan(
            current_state, waypoints, vacuum_during_plan
        )

        # Add plan suffix to change vacuum state at the end
        plan_suffix: list[NDArray[np.float32]] = [
            np.array([0, 0, 0, 0, vacuum_after_plan], dtype=np.float32),
        ]

        return waypoint_plan + plan_suffix


# Skill implementations with hardcoded parameters
class GraphPickBlockNotOnShelfSkill(BaseClutteredStorage2DSkill):
    """Skill for picking up blocks not on shelf."""

    def _get_operator_name(self) -> str:
        return "PickBlockNotOnShelf"

    def _get_action_given_objects(
        self, objects: Sequence[Object], obs: ObjectCentricState
    ) -> NDArray[np.float32] | None:
        # If we still have actions in the plan, return next one
        if self._current_action_plan:
            return self._current_action_plan.pop(0)

        # Otherwise, generate new plan with HARDCODED parameters
        action_plan = self._generate_action_plan(objects, obs)

        if action_plan is None:
            return None

        self._current_action_plan = action_plan
        if self._current_action_plan:
            return self._current_action_plan.pop(0)
        return None

    def _generate_action_plan(
        self, objects: Sequence[Object], state: ObjectCentricState
    ) -> list[NDArray[np.float32]] | None:
        """Generate action plan with parameter resampling for collision-free waypoints."""
        robot_obj = state.get_objects(CRVRobotType)[0]
        max_arm_length = state.get(robot_obj, "arm_length")
        min_arm_length = (
            state.get(robot_obj, "base_radius")
            + state.get(robot_obj, "gripper_width") / 2
            + 1e-4
        )

        # Create controller for parameter sampling
        controller = GroundPickBlockNotOnShelfController(
            objects, self._action_space, self._init_constant_state
        )

        # Sample parameters
        rng = np.random.default_rng()
        grasp_ratio, arm_length = controller.sample_parameters(state, rng)
        controller._current_params = (grasp_ratio, arm_length)

        # Calculate the target grasp pose
        try:
            target_pose = controller._calculate_grasp_robot_pose(state)
        except Exception:
            return None

        # Check if the target end waypoint would cause collision
        test_state = state.copy()
        if self._init_constant_state is not None:
            test_state.data.update(self._init_constant_state.data)
        test_state.set(robot_obj, "x", target_pose.x)
        test_state.set(robot_obj, "y", target_pose.y)
        test_state.set(robot_obj, "theta", target_pose.theta)
        test_state.set(robot_obj, "arm_joint", arm_length)

        # Check collision at target pose
        moving_objects = {robot_obj}
        static_objects = set(test_state) - moving_objects
        if state_2d_has_collision(test_state, moving_objects, static_objects, {}):
            return None  # Try next sample

        # Generate waypoints using existing logic
        try:
            waypoints = controller._generate_waypoints(state)
            # Convert waypoints to action sequence using controller's method
            return self._waypoints_to_actions(controller, waypoints, state)
        except Exception:
            return None  # Try next sample

        return None  # Failed to find valid plan after max attempts


class GraphPickBlockOnShelfSkill(BaseClutteredStorage2DSkill):
    """Skill for picking up blocks on shelf."""

    def _get_operator_name(self) -> str:
        return "PickBlockOnShelf"

    def _get_action_given_objects(
        self, objects: Sequence[Object], obs: ObjectCentricState
    ) -> NDArray[np.float32] | None:
        # If we still have actions in the plan, return next one
        if self._current_action_plan:
            return self._current_action_plan.pop(0)

        # Otherwise, generate new plan with HARDCODED parameters
        action_plan = self._generate_action_plan(objects, obs)

        if action_plan is None:
            return None

        self._current_action_plan = action_plan
        if self._current_action_plan:
            return self._current_action_plan.pop(0)
        return None

    def _generate_action_plan(
        self, objects: Sequence[Object], state: ObjectCentricState
    ) -> list[NDArray[np.float32]] | None:
        """Generate action plan with parameter resampling for collision-free waypoints."""
        robot_obj = state.get_objects(CRVRobotType)[0]
        max_arm_length = state.get(robot_obj, "arm_length")
        min_arm_length = (
            state.get(robot_obj, "base_radius")
            + state.get(robot_obj, "gripper_width") / 2
            + 1e-4
        )

        # Create controller for parameter sampling
        controller = GroundPickBlockOnShelfController(
            objects, self._action_space, self._init_constant_state
        )

        # Try multiple parameter samples to find collision-free waypoints
        rng = np.random.default_rng()
        grasp_ratio, arm_length = controller.sample_parameters(state, rng)
        controller._current_params = (grasp_ratio, arm_length)

        # Calculate the target grasp pose
        try:
            target_pose = controller._calculate_grasp_robot_pose(state)
        except Exception:
            return None

        # Check if the target end waypoint would cause collision
        test_state = state.copy()
        if self._init_constant_state is not None:
            test_state.data.update(self._init_constant_state.data)
        test_state.set(robot_obj, "x", target_pose.x)
        test_state.set(robot_obj, "y", target_pose.y)
        test_state.set(robot_obj, "theta", target_pose.theta)
        test_state.set(robot_obj, "arm_joint", arm_length)

        # Check collision at target pose
        moving_objects = {robot_obj}
        static_objects = set(test_state) - moving_objects
        if state_2d_has_collision(test_state, moving_objects, static_objects, {}):
            return None  # Try next sample

        # Generate waypoints using existing logic
        try:
            waypoints = controller._generate_waypoints(state)
            # Convert waypoints to action sequence using controller's method
            return self._waypoints_to_actions(controller, waypoints, state)
        except Exception:
            return None  # Try next sample

        return None  # Failed to find valid plan after max attempts


class GraphPlaceBlockOnShelfSkill(BaseClutteredStorage2DSkill):
    """Skill for placing blocks on shelf."""

    def _get_operator_name(self) -> str:
        return "PlaceBlockOnShelf"

    def _get_action_given_objects(
        self, objects: Sequence[Object], obs: ObjectCentricState
    ) -> NDArray[np.float32] | None:
        # If we still have actions in the plan, return next one
        if self._current_action_plan:
            return self._current_action_plan.pop(0)

        # Otherwise, generate new plan with HARDCODED parameters
        action_plan = self._generate_action_plan(objects, obs)

        if action_plan is None:
            return None

        self._current_action_plan = action_plan
        if self._current_action_plan:
            return self._current_action_plan.pop(0)
        return None

    def _generate_action_plan(
        self, objects: Sequence[Object], state: ObjectCentricState
    ) -> list[NDArray[np.float32]] | None:
        """Generate action plan with parameter resampling for collision-free waypoints."""
        robot_obj = state.get_objects(CRVRobotType)[0]

        # Create controller for parameter sampling
        controller = GroundPlaceBlockOnShelfController(
            objects, self._action_space, self._init_constant_state, 
        )


        # Generate waypoints using existing logic
        try:
            waypoints = controller._generate_waypoints(state)

            # Check if final waypoint would cause collision
            if waypoints:
                final_pose, final_arm_length = waypoints[-1]
                test_state = state.copy()
                if self._init_constant_state is not None:
                    test_state.data.update(self._init_constant_state.data)
                test_state.set(robot_obj, "x", final_pose.x)
                test_state.set(robot_obj, "y", final_pose.y)
                test_state.set(robot_obj, "theta", final_pose.theta)
                test_state.set(robot_obj, "arm_joint", final_arm_length)

                # Get suctioned objects and snap them to final position
                from prbench.envs.geom2d.utils import get_suctioned_objects, snap_suctioned_objects
                suctioned_objects = get_suctioned_objects(state, robot_obj)
                snap_suctioned_objects(test_state, robot_obj, suctioned_objects)

                # Check collision at target pose
                moving_objects = {robot_obj} | {o for o, _ in suctioned_objects}
                static_objects = set(test_state) - moving_objects

                if state_2d_has_collision(test_state, moving_objects, static_objects, {}):
                    return None 

            # Convert waypoints to action sequence using controller's method
            return self._waypoints_to_actions(controller, waypoints, state)
        except Exception:
            return None  # Try next sample

        return None  # Failed to find valid plan after max attempts


class GraphPlaceBlockNotOnShelfSkill(BaseClutteredStorage2DSkill):
    """Skill for placing blocks not on shelf."""

    def _get_operator_name(self) -> str:
        return "PlaceBlockNotOnShelf"

    def _get_action_given_objects(
        self, objects: Sequence[Object], obs: ObjectCentricState
    ) -> NDArray[np.float32] | None:
        # If we still have actions in the plan, return next one
        if self._current_action_plan:
            return self._current_action_plan.pop(0)

        # Otherwise, generate new plan with HARDCODED parameters
        action_plan = self._generate_action_plan(objects, obs)

        if action_plan is None:
            return None

        self._current_action_plan = action_plan
        if self._current_action_plan:
            return self._current_action_plan.pop(0)
        return None

    def _generate_action_plan(
        self, objects: Sequence[Object], state: ObjectCentricState
    ) -> list[NDArray[np.float32]] | None:
        """Generate action plan with parameter resampling for collision-free waypoints."""
        robot_obj = state.get_objects(CRVRobotType)[0]

        # Create controller for parameter sampling
        controller = GroundPlaceBlockNotOnShelfController(
            objects, self._action_space, self._init_constant_state
        )

        # Try multiple parameter samples to find collision-free waypoints
        rng = np.random.default_rng()
        
        # Sample parameters
        rel_x, rel_y, rel_theta = controller.sample_parameters(state, rng)
        controller._current_params = (rel_x, rel_y, rel_theta)

        # Generate waypoints using existing logic
        # Note: GroundPlaceBlockNotOnShelfController already checks collision
        # in its _generate_waypoints method (lines 392-407)
        try:
            waypoints = controller._generate_waypoints(state)

            # If waypoints were successfully generated and have more than just
            # the initial pose, we found a valid plan
            if waypoints and len(waypoints) > 1:
                # Convert waypoints to action sequence using controller's method
                return self._waypoints_to_actions(controller, waypoints, state)
        except Exception:
            return None  # Try next sample

        return None  # Failed to find valid plan after max attempts
