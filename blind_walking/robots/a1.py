# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pybullet simulation of a Laikago robot."""
import copy
import math
import re
from typing import Tuple

import numba
import numpy as np
import pybullet as pyb  # pytype: disable=import-error
from blind_walking.envs import locomotion_gym_config
from blind_walking.robots import laikago_constants, laikago_motor, minitaur, robot_config

NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "FR_hip_joint",
    "FR_upper_joint",
    "FR_lower_joint",
    "FL_hip_joint",
    "FL_upper_joint",
    "FL_lower_joint",
    "RR_hip_joint",
    "RR_upper_joint",
    "RR_lower_joint",
    "RL_hip_joint",
    "RL_upper_joint",
    "RL_lower_joint",
]
INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.32]
JOINT_DIRECTIONS = np.ones(12)
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
KNEE_JOINT_OFFSET = 0.0
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array([HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
PI = math.pi

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.15
_DEFAULT_HIP_POSITIONS = (
    (0.17, -0.135, 0),
    (0.17, 0.13, 0),
    (-0.195, -0.135, 0),
    (-0.195, 0.13, 0),
)

# flake8: noqa

COM_OFFSET = -np.array([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = (
    np.array(
        [
            [0.183, -0.047, 0.0],
            [0.183, 0.047, 0.0],
            [-0.183, -0.047, 0.0],
            [-0.183, 0.047, 0.0],
        ]
    )
    + COM_OFFSET
)

ABDUCTION_P_GAIN = 100.0
ABDUCTION_D_GAIN = 1.0
HIP_P_GAIN = 100.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 100.0
KNEE_D_GAIN = 2.0

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([0, 0.9, -1.8] * NUM_LEGS)

HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
UPPER_NAME_PATTERN = re.compile(r"\w+_upper_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_lower_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_toe\d*")
IMU_NAME_PATTERN = re.compile(r"imu\d*")

URDF_FILENAME = "a1/a1.urdf"

_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3
_NORMAL_FORCE_FIELD_NUMBER = 9


@numba.jit(nopython=True, cache=True)
def foot_position_in_hip_frame_to_joint_angle(foot_position, l_hip_sign=1):
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    x, y, z = foot_position[0], foot_position[1], foot_position[2]
    theta_knee = -np.arccos((x ** 2 + y ** 2 + z ** 2 - l_hip ** 2 - l_low ** 2 - l_up ** 2) / (2 * l_low * l_up))
    l = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(theta_knee))
    theta_hip = np.arcsin(-x / l) - theta_knee / 2
    c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
    s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
    theta_ab = np.arctan2(s1, c1)
    return np.array([theta_ab, theta_hip, theta_knee])


@numba.jit(nopython=True, cache=True)
def foot_position_in_hip_frame(angles, l_hip_sign=1):
    theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    leg_distance = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(theta_knee))
    eff_swing = theta_hip + theta_knee / 2

    off_x_hip = -leg_distance * np.sin(eff_swing)
    off_z_hip = -leg_distance * np.cos(eff_swing)
    off_y_hip = l_hip

    off_x = off_x_hip
    off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
    off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
    return np.array([off_x, off_y, off_z])


@numba.jit(nopython=True, cache=True)
def analytical_leg_jacobian(leg_angles, leg_id):
    """
    Computes the analytical Jacobian.
    Args:
    ` leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
      l_hip_sign: whether it's a left (1) or right(-1) leg.
    """
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * (-1) ** (leg_id + 1)

    t1, t2, t3 = leg_angles[0], leg_angles[1], leg_angles[2]
    l_eff = np.sqrt(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * np.cos(t3))
    t_eff = t2 + t3 / 2
    J = np.zeros((3, 3))
    J[0, 0] = 0
    J[0, 1] = -l_eff * np.cos(t_eff)
    J[0, 2] = l_low * l_up * np.sin(t3) * np.sin(t_eff) / l_eff - l_eff * np.cos(t_eff) / 2
    J[1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
    J[1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
    J[1, 2] = -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
    J[2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
    J[2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
    J[2, 2] = l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2
    return J


@numba.jit(nopython=True, cache=True, parallel=True)
def foot_positions_in_base_frame(foot_angles):
    foot_angles = foot_angles.reshape((4, 3))
    foot_positions = np.zeros((4, 3))
    for i in range(4):
        foot_positions[i] = foot_position_in_hip_frame(foot_angles[i], l_hip_sign=(-1) ** (i + 1))
    return foot_positions + HIP_OFFSETS


def transform_to_rotated_frame(v: np.ndarray, theta: float):
    """Transform a 2D vector into a coordinate frame rotated anticlockwise by theta"""
    x, y = v
    xn = np.cos(theta) * x + np.sin(theta) * y
    yn = -np.sin(theta) * x + np.cos(theta) * y
    return np.array([xn, yn])


def world_frame_to_base_frame(v_world: np.ndarray, robot):
    """Transform a 2D displacement vector from world frame to robot base frame"""
    yaw = robot.GetTrueBaseRollPitchYaw()[2]
    return transform_to_rotated_frame(v_world, yaw)


def base_frame_to_world_frame(v_base: np.ndarray, robot):
    """Transform a 2D displacement vector from robot base frame to world frame"""
    neg_yaw = -robot.GetTrueBaseRollPitchYaw()[2]
    return transform_to_rotated_frame(v_base, neg_yaw)


def get_grid_coordinates(grid_unit, grid_size):
    """
    Returns:
      grid_size array of grid coordinates
    """
    kx = grid_size[0] / 2 - 0.5
    xvalues = np.linspace(-kx * grid_unit[0], kx * grid_unit[0], num=grid_size[0])
    ky = grid_size[1] / 2 - 0.5
    yvalues = np.linspace(-ky * grid_unit[1], ky * grid_unit[1], num=grid_size[1])
    xx, yy = np.meshgrid(xvalues, yvalues)
    coordinates = np.array(list(zip(xx.flatten(), yy.flatten())))
    return coordinates


def draw_debug_sphere(pybullet_client, position: Tuple[float, float, float], rgba_color=[1, 0, 0, 1], radius=0.02) -> int:
    ballShape = pybullet_client.createCollisionShape(shapeType=pybullet_client.GEOM_SPHERE, radius=radius)
    ball_id = pybullet_client.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=ballShape, basePosition=position, baseOrientation=[0, 0, 0, 1]
    )
    pybullet_client.changeVisualShape(ball_id, -1, rgbaColor=rgba_color)
    pybullet_client.setCollisionFilterGroupMask(ball_id, -1, 0, 0)
    return ball_id


class A1(minitaur.Minitaur):
    """A simulation for the Laikago robot."""

    # At high replanning frequency, inaccurate values of BODY_MASS/INERTIA
    # doesn't seem to matter much. However, these values should be better tuned
    # when the replan frequency is low (e.g. using a less beefy CPU).
    MPC_BODY_MASS = 108 / 9.8
    MPC_BODY_INERTIA = np.array((0.017, 0, 0, 0, 0.057, 0, 0, 0, 0.064)) * 4.0
    MPC_BODY_HEIGHT = 0.24
    MPC_VELOCITY_MULTIPLIER = 0.5
    ACTION_CONFIG = [
        locomotion_gym_config.ScalarField(name="FR_hip_motor", upper_bound=0.802851455917, lower_bound=-0.802851455917),
        locomotion_gym_config.ScalarField(name="FR_upper_joint", upper_bound=4.18879020479, lower_bound=-1.0471975512),
        locomotion_gym_config.ScalarField(
            name="FR_lower_joint",
            upper_bound=-0.916297857297,
            lower_bound=-2.69653369433,
        ),
        locomotion_gym_config.ScalarField(name="FL_hip_motor", upper_bound=0.802851455917, lower_bound=-0.802851455917),
        locomotion_gym_config.ScalarField(name="FL_upper_joint", upper_bound=4.18879020479, lower_bound=-1.0471975512),
        locomotion_gym_config.ScalarField(
            name="FL_lower_joint",
            upper_bound=-0.916297857297,
            lower_bound=-2.69653369433,
        ),
        locomotion_gym_config.ScalarField(name="RR_hip_motor", upper_bound=0.802851455917, lower_bound=-0.802851455917),
        locomotion_gym_config.ScalarField(name="RR_upper_joint", upper_bound=4.18879020479, lower_bound=-1.0471975512),
        locomotion_gym_config.ScalarField(
            name="RR_lower_joint",
            upper_bound=-0.916297857297,
            lower_bound=-2.69653369433,
        ),
        locomotion_gym_config.ScalarField(name="RL_hip_motor", upper_bound=0.802851455917, lower_bound=-0.802851455917),
        locomotion_gym_config.ScalarField(name="RL_upper_joint", upper_bound=4.18879020479, lower_bound=-1.0471975512),
        locomotion_gym_config.ScalarField(
            name="RL_lower_joint",
            upper_bound=-0.916297857297,
            lower_bound=-2.69653369433,
        ),
    ]

    def __init__(
        self,
        pybullet_client,
        urdf_filename=URDF_FILENAME,
        enable_clip_motor_commands=False,
        time_step=0.001,
        action_repeat=10,
        sensors=None,
        control_latency=0.002,
        on_rack=False,
        enable_action_interpolation=True,
        enable_action_filter=False,
        motor_control_mode=None,
        reset_time=1,
        allow_knee_contact=False,
    ):
        self._urdf_filename = urdf_filename
        self._allow_knee_contact = allow_knee_contact
        self._enable_clip_motor_commands = enable_clip_motor_commands

        motor_kp = [
            ABDUCTION_P_GAIN,
            HIP_P_GAIN,
            KNEE_P_GAIN,
            ABDUCTION_P_GAIN,
            HIP_P_GAIN,
            KNEE_P_GAIN,
            ABDUCTION_P_GAIN,
            HIP_P_GAIN,
            KNEE_P_GAIN,
            ABDUCTION_P_GAIN,
            HIP_P_GAIN,
            KNEE_P_GAIN,
        ]
        motor_kd = [
            ABDUCTION_D_GAIN,
            HIP_D_GAIN,
            KNEE_D_GAIN,
            ABDUCTION_D_GAIN,
            HIP_D_GAIN,
            KNEE_D_GAIN,
            ABDUCTION_D_GAIN,
            HIP_D_GAIN,
            KNEE_D_GAIN,
            ABDUCTION_D_GAIN,
            HIP_D_GAIN,
            KNEE_D_GAIN,
        ]

        super(A1, self).__init__(
            pybullet_client=pybullet_client,
            time_step=time_step,
            action_repeat=action_repeat,
            num_motors=NUM_MOTORS,
            dofs_per_leg=DOFS_PER_LEG,
            motor_direction=JOINT_DIRECTIONS,
            motor_offset=JOINT_OFFSETS,
            motor_overheat_protection=False,
            motor_control_mode=motor_control_mode,
            motor_model_class=laikago_motor.LaikagoMotorModel,
            sensors=sensors,
            motor_kp=motor_kp,
            motor_kd=motor_kd,
            control_latency=control_latency,
            on_rack=on_rack,
            enable_action_interpolation=enable_action_interpolation,
            enable_action_filter=enable_action_filter,
            reset_time=reset_time,
        )

    def _LoadRobotURDF(self):
        a1_urdf_path = self.GetURDFFile()
        if self._self_collision_enabled:
            self.quadruped = self._pybullet_client.loadURDF(
                a1_urdf_path,
                self._GetDefaultInitPosition(),
                self._GetDefaultInitOrientation(),
                flags=self._pybullet_client.URDF_USE_SELF_COLLISION,
            )
        else:
            self.quadruped = self._pybullet_client.loadURDF(
                a1_urdf_path,
                self._GetDefaultInitPosition(),
                self._GetDefaultInitOrientation(),
            )

    def _SettleDownForReset(self, default_motor_angles, reset_time):
        self.ReceiveObservation()
        if reset_time <= 0:
            return

        for _ in range(500):
            self._StepInternal(
                INIT_MOTOR_ANGLES,
                motor_control_mode=robot_config.MotorControlMode.POSITION,
            )

        if default_motor_angles is not None:
            num_steps_to_reset = int(reset_time / self.time_step)
            for _ in range(num_steps_to_reset):
                self._StepInternal(
                    default_motor_angles,
                    motor_control_mode=robot_config.MotorControlMode.POSITION,
                )

    def GetHipPositionsInBaseFrame(self):
        return _DEFAULT_HIP_POSITIONS

    def GetFootContacts(self):
        all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)

        contacts = [False, False, False, False]
        for contact in all_contacts:
            # Ignore self contacts
            if contact[_BODY_B_FIELD_NUMBER] == self.quadruped:
                continue
            try:
                toe_link_index = self._foot_link_ids.index(contact[_LINK_A_FIELD_NUMBER])
                contacts[toe_link_index] = True
            except ValueError:
                continue

        return contacts

    def GetFootForces(self):
        all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)

        contact_forces = [0] * NUM_LEGS
        for contact in all_contacts:
            # Ignore self contacts
            if contact[_BODY_B_FIELD_NUMBER] == self.quadruped:
                continue
            try:
                toe_link_index = self._foot_link_ids.index(contact[_LINK_A_FIELD_NUMBER])
                contact_forces[toe_link_index] = contact[_NORMAL_FORCE_FIELD_NUMBER]
            except ValueError:
                continue

        return contact_forces

    def GetFootDistancesToGround(self, max_detection_distance=5.0):
        distances = [0] * NUM_LEGS
        for foot_id in self._foot_link_ids:
            toe_link_index = self._foot_link_ids.index(foot_id)
            foot_position = self._pybullet_client.getLinkState(self.quadruped, foot_id)[0]  # index 0 is linkWorldPosition
            projection_position = copy.copy(foot_position)
            projection_position[2] = projection_position[2] - max_detection_distance  # always pointing downwards
            data = self._pybullet_client.rayTest(foot_position, projection_position)
            if data[0] == -1:  # -1 if did not detect anything within detectable distance (out of range)
                distances[toe_link_index] = max_detection_distance
            else:
                distances[toe_link_index] = (
                    data[2] * max_detection_distance
                )  # data[2] * (abs(foot_position[2]) - abs(projection_position[2]))

        return distances

    def _GetTerrainHeightUnderPoint(self, position_world, max_height=5.0):
        """Get the terrain height at a 2D position"""

        upper_bound = np.concatenate([position_world, np.array([max_height])])
        lower_bound = copy.copy(upper_bound)
        lower_bound[2] = lower_bound[2] - 2 * max_height

        data = self._pybullet_client.rayTest(upper_bound, lower_bound)[0]
        if data[0] == -1:  # -1 if did not detect anything within detectable distance (out of range)
            distance_to_ground = 2 * max_height
        else:
            distance_to_ground = data[2] * 2 * max_height
        return max_height - distance_to_ground

    def GetLocalTerrainView(self, grid_unit=0.1, grid_size=[10, 10], transform=(0, 0)):
        """Returns a view of the local terrain as seen from a single point.

        Args:
          grid_unit: Side length of one square in the grid
          grid_size: Number of squares along one side of grid
          transform: The direction to transform the terrain view

        Returns:
          N x N numpy array of floats
        """
        base_position_world = self.GetBasePosition()[:2]
        base_position_world = base_position_world + np.array(transform)
        base_position_base = world_frame_to_base_frame(base_position_world, self)
        grid_coordinates_base = get_grid_coordinates(grid_unit, grid_size) + base_position_base
        grid_coordinates_world = np.array([base_frame_to_world_frame(gcb, self) for gcb in grid_coordinates_base])
        grid_coordinates_world_3d = [np.concatenate([gcw, [0]]) for gcw in grid_coordinates_world]

        robot_positions = [self.GetBasePosition()] * len(grid_coordinates_world_3d)

        z_coordinates = []
        ray_intersection_infos = self._pybullet_client.rayTestBatch(robot_positions, grid_coordinates_world_3d)
        for info in ray_intersection_infos:
            hit_position = info[3]
            z_coordinates.append(hit_position[2])
        z_coordinates = np.array(z_coordinates).reshape(grid_size)
        return z_coordinates

    def GetLocalTerrainDepth(self, grid_unit=(0.1, 0.1), grid_size=[10, 10], transform=(0, 0), ray_origin="body"):
        """Returns the depth of the terrain as seen from a single point.

        Args:
          grid_unit: Side length of one square in the grid
          grid_size: Number of squares along one side of grid
          transform: The direction to transform the terrain view
          ray_origin: The origin of where the rays come from - "body" or "head"

        Returns:
          N x M numpy array of floats
        """
        # # For visualising rays
        # if not hasattr(self, "ball_ids"):
        #     self.ball_ids = []
        # if len(self.ball_ids) > 190:
        #     for i in self.ball_ids:
        #         self._pybullet_client.removeBody(i)
        #     self.ball_ids = []

        base_pos = self.GetBasePosition()
        rpy = self.GetTrueBaseRollPitchYaw()

        # Calculate origin position
        orientation = self.GetTrueBaseOrientation()
        rot_matrix = self._pybullet_client.getMatrixFromQuaternion(orientation)
        if ray_origin == "body":
            # slightly below base position
            local_axis_vec = rot_matrix[6:]
            tmp_coord = 0.07 * np.array([1.0, 1.0, -1.0]) * np.asarray(local_axis_vec)
        elif ray_origin == "head":
            # at robot head
            local_axis_vec = rot_matrix[:3]
            tmp_coord = 0.27 * np.array([1.0, -1.0, -1.0]) * np.asarray(local_axis_vec)
        else:
            raise Exception("Ray origin specified does not exist")
        origin_world = base_pos + tmp_coord
        # Transform origin_world to robot base yaw frame
        origin_base = np.array(origin_world)
        origin_base[:2] = transform_to_rotated_frame(origin_world[:2], rpy[2])

        # Calculate 2d target coordinates
        target_coords_base_2d = get_grid_coordinates(grid_unit, grid_size) + origin_base[:2] + transform
        # Transform 2d coordinates to world frame
        target_coords_world_2d = np.array([transform_to_rotated_frame(tcb, -rpy[2]) for tcb in target_coords_base_2d])
        # Calculate target z position
        test_target_coords = [np.concatenate([tcw, [-0.001]]) for tcw in target_coords_world_2d]
        test_origin_coords = [np.concatenate([tcw, [base_pos[2]]]) for tcw in target_coords_world_2d]
        test_hit_coordinates = []
        test_ray_intersection_infos = self._pybullet_client.rayTestBatch(test_origin_coords, test_target_coords)
        for i, info in enumerate(test_ray_intersection_infos):
            # If test ray hit robot, try again
            if info[0] == self.quadruped:
                info = self._pybullet_client.rayTest(info[3], test_target_coords[i])[0]
            # Check if test ray hit anything
            if info[0] == -1:
                test_hit_position = test_target_coords[i]
            else:
                test_hit_position = info[3]
            test_hit_coordinates.append(test_hit_position)
        # Form 3d target coordinates
        target_coords = [
            np.concatenate([tcw, [test_hit_coordinates[i][2] - 0.001]]) for i, tcw in enumerate(target_coords_world_2d)
        ]

        # Calculate depth data
        origin_coords = [origin_world] * len(target_coords)
        hit_coordinates = []
        ray_intersection_infos = self._pybullet_client.rayTestBatch(origin_coords, target_coords)
        for i, info in enumerate(ray_intersection_infos):
            if info[0] == -1:
                hit_position = target_coords[i]
            else:
                hit_position = info[3]
            hit_coordinates.append(hit_position)
        # depth_distances = np.subtract(origin_coords, hit_coordinates)
        # depth_view = [np.linalg.norm(d, 2) for d in depth_distances]
        depth_view = np.array(origin_coords)[:, 2] - np.array(hit_coordinates)[:, 2]
        depth_view = np.array(depth_view).reshape(grid_size)

        # # For visualising rays
        # ballid = draw_debug_sphere(self._pybullet_client, origin_world, [0, 0, 1, 1])
        # self.ball_ids.append(ballid)
        # for coord in hit_coordinates:
        #     ballid = draw_debug_sphere(self._pybullet_client, coord, [0, 1, 0, 1])
        #     self.ball_ids.append(ballid)

        return depth_view

    def GetLocalTerrainDepthByAngle(
        self, grid_angle=(0.1, 0.1), grid_size=[10, 10], transform_angle=(0, 0), ray_origin="body"
    ):
        """Returns the depth of the terrain as seen from a single point.

        Args:
          grid_angle: Angle between each ray
          grid_size: Number of squares along one side of grid
          transform_angle: The angle to transform the terrain view
          ray_origin: The origin of where the rays come from - "body" or "head"

        Returns:
          N x M numpy array of floats
        """
        # # For visualising rays
        # if not hasattr(self, 'ball_ids'):
        #     self.ball_ids = []
        # if len(self.ball_ids) > 40:
        #    for i in self.ball_ids:
        #        self._pybullet_client.removeBody(i)
        #    self.ball_ids = []

        base_pos = self.GetBasePosition()
        rpy = self.GetTrueBaseRollPitchYaw()
        imaginary_wall_dist = 8.0

        # Calculate origin position
        orientation = self.GetTrueBaseOrientation()
        rot_matrix = self._pybullet_client.getMatrixFromQuaternion(orientation)
        if ray_origin == "body":
            # slightly below base position
            local_axis_vec = rot_matrix[6:]
            tmp_coord = 0.07 * np.array([1.0, 1.0, -1.0]) * np.asarray(local_axis_vec)
        elif ray_origin == "head":
            # at robot head
            local_axis_vec = rot_matrix[:3]
            tmp_coord = 0.27 * np.array([1.0, -1.0, -1.0]) * np.asarray(local_axis_vec)
        else:
            raise Exception("Ray origin specified does not exist")
        origin_world = base_pos + tmp_coord
        # Transform origin_world to robot base yaw frame
        origin_base = np.array(origin_world)
        origin_base[:2] = transform_to_rotated_frame(origin_world[:2], rpy[2])

        # Calculate grid coordinates, taking into account roll and pitch
        target_coords = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                x_angle = rpy[1] + grid_angle[0] * (i - (grid_size[0] - 1) / 2) + transform_angle[0]
                y_angle = rpy[0] + grid_angle[1] * (j - (grid_size[1] - 1) / 2) + transform_angle[1]
                z = -0.001
                if abs(x_angle) < np.pi / 2:
                    # Projected to ground
                    x = origin_base[0] - origin_base[2] * np.tan(x_angle)
                else:
                    # Projected to imaginary wall
                    x = origin_base[0] - np.sign(x_angle) * imaginary_wall_dist
                    z = origin_base[2] + imaginary_wall_dist * np.tan(abs(x_angle) - np.pi / 2)
                if abs(y_angle) < np.pi / 2:
                    # Projected to ground
                    y = origin_base[1] + origin_base[2] * np.tan(y_angle)
                else:
                    # Projected to imaginary wall
                    y = origin_base[1] + np.sign(y_angle) * imaginary_wall_dist
                    z = origin_base[2] + imaginary_wall_dist * np.tan(abs(y_angle) - np.pi / 2)
                # Tranform x, y coordinates to world frame
                x, y = transform_to_rotated_frame((x, y), -rpy[2])
                target_coords.append((x, y, z))

        # Calculate depth data
        origin_coords = [origin_world] * len(target_coords)
        hit_coordinates = []
        ray_intersection_infos = self._pybullet_client.rayTestBatch(origin_coords, target_coords)
        for i, info in enumerate(ray_intersection_infos):
            if info[0] == -1:
                hit_position = target_coords[i]
            else:
                hit_position = info[3]
            hit_coordinates.append(hit_position)
        depth_distances = np.subtract(origin_coords, hit_coordinates)
        depth_view = [np.linalg.norm(d, 2) for d in depth_distances]
        depth_view = np.array(depth_view).reshape(grid_size)

        # # For visualising rays
        # ballShape = self._pybullet_client.createCollisionShape(shapeType=self._pybullet_client.GEOM_SPHERE, radius=0.02)
        # ballid = self._pybullet_client.createMultiBody(
        #     baseMass=0, baseCollisionShapeIndex=ballShape, basePosition=origin_world, baseOrientation=[0, 0, 0, 1]
        # )
        # self._pybullet_client.changeVisualShape(ballid, -1, rgbaColor=[0, 0, 1, 1])
        # self._pybullet_client.setCollisionFilterGroupMask(ballid, -1, 0, 0)
        # self.ball_ids.append(ballid)
        # for coord in hit_coordinates:
        #     ballid = self._pybullet_client.createMultiBody(
        #         baseMass=0, baseCollisionShapeIndex=ballShape, basePosition=coord, baseOrientation=[0, 0, 0, 1]
        #     )
        #     self._pybullet_client.changeVisualShape(ballid, -1, rgbaColor=[1, 0, 0, 1])
        #     self._pybullet_client.setCollisionFilterGroupMask(ballid, -1, 0, 0)
        #     self.ball_ids.append(ballid)

        return depth_view

    def GetLocalDistancesToGround(self, grid_unit=0.05, grid_size=16):
        """Get the vertical distance from base height to ground in a NxN grid around the robot.

        Args:
          grid_unit: Side length of one square in the grid
          grid_size: Number of squares along one side of grid

        Returns:
          N x N numpy array of floats
        """
        base_position_world = self.GetBasePosition()[:2]
        base_height = self.GetBasePosition()[2]
        base_position_base = world_frame_to_base_frame(base_position_world, self)
        grid_coordinates_base = get_grid_coordinates(grid_unit, grid_size) + base_position_base
        grid_coordinates_world = np.array([base_frame_to_world_frame(gcb, self) for gcb in grid_coordinates_base])

        heights = []
        for coord in grid_coordinates_world:
            heights.append(self._GetTerrainHeightUnderPoint(coord))
        # Subtract terrain height from base height to get vertical distance to ground
        return base_height - np.array(heights).reshape((grid_size, grid_size))

    def ResetPose(self, add_constraint):
        del add_constraint
        for name in self._joint_name_to_id:
            joint_id = self._joint_name_to_id[name]
            self._pybullet_client.setJointMotorControl2(
                bodyIndex=self.quadruped,
                jointIndex=(joint_id),
                controlMode=self._pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0,
            )
        for name, i in zip(MOTOR_NAMES, range(len(MOTOR_NAMES))):
            if "hip_joint" in name:
                angle = INIT_MOTOR_ANGLES[i] + HIP_JOINT_OFFSET
            elif "upper_joint" in name:
                angle = INIT_MOTOR_ANGLES[i] + UPPER_LEG_JOINT_OFFSET
            elif "lower_joint" in name:
                angle = INIT_MOTOR_ANGLES[i] + KNEE_JOINT_OFFSET
            else:
                raise ValueError("The name %s is not recognized as a motor joint." % name)
            self._pybullet_client.resetJointState(self.quadruped, self._joint_name_to_id[name], angle, targetVelocity=0)

    def GetURDFFile(self):
        return self._urdf_filename

    def _BuildUrdfIds(self):
        """Build the link Ids from its name in the URDF file.

        Raises:
          ValueError: Unknown category of the joint name.
        """
        num_joints = self.pybullet_client.getNumJoints(self.quadruped)
        self._hip_link_ids = [-1]
        self._leg_link_ids = []
        self._motor_link_ids = []
        self._lower_link_ids = []
        self._foot_link_ids = []
        self._imu_link_ids = []

        for i in range(num_joints):
            joint_info = self.pybullet_client.getJointInfo(self.quadruped, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_id = self._joint_name_to_id[joint_name]
            if HIP_NAME_PATTERN.match(joint_name):
                self._hip_link_ids.append(joint_id)
            elif UPPER_NAME_PATTERN.match(joint_name):
                self._motor_link_ids.append(joint_id)
            # We either treat the lower leg or the toe as the foot link, depending on
            # the urdf version used.
            elif LOWER_NAME_PATTERN.match(joint_name):
                self._lower_link_ids.append(joint_id)
            elif TOE_NAME_PATTERN.match(joint_name):
                # assert self._urdf_filename == URDF_WITH_TOES
                self._foot_link_ids.append(joint_id)
            elif IMU_NAME_PATTERN.match(joint_name):
                self._imu_link_ids.append(joint_id)
            else:
                raise ValueError("Unknown category of joint %s" % joint_name)

        self._leg_link_ids.extend(self._lower_link_ids)
        self._leg_link_ids.extend(self._foot_link_ids)

        # assert len(self._foot_link_ids) == NUM_LEGS
        self._hip_link_ids.sort()
        self._motor_link_ids.sort()
        self._lower_link_ids.sort()
        self._foot_link_ids.sort()
        self._leg_link_ids.sort()

    def _GetMotorNames(self):
        return MOTOR_NAMES

    def _GetDefaultInitPosition(self):
        if self._on_rack:
            return [sum(x) for x in zip(INIT_RACK_POSITION, self._adjust_position)]
        else:
            return [sum(x) for x in zip(INIT_POSITION, self._adjust_position)]

    def _GetDefaultInitOrientation(self):
        # The Laikago URDF assumes the initial pose of heading towards z axis,
        # and belly towards y axis. The following transformation is to transform
        # the Laikago initial orientation to our commonly used orientation: heading
        # towards -x direction, and z axis is the up direction.
        init_orientation = pyb.getQuaternionFromEuler([0.0, 0.0, 0.0])
        return init_orientation

    def GetDefaultInitPosition(self):
        """Get default initial base position."""
        return self._GetDefaultInitPosition()

    def GetDefaultInitOrientation(self):
        """Get default initial base orientation."""
        return self._GetDefaultInitOrientation()

    def GetDefaultInitJointPose(self):
        """Get default initial joint pose."""
        joint_pose = (INIT_MOTOR_ANGLES + JOINT_OFFSETS) * JOINT_DIRECTIONS
        return joint_pose

    def ApplyAction(self, motor_commands, motor_control_mode=None):
        """Clips and then apply the motor commands using the motor model.

        Args:
          motor_commands: np.array. Can be motor angles, torques, hybrid commands,
            or motor pwms (for Minitaur only).N
          motor_control_mode: A MotorControlMode enum.
        """
        if self._enable_clip_motor_commands:
            motor_commands = self._ClipMotorCommands(motor_commands)
        super(A1, self).ApplyAction(motor_commands, motor_control_mode)

    def _ClipMotorCommands(self, motor_commands):
        """Clips motor commands.

        Args:
          motor_commands: np.array. Can be motor angles, torques, hybrid commands,
            or motor pwms (for Minitaur only).

        Returns:
          Clipped motor commands.
        """

        # clamp the motor command by the joint limit, in case weired things happens
        max_angle_change = MAX_MOTOR_ANGLE_CHANGE_PER_STEP
        current_motor_angles = self.GetMotorAngles()
        motor_commands = np.clip(
            motor_commands,
            current_motor_angles - max_angle_change,
            current_motor_angles + max_angle_change,
        )
        return motor_commands

    @classmethod
    def GetConstants(cls):
        del cls
        return laikago_constants

    def ComputeMotorAnglesFromFootLocalPosition(self, leg_id, foot_local_position):
        """Use IK to compute the motor angles, given the foot link's local position.

        Args:
          leg_id: The leg index.
          foot_local_position: The foot link's position in the base frame.

        Returns:
          A tuple. The position indices and the angles for all joints along the
          leg. The position indices is consistent with the joint orders as returned
          by GetMotorAngles API.
        """
        assert len(self._foot_link_ids) == self.num_legs
        # toe_id = self._foot_link_ids[leg_id]

        motors_per_leg = self.num_motors // self.num_legs
        joint_position_idxs = list(range(leg_id * motors_per_leg, leg_id * motors_per_leg + motors_per_leg))

        joint_angles = foot_position_in_hip_frame_to_joint_angle(
            foot_local_position - HIP_OFFSETS[leg_id], l_hip_sign=(-1) ** (leg_id + 1)
        )

        # Joint offset is necessary for Laikago.
        joint_angles = np.multiply(
            np.asarray(joint_angles) - np.asarray(self._motor_offset)[joint_position_idxs],
            self._motor_direction[joint_position_idxs],
        )

        # Return the joing index (the same as when calling GetMotorAngles) as well
        # as the angles.
        return joint_position_idxs, joint_angles.tolist()

    def GetFootPositionsInBaseFrame(self):
        """Get the robot's foot position in the base frame."""
        motor_angles = self.GetMotorAngles()
        return foot_positions_in_base_frame(motor_angles)

    def ComputeJacobian(self, leg_id):
        """Compute the Jacobian for a given leg."""
        # Does not work for Minitaur which has the four bar mechanism for now.
        motor_angles = self.GetMotorAngles()[leg_id * 3 : (leg_id + 1) * 3]
        return analytical_leg_jacobian(motor_angles, leg_id)
