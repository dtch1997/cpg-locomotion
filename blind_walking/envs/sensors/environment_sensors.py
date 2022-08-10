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

"""Simple sensors related to the environment."""
import csv
import typing

import numpy as np
import torch
from blind_walking.envs.sensors import sensor
from numpy.lib.function_base import _angle_dispatcher

_ARRAY = typing.Iterable[float]  # pylint:disable=invalid-name
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY]  # pylint:disable=invalid-name
_DATATYPE_LIST = typing.Iterable[typing.Any]  # pylint:disable=invalid-name

maxdist_schedule = lambda t: 0.02 if t < 300 else 0.01


class ForwardTargetPositionSensor(sensor.BoxSpaceSensor):
    """A sensor that reports the relative target position."""

    def __init__(
        self,
        max_range: float = 0.01,
        min_range: float = 0.01,
        lower_bound: _FLOAT_OR_ARRAY = -1.0,
        upper_bound: _FLOAT_OR_ARRAY = 1.0,
        name: typing.Text = "TargetPosition",
        enc_name: typing.Text = "flatten",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs ForwardTargetPositionSensor.
        Args:
          lower_bound: the lower bound of the target position
          upper_bound: the upper bound of the target position
          name: the name of the sensor
          dtype: data type of sensor value
        """
        self._env = None

        super(ForwardTargetPositionSensor, self).__init__(
            name=name,
            shape=(2,),
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

        self._max_range = max_range
        self._min_range = min_range
        self._max_distance = self._max_range

        self._last_base_pos = np.zeros(3)
        self._current_base_pos = np.zeros(3)
        self._last_yaw = 0
        self._current_yaw = 0

    def on_step(self, env):
        # self._max_distance = maxdist_schedule(env.env_step_counter)

        self._last_base_pos = self._current_base_pos
        self._current_base_pos = self._env._robot.GetBasePosition()
        self._last_yaw = self._current_yaw
        self._current_yaw = self._env._robot.GetTrueBaseRollPitchYaw()[2]

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.
        Args:
          env: the environment who invokes this callback function.
        """
        self._env = env

        self._current_base_pos = self._env._robot.GetBasePosition()
        self._last_base_pos = self._current_base_pos
        self._current_yaw = self._env._robot.GetTrueBaseRollPitchYaw()[2]
        self._last_yaw = self._current_yaw

        # random sampling of target distance from range
        self._max_distance = np.random.uniform(self._min_range, self._max_range)

        # # target distance is proportional to gait frequency
        # ref_gait_sensor = env.all_sensors()[-3]
        # min_freq, max_freq = ref_gait_sensor._gait_frequency_range
        # tgt_freq = 1 / ref_gait_sensor.get_period()
        # self._max_distance = self._min_range + \
        #     ((tgt_freq - min_freq) / (max_freq - min_freq)) * (self._max_range - self._min_range)

    def _get_observation(self) -> _ARRAY:
        # target y position is always zero
        dy_target = 0 - self._current_base_pos[1]
        # give some leeway for the robot to walk forward
        dy_target = max(min(dy_target, self._max_distance / 2), -self._max_distance / 2)
        # target x position is always forward
        dx_target = np.sqrt(pow(self._max_distance, 2) - pow(dy_target, 2))
        # Transform to local frame
        dx_target_local, dy_target_local = self.to_local_frame(dx_target, dy_target, self._current_yaw)
        return [dx_target_local, dy_target_local]

    @staticmethod
    def to_local_frame(dx, dy, yaw):
        # Transform the x and y direction distances to the robot's local frame
        dx_local = np.cos(yaw) * dx + np.sin(yaw) * dy
        dy_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
        return dx_local, dy_local


class LocalTerrainDepthSensor(sensor.BoxSpaceSensor):
    """A sensor that gets the depth from the robot to the ground"""

    def __init__(
        self,
        noisy_reading: bool = True,
        grid_unit: typing.Tuple[float] = (0.1, 0.1),
        grid_size: typing.Tuple[int] = (10, 10),
        transform: typing.Tuple[float] = (0, 0),
        ray_origin: typing.Text = "body",
        lower_bound: _FLOAT_OR_ARRAY = 0.0,
        upper_bound: _FLOAT_OR_ARRAY = 8.0,
        name: typing.Text = "LocalTerrainDepth",
        enc_name: typing.Text = "flatten",
        dtype: typing.Type[typing.Any] = np.float64,
        encoder: typing.Type[typing.Any] = None,
    ) -> None:
        """Constructs LocalTerrainDepthSensor.
        Args:
          grid_unit: Side length of one square in the grid
          grid_size: Number of squares along one side of grid
          lower_bound: the lower bound of the terrain view.
          upper_bound: the upper bound of the terrain view.
          name: the name of the sensor.
          dtype: data type of sensor value.
          encoder: pretrained encoder which the raw observations pass through.
        """
        self._env = None
        self._noisy_reading = noisy_reading
        self.grid_unit = grid_unit
        self.grid_size = grid_size
        self.transform = transform
        self.ray_origin = ray_origin
        self.encoder = encoder

        shape = (1, grid_size[0], grid_size[1]) if not encoder else (1, 32)
        super(LocalTerrainDepthSensor, self).__init__(
            name=name,
            shape=shape,
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.
        Args:
          env: the environment who invokes this callback function.
        """
        self._env = env

    def _get_observation(self) -> _ARRAY:
        """Returns the local distances to ground"""
        heightmap = self._env.robot.GetLocalTerrainDepth(
            grid_unit=self.grid_unit,
            grid_size=self.grid_size,
            transform=self.transform,
            ray_origin=self.ray_origin,
        ).reshape(1, self.grid_size[0], self.grid_size[1])
        # Add noise
        if self._noisy_reading:
            heightmap = heightmap + np.random.normal(scale=0.01, size=heightmap.shape)
        # Clip readings
        heightmap = np.minimum(np.maximum(heightmap, 0.1), 8.0)
        # Encode raw observations
        if self.encoder:
            heightmap = heightmap.reshape(-1, np.prod(self.grid_size))
            return self.encoder(torch.Tensor(heightmap)).detach().numpy()
        return heightmap


class LocalTerrainDepthByAngleSensor(sensor.BoxSpaceSensor):
    """A sensor that gets the depth from the robot to the ground"""

    def __init__(
        self,
        noisy_reading: bool = True,
        grid_angle: typing.Tuple[float] = (0.1, 0.1),
        grid_size: typing.Tuple[int] = (10, 10),
        transform_angle: typing.Tuple[float] = (0, 0),
        ray_origin: typing.Text = "body",
        lower_bound: _FLOAT_OR_ARRAY = 0.0,
        upper_bound: _FLOAT_OR_ARRAY = 8.0,
        name: typing.Text = "LocalTerrainDepthByAngle",
        enc_name: typing.Text = "flatten",
        dtype: typing.Type[typing.Any] = np.float64,
    ) -> None:
        """Constructs LocalTerrainDepthByAngleSensor.
        Args:
          grid_unit: Side length of one square in the grid
          grid_size: Number of squares along one side of grid
          lower_bound: the lower bound of the terrain view.
          upper_bound: the upper bound of the terrain view.
          name: the name of the sensor.
          dtype: data type of sensor value.
        """
        self._env = None
        self._noisy_reading = noisy_reading
        self.grid_angle = grid_angle
        self.grid_size = grid_size
        self.transform_angle = transform_angle
        self.ray_origin = ray_origin

        shape = (1, grid_size[0], grid_size[1])
        super(LocalTerrainDepthByAngleSensor, self).__init__(
            name=name,
            shape=shape,
            enc_name=enc_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            dtype=dtype,
        )

    def on_reset(self, env):
        """From the callback, the sensor remembers the environment.
        Args:
          env: the environment who invokes this callback function.
        """
        self._env = env

    def _get_observation(self) -> _ARRAY:
        """Returns the local distances to ground"""
        heightmap = self._env.robot.GetLocalTerrainDepthByAngle(
            grid_angle=self.grid_angle,
            grid_size=self.grid_size,
            transform_angle=self.transform_angle,
            ray_origin=self.ray_origin,
        ).reshape(1, self.grid_size[0], self.grid_size[1])
        # Add noise
        if self._noisy_reading:
            heightmap = heightmap + np.random.normal(scale=0.01, size=heightmap.shape)
        # Clip readings
        heightmap = np.minimum(np.maximum(heightmap, 0.1), 8.0)
        return heightmap
