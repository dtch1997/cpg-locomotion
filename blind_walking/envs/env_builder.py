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
import os

import torch as th

"""Utilities for building environments."""
from blind_walking.envs import locomotion_gym_config, locomotion_gym_env
from blind_walking.envs.env_modifiers import train_course
from blind_walking.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_array_wrapper
from blind_walking.envs.env_wrappers import simple_openloop, trajectory_generator_wrapper_env
from blind_walking.envs.sensors import cpg_sensors, environment_sensors, robot_sensors, sensor_wrappers
from blind_walking.envs.tasks import imitation_task
from blind_walking.envs.utilities.controllable_env_randomizer_from_config import ControllableEnvRandomizerFromConfig
from blind_walking.robots import a1, laikago, robot_config
from train_autoencoder import LinearAE


# Load heightmap encoder
def load_encoder():
    model = LinearAE(input_size=12 * 16, code_size=32)
    model_state, optimizer_state = th.load(os.path.join(os.getcwd(), "saved_models/autoencoder/model_bs32_cs32_lr0.001"))
    model.load_state_dict(model_state)
    model.eval()
    _hm_encoder = model.encoder
    for param in _hm_encoder.parameters():
        param.requires_grad = False
    return _hm_encoder

data_path = os.path.join(os.getcwd(), "blind_walking/data")

def build_regular_env(
    robot_class,
    enable_rendering=False,
    on_rack=False,
    action_limit=(0.5, 0.5, 0.5),
    robot_sensor_list=None,
    env_sensor_list=None,
    env_randomizer_list=None,
    env_modifier_list=None,
    task=None,
    # CPG sensor kwargs
    gait_name = None,
    gait_frequency = None,
    duty_factor = None
):


    sim_params = locomotion_gym_config.SimulationParameters()
    sim_params.enable_rendering = enable_rendering
    sim_params.motor_control_mode = robot_config.MotorControlMode.POSITION
    sim_params.reset_time = 2
    sim_params.num_action_repeat = 25
    sim_params.enable_action_interpolation = False
    sim_params.enable_action_filter = True
    sim_params.enable_clip_motor_commands = True
    sim_params.robot_on_rack = on_rack

    gym_config = locomotion_gym_config.LocomotionGymConfig(simulation_parameters=sim_params)

    if robot_sensor_list is None:
        robot_sensor_list = [
            robot_sensors.BaseVelocitySensor(convert_to_local_frame=True),
            sensor_wrappers.HistoricSensorWrapper(
                robot_sensors.IMUSensor(channels=["R", "P", "dR", "dP", "dY"]), num_history=3
            ),
            sensor_wrappers.HistoricSensorWrapper(robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS), num_history=3),
            cpg_sensors.ReferenceGaitSensor(
                gait_names=["trot"] if gait_name is None else [gait_name],
                gait_frequency_upper=2.0 if gait_frequency is None else gait_frequency,
                gait_frequency_lower=2.0 if gait_frequency is None else gait_frequency,
                duty_factor_upper=0.75 if duty_factor is None else duty_factor,
                duty_factor_lower=0.75 if duty_factor is None else duty_factor,
                obs_steps_ahead=[0, 1, 2, 10, 50],
            ),
        ]
    if env_sensor_list is None:
        env_sensor_list = [
            environment_sensors.ForwardTargetPositionSensor(min_range=0.015, max_range=0.015),
        ]

    if env_randomizer_list is None:
        env_randomizer_list = []

    if env_modifier_list is None:
        env_modifier_list = []

    if task is None:
        task = imitation_task.ImitationTask()

    env = locomotion_gym_env.LocomotionGymEnv(
        gym_config=gym_config,
        robot_class=robot_class,
        robot_sensors=robot_sensor_list,
        env_sensors=env_sensor_list,
        task=task,
        env_randomizers=env_randomizer_list,
        env_modifiers=env_modifier_list,
        data_path = data_path
    )

    env = obs_array_wrapper.ObservationDictionaryToArrayWrapper(env)
    env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
        env,
        trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(action_limit=action_limit),
    )

    return env
