import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from blind_walking.envs.utilities import env_randomizer_base


@dataclass
class RMAEnvRandomizerConfig:
    resample_probability: float
    controller_Kp_lower_bound: float
    controller_Kp_upper_bound: float
    controller_Kd_lower_bound: float
    controller_Kd_upper_bound: float
    motor_strength_ratios_lower_bound: float
    motor_strength_ratios_upper_bound: float


config_registry = {
    "no_var": RMAEnvRandomizerConfig(
        resample_probability=0,
        controller_Kp_lower_bound=55,
        controller_Kp_upper_bound=55,
        controller_Kd_lower_bound=0.6,
        controller_Kd_upper_bound=0.6,
        motor_strength_ratios_lower_bound=1.0,
        motor_strength_ratios_upper_bound=1.0,
    ),
    "rma_easy": RMAEnvRandomizerConfig(
        resample_probability=0.004,
        controller_Kp_lower_bound=50,
        controller_Kp_upper_bound=60,
        controller_Kd_lower_bound=0.4,
        controller_Kd_upper_bound=0.8,
        motor_strength_ratios_lower_bound=0.9,
        motor_strength_ratios_upper_bound=1.0,
    ),
    "rma_hard": RMAEnvRandomizerConfig(
        resample_probability=0.01,
        controller_Kp_lower_bound=45,
        controller_Kp_upper_bound=65,
        controller_Kd_lower_bound=0.3,
        controller_Kd_upper_bound=0.9,
        motor_strength_ratios_lower_bound=0.88,
        motor_strength_ratios_upper_bound=1.0,
    ),
}


class RMAEnvRandomizer(env_randomizer_base.EnvRandomizerBase):
    """A randomizer that perturbs the A1 gym env according to RMA paper"""

    def __init__(self, config):
        self.config = config

    def _randomize(self, env):
        robot = env._robot
        if np.random.uniform() < self.config.resample_probability:
            Kp = np.random.uniform(
                self.config.controller_Kp_lower_bound,
                self.config.controller_Kp_upper_bound,
            )
            Kd = np.random.uniform(
                self.config.controller_Kd_lower_bound,
                self.config.controller_Kd_upper_bound,
            )
            motor_strength_ratio = np.random.uniform(
                self.config.motor_strength_ratios_lower_bound,
                self.config.motor_strength_ratios_upper_bound,
            )
            robot.SetMotorGains(Kp, Kd)
            robot.SetMotorStrengthRatio(motor_strength_ratio)

    def randomize_env(self, env):
        return self._randomize(env)

    def randomize_step(self, env):
        return self._randomize(env)
