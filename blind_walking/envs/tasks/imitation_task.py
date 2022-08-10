import math

import numpy as np


class ImitationTask(object):
    def __init__(self, num_legs=4, num_motors=12):
        """Initializes the task."""
        self._num_legs = num_legs
        self._num_motors = num_motors

        self.current_base_pos = np.zeros(3)
        self.last_base_pos = np.zeros(3)
        self.current_base_rpy = np.zeros(3)
        self.last_base_rpy = np.zeros(3)
        self.current_base_rpy_rate = np.zeros(3)
        self.last_base_rpy_rate = np.zeros(3)
        self.current_motor_velocities = np.zeros(num_motors)
        self.last_motor_velocities = np.zeros(num_motors)
        self.current_motor_torques = np.zeros(num_motors)
        self.last_motor_torques = np.zeros(num_motors)
        self.current_base_orientation = np.zeros(3)
        self.last_base_orientation = np.zeros(3)
        self.current_foot_contacts = np.zeros(num_legs)
        self.last_foot_contacts = np.zeros(num_legs)
        self.feet_air_time = np.zeros(num_legs)
        self.feet_contact_lost = np.zeros(num_legs)

        self._target_pos = [0, 0]

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        """Resets the internal state of the task."""
        self._env = env

        self.last_base_pos = env.robot.GetBasePosition()
        self.current_base_pos = self.last_base_pos
        self.last_base_rpy = env.robot.GetBaseRollPitchYaw()
        self.current_base_rpy = self.last_base_rpy
        self.last_base_rpy_rate = env.robot.GetBaseRollPitchYawRate()
        self.current_base_rpy_rate = self.last_base_rpy_rate
        self.last_motor_velocities = env.robot.GetMotorVelocities()
        self.current_motor_velocities = self.last_motor_velocities
        self.last_motor_torques = env.robot.GetMotorTorques()
        self.current_motor_torques = self.last_motor_torques
        self.last_base_orientation = env.robot.GetBaseOrientation()
        self.current_base_orientation = self.last_base_orientation
        self.last_foot_contacts = env.robot.GetFootContacts()
        self.current_foot_contacts = self.last_foot_contacts
        self.feet_air_time = env.robot._feet_air_time
        self.feet_contact_lost = env.robot._feet_contact_lost

    def update(self, env):
        """Updates the internal state of the task."""
        self.last_base_pos = self.current_base_pos
        self.current_base_pos = env.robot.GetBasePosition()
        self.last_base_rpy = self.current_base_rpy
        self.current_base_rpy = env.robot.GetBaseRollPitchYaw()
        self.last_base_rpy_rate = self.current_base_rpy_rate
        self.current_base_rpy_rate = env.robot.GetBaseRollPitchYawRate()
        self.last_motor_velocities = self.current_motor_velocities
        self.current_motor_velocities = env.robot.GetMotorVelocities()
        self.last_motor_torques = self.current_motor_torques
        self.current_motor_torques = env.robot.GetMotorTorques()
        self.last_base_orientation = self.current_base_orientation
        self.current_base_orientation = env.robot.GetBaseOrientation()
        self.last_foot_contacts = self.current_foot_contacts
        self.current_foot_contacts = env.robot.GetFootContacts()
        self.feet_air_time = env.robot._feet_air_time

        # Update reference displacement
        self._reference_displacement = env._observations["TargetPosition_flatten"]
        # Update actual displacement
        dx, dy, dz = np.array(self.current_base_pos) - np.array(self.last_base_pos)
        dx_local, dy_local = self.to_local_frame(dx, dy, self.last_base_rpy[2])
        self._actual_displacement = np.array([dx_local, dy_local])

        # Assume gait sensor is last sensor
        ref_gait_sensor = env.all_sensors()[-2]
        self._reference_foot_contacts = ref_gait_sensor.get_current_reference_state()
        t = env.env_time_step
        self._actual_foot_contacts = (t - 2 * self.feet_air_time) / t

    def done(self, env):
        """Checks if the episode is over.
        If the robot base becomes unstable (based on orientation), the episode
        terminates early.
        """
        rot_quat = env.robot.GetBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
        return rot_mat[-1] < 0.5

    def _calc_reward_distance(self):
        """Reward term for travelling in the indicated direction"""
        # Reward distance travelled in target direction.
        distance_target = np.linalg.norm(self._reference_displacement)
        distance_towards = np.dot(self._actual_displacement, self._reference_displacement) / distance_target
        diff = (distance_towards - distance_target) / distance_target
        alpha = 1.0
        if diff < 0:
            # if distance travelled is less than target
            distance_reward = math.exp(-((4 * diff) ** 2))
        elif round(diff, 5) < 1 / (5 * alpha):
            # if distance travelled is more than target
            distance_reward = math.exp(1 - 1 / (1 - (5 * alpha * diff) ** 2))
        else:
            distance_reward = 0
        return distance_reward

    def _calc_reward_shake(self):
        """Reward term for staying upright"""
        orientation = self.current_base_orientation
        rot_matrix = self._env.pybullet_client.getMatrixFromQuaternion(orientation)
        local_up_vec = rot_matrix[6:]
        shake_reward = -abs(np.dot(np.asarray([1, 1, 0]), np.asarray(local_up_vec)))
        return shake_reward

    def _calc_reward_energy(self):
        energy_reward = -np.abs(np.dot(self.current_motor_torques, self.current_motor_velocities))
        return energy_reward

    def _calc_reward_imitation(self):
        feet_ground_time = (self._env.env_time_step - self.feet_air_time) / self._env.env_time_step
        ref_foot_contact_imitation_reward = np.dot(feet_ground_time, self._reference_foot_contacts)
        # Rescale from [-4,4] to [-1,1]
        ref_foot_contact_imitation_reward /= 4
        return ref_foot_contact_imitation_reward

    def reward(self, env):
        """Get the reward without side effects.

        Also return a dict of reward components"""
        del env

        distance_reward = self._calc_reward_distance()
        shake_reward = self._calc_reward_shake()
        energy_reward = self._calc_reward_energy()
        imitation_reward = self._calc_reward_imitation()

        # Dictionary of:
        # - {name: reward * weight}
        # for all reward components
        weighted_objectives = {
            "distance": distance_reward * 1.0,
            "shake": shake_reward * 1.5,
            "energy": energy_reward * 0.0001,
            "imitation": imitation_reward * 2.0,
        }

        reward = sum([o for o in weighted_objectives.values()])
        reward = reward * self._env.env_time_step
        return reward, weighted_objectives

    @staticmethod
    def to_local_frame(dx, dy, yaw):
        # Transform the x and y direction distances to the robot's local frame
        dx_local = np.cos(yaw) * dx + np.sin(yaw) * dy
        dy_local = -np.sin(yaw) * dx + np.cos(yaw) * dy
        return dx_local, dy_local
