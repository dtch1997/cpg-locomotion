from dataclasses import dataclass

import numpy as np


@dataclass
class CPGParameters:
    a: float
    b: float
    mu: float
    alpha: float
    beta: float
    gamma: float
    period: float
    dt: float


class CPG:

    state_dim = 2

    def __init__(self, params: CPGParameters, initial_state: np.ndarray):
        self.params = params
        self.set_state(initial_state)

    def get_state(self):
        return self.state.copy()

    def set_state(self, state: np.ndarray):
        self.state = state.copy()

    def step(self, extrinsic_delta_state: np.ndarray):
        state = self.get_state()
        state += self._calc_intrinsic_delta_state() * self.params.dt
        state += extrinsic_delta_state * self.params.dt
        self.set_state(state)

    def get_phase(self):
        x, y = self.get_state()
        return np.arctan2(y, x)

    def _calc_radius(self):
        x, y = self.get_state()
        return np.sqrt(x ** 2 + y ** 2)

    def _calc_ang_vel(self):
        _, y = self.get_state()
        return (np.pi / self.params.beta / self.params.period / (np.exp(-self.params.b * y) + 1)) + (
            np.pi / (1 - self.params.beta) / self.params.period / (np.exp(self.params.b * y) + 1)
        )

    def _calc_intrinsic_delta_state(self):
        x, y = self.get_state()
        angvel = self._calc_ang_vel()
        delta_x = self.params.alpha * (self.params.mu ** 2 - self._calc_radius() ** 2) * x + angvel * y
        delta_y = self.params.alpha * (self.params.mu ** 2 - self._calc_radius() ** 2) * y - angvel * x
        return np.array([delta_x, delta_y])


class CPGSystem:
    def __init__(
        self,
        params: CPGParameters,
        coupling_strength: float,
        desired_phase_offsets: np.ndarray,
        initial_state: np.ndarray,
    ):
        self.num_cpgs = desired_phase_offsets.shape[0]
        self.params = params
        self._init_cpgs()
        self.set_state(initial_state)

        self.coupling_strength = coupling_strength
        self.set_phase_offsets(desired_phase_offsets)

    @staticmethod
    def sample_initial_state(desired_phase_offsets: np.ndarray):
        """Sample a random point obeying the phase offsets"""
        num_cpgs = desired_phase_offsets.shape[0]
        initial_phase = np.zeros(num_cpgs)
        initial_phase[0] = np.random.uniform(low=-np.pi, high=np.pi)
        for i in range(num_cpgs):
            initial_phase[i] = desired_phase_offsets[i] - desired_phase_offsets[0] + initial_phase[0]

        initial_state = np.zeros((num_cpgs, 2))
        for i in range(num_cpgs):
            initial_state[i, 0] = np.cos(initial_phase[i])
            initial_state[i, 1] = np.sin(initial_phase[i])
        return initial_state

    def get_state(self):
        return np.vstack([cpg.get_state() for cpg in self.cpgs])

    def set_state(self, state: np.ndarray):
        for i in range(self.num_cpgs):
            self.cpgs[i].set_state(state[i])

    def set_phase_offsets(self, desired_phase_offsets: np.ndarray):
        self.desired_phase_offsets = desired_phase_offsets
        # Recalculate coupling coefficients based on new target phase offsets
        self._init_coupling_coeff()

    def get_phase(self):
        return np.array([cpg.get_phase() for cpg in self.cpgs])

    def step(self):
        state = self.get_state()
        delta_state = np.zeros_like(state)
        for i in range(self.num_cpgs):
            # calculate coupling term in update eqn
            delta_state[i] = np.zeros_like(state[i])
            for j in range(self.num_cpgs):
                if j == i:
                    continue
                delta_state[i] += self.coupling_coeff[i, j] @ state[j]
        delta_state *= self.coupling_strength
        for i in range(self.num_cpgs):
            self.cpgs[i].step(extrinsic_delta_state=delta_state[i])

    def _init_cpgs(self):
        self.cpgs = [CPG(self.params, np.zeros(4)) for i in range(self.num_cpgs)]

    def _init_coupling_coeff(self):
        coupling_coeff = np.zeros((self.num_cpgs, self.num_cpgs, CPG.state_dim, CPG.state_dim))
        for i in range(self.num_cpgs):
            for j in range(self.num_cpgs):
                dpij = self.desired_phase_offsets[i] - self.desired_phase_offsets[j]
                coupling_coeff[i, j] = np.array([[np.cos(dpij), -np.sin(dpij)], [np.sin(dpij), np.cos(dpij)]])
        self.coupling_coeff = coupling_coeff
