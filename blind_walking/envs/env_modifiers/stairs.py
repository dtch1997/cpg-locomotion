import pybullet as p
from blind_walking.envs.env_modifiers.env_modifier import EnvModifier

boxHalfLength = 0.5
boxHalfWidth = 10
boxHalfHeight = 0.1


class Stairs(EnvModifier):
    def __init__(self):
        self.stepShape = 0
        self.steps = []
        self.base_pos = [0, 0, 0]
        super().__init__()

    def _generate(
        self,
        env,
        start_x=3,
        num_steps=5,
        step_rise=0.1,
        step_run=0.3,
        friction=0.5,
        boxHalfLength=0.5,
        boxHalfWidth=10,
        pos_y=0,
    ):
        env.pybullet_client.configureDebugVisualizer(env.pybullet_client.COV_ENABLE_RENDERING, 0)

        stepShape = env.pybullet_client.createCollisionShape(
            p.GEOM_BOX, halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight]
        )
        colors = [[0.75, 0.6, 0.2, 1], [0.80, 0.6, 0.2, 1], [0.85, 0.6, 0.2, 1]]

        # Create upwards stairs
        base_pos = [start_x, pos_y, step_rise - boxHalfHeight]
        self.base_pos = base_pos
        for i in range(num_steps):
            step = env.pybullet_client.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=stepShape, basePosition=base_pos, baseOrientation=[0, 0, 0, 1]
            )
            self.steps.append(step)
            env.pybullet_client.changeDynamics(step, -1, lateralFriction=friction)
            env.pybullet_client.changeVisualShape(step, -1, rgbaColor=colors[i % len(colors)])
            base_pos = [sum(x) for x in zip(base_pos, [step_run, 0, step_rise])]

        # Create downwards stairs
        base_pos = [sum(x) for x in zip(base_pos, [boxHalfLength * 2 - step_run, 0, -step_rise])]
        for i in range(num_steps):
            step = env.pybullet_client.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=stepShape, basePosition=base_pos, baseOrientation=[0, 0, 0, 1]
            )
            self.steps.append(step)
            env.pybullet_client.changeDynamics(step, -1, lateralFriction=friction)
            env.pybullet_client.changeVisualShape(step, -1, rgbaColor=colors[(-i - 1) % len(colors)])
            base_pos = [sum(x) for x in zip(base_pos, [step_run, 0, -step_rise])]

        self.stepShape = stepShape
        env.pybullet_client.configureDebugVisualizer(env.pybullet_client.COV_ENABLE_RENDERING, 1)

    def _reset(self, env, step_rise=0.1, step_run=0.3):
        # Do not change the box shape but change the position of the steps
        base_pos = self.base_pos
        for i, step in enumerate(self.steps):
            p.resetBasePositionAndOrientation(step, base_pos, [0, 0, 0, 1])
            if i < len(self.steps) / 2 - 1:
                base_pos = [sum(x) for x in zip(base_pos, [step_run, 0, step_rise])]
            elif i == len(self.steps) / 2 - 1:
                base_pos = [sum(x) for x in zip(base_pos, [boxHalfLength * 2, 0, 0])]
            else:
                base_pos = [sum(x) for x in zip(base_pos, [step_run, 0, -step_rise])]
