from abc import ABC, abstractmethod


class EnvModifier(ABC):
    def __init__(self, adjust_position=(0, 0, 0), deformable=False, hardreset=False):
        self.adjust_position = adjust_position  # Adjust initial position of robot
        self.deformable = deformable  # Whether the modifier requires deformable physics
        self.hardreset = hardreset  # Whether the modifier requires hard reset

    @abstractmethod
    def _generate(self, env, **kwargs):
        """Generate environment modifier from scratch"""
        pass

    def _reset(self, env, **kwargs):
        """Reset environment modifier without regenerating it"""
        pass
