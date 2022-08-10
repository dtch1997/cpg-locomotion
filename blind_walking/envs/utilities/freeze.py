from copy import deepcopy


class freeze_a1_gym_env:
    """Provides a context to freeze an A1GymEnv-v0 environment.
    This context allows the user to manipulate the state of the A1GymEnv-v0 environment and return it
    to its original state upon exiting the context.

    Example usage:
    .. code-block:: python
       env = A1GymEnv()
       env.reset()
       action = env.action_space.sample()
       # o1_expected, *_ = env.step(action)
       with freeze_a1_gym_env(env):
           step_the_env_a_bunch_of_times()
       o1, *_ = env.step(action) # o1 will be equal to what o1_expected would have been
    Args:
        env: the environment to freeze.
    """

    def __init__(self, env):
        self._env = env

    def __enter__(self):
        self._state_id = self._env.get_state()

    def __exit__(self, *_args):
        self._env.set_state(self._state_id)
