from gym.envs.registration import register
from .envs.deform_env import DeformEnv
from .utils.task_info import TASK_TYPES

for task in TASK_TYPES:
    register(id=task+'-v0', entry_point='dedo.envs:DeformEnv')
