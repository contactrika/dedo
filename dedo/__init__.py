from gym.envs.registration import register
from .envs.deform_env import DeformEnv
from .utils.task_info import TASK_INFO

for task, versions in TASK_INFO.items():
    for version, obj_name in enumerate(versions):
        register(id=task+'-v'+str(version), entry_point='dedo.envs:DeformEnv')
