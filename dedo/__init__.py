from gym.envs.registration import register
from .envs.deform_env import DeformEnv
from .utils.task_info import TASK_INFO


for task, versions in TASK_INFO.items():
    if task == 'HangProcCloth':
        # HangProcCloth have v0 and v1
        versions += versions
    # TODO add v0 to all scenes
    # TODO add all 100+ hang bags
    for version, obj_name in enumerate(versions):
        register(id=task+'-v'+str(version), entry_point='dedo.envs:DeformEnv')
