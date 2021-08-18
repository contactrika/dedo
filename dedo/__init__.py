from gym.envs.registration import register
from .envs.deform_env import DeformEnv
from .utils.task_info import TASK_INFO
import numpy as np


for task, versions in TASK_INFO.items():
    # versions = versions.copy() # Not to modify the
    if task == 'HangProcCloth':
        # HangProcCloth have v0 and v1
        versions += versions
    elif task in ['HangCloth', 'HangBag', 'Mask', 'Dress', 'Hoop', 'Lasso', 'ButtonSimple']: # These tasks have v0 as random material textures
        obj_name = np.random.choice(versions)
        versions.insert( 0,obj_name,) # prepend a randomly sampled object to v0

    # TODO add v0 to all scenes
    # TODO add all 100+ hang bags
    for version_id, obj_name in enumerate(versions):
        register(id=task+'-v'+str(version_id), entry_point='dedo.envs:DeformEnv')
