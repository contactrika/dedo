from gym.envs.registration import register
from .envs.deform_env import DeformEnv
from .utils.task_info import TASK_INFO
import numpy as np
import os
for task, versions in TASK_INFO.items():

    # dynamically add all bags
    if task == 'HangBag':
        bp = os.path.dirname(__file__)
        bag_dir = os.path.join(bp,'data/bags/bags_zehang')
        for fn in sorted(os.listdir(bag_dir)):
            obj_name = os.path.join('bags/bags_zehang/', fn)
            if obj_name.endswith('obj') and obj_name not in versions:
                versions.append(obj_name)

    if task == 'HangProcCloth':
        # HangProcCloth have v0 and v1
        versions += versions
    # These tasks have v0 as random material textures
    register(id=task+'-v'+str(0), entry_point='dedo.envs:DeformEnv')
    for version_id, obj_name in enumerate(versions):
        register(id=task+'-v'+str(version_id+1), entry_point='dedo.envs:DeformEnv')
