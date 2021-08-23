import os

from gym.envs.registration import register
from .envs.deform_env import DeformEnv
from .utils.task_info import (
    TASK_INFO, TOTE_MAJOR_VERSIONS, TOTE_VARS_PER_VERSION, DEFORM_INFO)


for task, versions in TASK_INFO.items():
    # Dynamically add all tote bags.
    if task == 'HangBag':
        bp = os.path.dirname(__file__)
        bag_dir = os.path.join(bp, 'data/bags/totes')
        for fn in sorted(os.listdir(bag_dir)):
            obj_name = os.path.join('bags/totes/', fn)
            if obj_name.endswith('obj') and obj_name not in versions:
                versions.append(obj_name)
        assert(len(TASK_INFO['HangBag']) ==
               TOTE_MAJOR_VERSIONS*TOTE_VARS_PER_VERSION)  # sanity check
    if task == 'HangProcCloth':
        versions += versions  # add v2 for HangProcCloth
    if task == 'BGarments':
        bp = os.path.dirname(__file__)
        garm_dir = os.path.join(bp, 'data/berkeley_garments')
        for fn in sorted(os.listdir(garm_dir)):
            obj_name = os.path.join('berkeley_garments/', fn)
            if obj_name.endswith('obj') and obj_name not in versions:
                versions.append(obj_name)
    # Register v0 as random material textures and the rest of task versions.
    register(id=task+'-v'+str(0), entry_point='dedo.envs:DeformEnv')
    for version_id, obj_name in enumerate(versions):
        register(id=task+'-v'+str(version_id+1),
                 entry_point='dedo.envs:DeformEnv')
