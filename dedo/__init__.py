from gym.envs.registration import register
from deform_env import DeformEnv


for task in DeformEnv.TASK_NAMES:
    for obs_type in ['', 'RGB']:
        for resolution in ['64', '200']:
            obs = obs_type
            if obs == 'RGB':
                obs += str(resolution)
            task_id = task+'_'+obs+'-v0'
            print('register', task_id)
            register(id=task_id, entry_point='dedo.envs:DeformEnv',
                     task_name=task, resolution=resolution)
