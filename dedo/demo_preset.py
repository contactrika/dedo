"""
A simple demo for envs (with random actions).

python -m dedo.demo --env=HangCloth-v0 --viz --debug
python -m dedo.demo --env=HangBag-v1 --viz --debug

@contactrika

"""
import matplotlib.pyplot as plt
from matplotlib import interactive

interactive(True)
import numpy as np

import gym

from dedo.utils.args import get_args
from dedo.utils.anchor_utils import create_anchor_geom, target_pos_to_velocity_controller
from dedo.utils.waypoint_utils import create_trajectory, interpolate_waypts

preset_traj = {
    # TODO add constraint to scene name
    'cloth/apron_0.obj': {  # HangCloth-v0, 600 steps
        'waypoints': {
            'a': [
                # [ x, y, z, timesteps]
                [2, 3.5, 7.5, 24],  # waypoint 0
                [2, 2, 7, 24],  # waypoint 0
                [2, 1, 7, 24],
                [2, -1, 6, 48],
            ],
            'b': [
                # [ x, y, z, timesteps]
                [-2, 3.5, 7.5, 24],  # waypoint 0
                [-2, 2, 7, 24],  # waypoint 0
                [-2, 1, 7, 24],
                [-2, -1, 6, 48],
            ],
        },
    },
    'cloth/tshirt_0.obj': {  # HangCloth-v5, 1500 steps
        'waypoints': {
            'a': [
                # [ x, y, z, timesteps]
                [1.3356, 0.0892, 6, 100],
                [1.9356, 0.0892, 5, 100],
                [1.9356, 0.0892, 1, 500],
                [1.9356, 0.0892, 1, 200],
            ],
            'b': [
                # [ x, y, z, timesteps]
                [-1.3095, - 0.1382, 6, 100],
                [-1.9095, - 0.1382, 5, 100],
                [-1.9095, - 0.1382, 1, 500],
                [-1.9095, - 0.1382, 1, 200],
            ]
        },
    },
    'cloth/button_cloth.obj': {  # ButtonSimple-v0, 800 steps
        'waypoints': {
            'a': [
                # [ x, y, z, timesteps]
                [1.0, 2, 4.5, 100],
                [2.0, 1.8, 4.5, 50],
                [2, -3, 5, 50],
                [2, -10, 5, 50],
                [2, -10, 5, 100],
                [2, -10, 3.9, 50],
            ],
            'b': [
                # [ x, y, z, timesteps]
                [2, 2, 0.7, 100],
                [3, 1.8, 0.7, 50],
                [3, -3, 0, 50],
                [3, -10, 0.0, 50],
                [3, -10, 0.0, 100],
                [3, -10, 0.0, 50],

            ]
        },
    },
    'bags/bags_zehang/bag1_0.obj': {  # HangBag-v0, 1500 steps
        'waypoints': {
            'a': [
                [0.2, 2, 20, 500],
                [0.2, 2, 20, 500],
                [0.2, 1, 20, 100],
            ],
            'b': [
                [-0.2, 2, 20, 500],
                [-0.2, 2, 20, 500],
                [-0.2, 1, 20, 100],
            ]
        },
    },
    'cloth/cardigan_0.obj':{ # Dress-v5
        'waypoints': {
            'a': [
                # [-0.278 ,  1.7888,  6.245 ],
                [0.6 ,  1.7888,  6.245, 300 ],
                # [0.6 ,  1.1,  6.245, 100 ],
                # [0.6 ,  0.8,  6.245, 100 ],
                [0.6 ,  0.0,  6.245, 400 ],
                [0 ,  0,  6.445, 600 ],

            ],
            'b': [
                # [0.3004, 1.7888, 6.245 ]
                [2.3, 1.7888, 6.245, 300 ],
                # [2.3, 0.5, 6.245, 100 ],
                # [2.3, -0.3, 6.245, 100 ],
                [2.8, -1, 6.245, 400 ],
                [0, -3, 6.245, 450 ],
                [-2, 0, 6.245, 300 ],
                [-2, 2, 6.245, 400 ],
                [-1, 5, 6.245, 600 ],
                [-1, 5, 6.245, 300 ],
                [-1, 3, 6.245, 300 ],

            ]
        },
    },
    'cloth/mask_1.obj':{ # Mask-v0
        'waypoints': {
            'a': [
                # [0.4332, 1.9885, 6.1941],
                [0.6332, 1.3885, 6.1941, 100]

            ],
            'b': [
                # [-0.8332 , 1.9885 , 6.1941]
                [-1 , 1.3885 , 6.1941, 100]

            ]
        },
    },
    'ropes/lasso3d_0.obj':{
        'waypoints': {
            'a': [
                # [1.1346, 3.3335, 6.1546],
                [1.1346, 3.3335, 6.1546, 600],
                [1, 1.5,6, 300],
                [-0.5, -0.6, 4.7, 600],
                [-0.5, -0.6, 1, 300],

            ],
            'b': [
                # [-0.8025 , 1.6467 , 5.9768]
                [-0.8025 , 1.6467 , 5.9768, 600],
                [-1 , -1 , 4.9768, 800],
                [-1 , -1 , 1, 300],

            ]
        },
    }
}


def policy_simple(obs, act, task):
    act = act.reshape(2, 3)
    obs = obs.reshape(-1, 3)
    if task == 'Button':
        act[:, :] = 0.0
        if obs[0, 0] < 0.10:
            act[:, 0] = 0.10  # increase x
    elif task in ['Lasso', 'Hoop', 'Dress']:
        if obs[0, 1] > 0.0:
            act[:, 1] = -0.25  # decrease y
    elif obs[0, 2] > 0.50:
        act[:, 1] = -0.10  # decrease y
        act[:, 2] = -0.06  # decrease z
    return act.reshape(-1)


def viz_waypoints(sim, waypoints, rgba):
    waypoints = np.array(waypoints)
    for waypoint in waypoints:
        create_anchor_geom(sim, waypoint[:3], mass=0, rgba=rgba, use_collision=False)


def play(env, num_episodes, args):
    deform_obj = env.deform_obj
    assert deform_obj in preset_traj, f'The e environment name / deform obj "{deform_obj}" does not exist in presets. ' \
                                      f'Available keys: {preset_traj.keys()}'
    preset_wp = preset_traj[deform_obj]['waypoints']

    for epsd in range(num_episodes):
        print('------------ Play episode ', epsd, '------------------')
        obs = env.reset()
        if args.debug:
            viz_waypoints(env.sim, preset_wp['a'], (1, 0, 0, 1))
            viz_waypoints(env.sim, preset_wp['b'], (1, 0, 0, 0.5))
        # Need to step to get low-dim state from info.
        step = 0

        traj_a = build_traj(env, preset_wp, 'a', anchor_idx=0, sim_freq=args.ctrl_freq)
        traj_b = build_traj(env, preset_wp, 'b', anchor_idx=1, sim_freq=args.ctrl_freq)
        # traj_b = np.zeros_like(traj_b)
        traj = merge_traj(traj_a, traj_b)

        while True:
            assert (not isinstance(env.action_space, gym.spaces.Discrete))

            act = traj[step] if step < len(traj) else np.zeros_like(traj[0
                                                                    ])
            next_obs, rwd, done, info = env.step(act, unscaled_velocity=True)

            obs = next_obs
            step += 1

        input('Episode ended; press enter to go on')


def merge_traj(traj_a, traj_b):

    if traj_a.shape[0] != traj_b.shape[0]:      # padding is required
        n_pad = np.abs(traj_a.shape[0] - traj_b.shape[0])
        zero_pad = np.zeros((n_pad, traj_a.shape[1]))
        if traj_a.shape[0] > traj_b.shape[0]:   # pad b
            traj_b = np.concatenate([traj_b, zero_pad, ], axis=0)
        else:                                   # pad a
            traj_a = np.concatenate([traj_a, zero_pad, ], axis=0)

    traj = np.concatenate([traj_a, traj_b, ], axis=-1)
    return traj
def build_traj(env, preset_wp, left_or_right, anchor_idx, sim_freq):
    anc_id = list(env.anchors.keys())[anchor_idx]
    init_anc_pos = env.anchors[anc_id]['pos']
    print(f'init_anc_pos {left_or_right}', init_anc_pos)
    wp = np.array(preset_wp[left_or_right])
    # Traditional wp
    pos = create_trajectory(init_anc_pos, wp[:, :3], wp[:, -1].astype('int32'), sim_freq)  # [:,3:]
    # traj = pos[:, :3]
    # traj = pos[1:, :3] - pos[:-1, :3]
    traj = pos[:, 3:]

    # Using the savgol_filter
    # wp_pos = interpolate_waypts(wp, 1000)
    # traj = wp_pos[1:] - wp_pos[:-1]

    return traj


def main(args):
    np.set_printoptions(precision=4, linewidth=150, suppress=True)
    kwargs = {'args': args}
    env = gym.make(args.env, **kwargs)
    env.seed(env.args.seed)
    print('Created', args.task, 'with observation_space',
          env.observation_space.shape, 'action_space', env.action_space.shape)
    play(env, env.args.num_play_runs, args)
    env.close()


if __name__ == "__main__":
    main(get_args())
