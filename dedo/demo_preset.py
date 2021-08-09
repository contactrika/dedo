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
                # [ x, y, z, seconds(time)]
                [2, 3.5, 7.5, 1],  # waypoint 0
                [2, 2, 7, 1],  # waypoint 0
                [2, 1, 7, 1],
                [2, -1, 6.3, 2],
            ],
            'b': [
                # [ x, y, z, seconds(time)]
                [-2, 3.5, 7.5, 1],  # waypoint 0
                [-2, 2, 7, 1],  # waypoint 0
                [-2, 1, 7, 1],
                [-2, -1, 6.3, 2],
            ],
        },
    },
    'cloth/tshirt_0.obj': {  # HangCloth-v5, 1500 steps # TODO flip it
        'waypoints': {
            'a': [
                # [ x, y, z, seconds(time)]
                # [1.5594, 1.5181, 6.6771, 1],

                [1.5594, 0.0, 20, 4],
                [1.5594, 0.0, 4, 4],
                [1.9356, 0.0892, 1, 2],
            ],
            'b': [
                # [ x, y, z, seconds(time)]
                # [-1.5158 , 1.897,  6.7119, 1]
                [-1.5158 , 0.0,  20, 4],
                [-1.5158 , 0.0,  4, 4],
                [-1.9095, - 0.1382, 1, 2],
            ]
        },
    },
    'cloth/button_cloth.obj': {  # ButtonSimple-v0, 800 steps
        'waypoints': {
            'a': [
                # [ x, y, z, timesteps]
                [1.0, 2, 4.5, 1],
                [2.0, 1.8, 4.5, 0.5],
                [2, -3, 5, 0.5],
                [2, -10, 5, 0.5],
                [2, -10, 5, 1],
                [2, -10, 3.9, 0.5],
            ],
            'b': [
                # [ x, y, z, timesteps]
                [2, 2, 0.7, 1],
                [3, 1.8, 0.7, 0.5],
                [3, -3, 0, 0.5],
                [3, -10, 0.0, 0.5],
                [3, -10, 0.0, 1],
                [3, -10, 0.0, 0.5],

            ]
        },
    },
    'bags/bags_zehang/bag1_0.obj': {  # HangBag-v0, 1500 steps # TODO make it larger
        'waypoints': {
            'a': [
                [0.2, 2, 10.4, 3],
                [0.2, 1, 10.4, 1],
            ],
            'b': [
                [-0.2, 2, 10.4, 3],
                [-0.2, 1, 10.4, 1],
            ]
        },
    },
    'cloth/cardigan_0.obj':{ # Dress-v5 #
        'waypoints': {
            'a': [
                # [-0.278 ,  1.7888,  6.245 ],
                [0.6 ,  1.7888,  6.245, 0.6 ],
                # [0.6 ,  1.1,  6.245, 100 ],
                # [0.6 ,  0.8,  6.245, 100 ],
                [0.6 ,  0.0,  6.245, 0.8 ],
                [0 ,  0,  6.445, 1.2 ],

            ],
            'b': [
                # [0.3004, 1.7888, 6.245 ]
                [2.3, 1.7888, 6.245, 0.6 ],
                # [2.3, 0.5, 6.245, 100 ],
                # [2.3, -0.3, 6.245, 100 ],
                [2.8, -1, 6.245, 0.8 ],
                [-0.5, -4, 6.245, 1.2 ],
                [-3, -1, 6.245, 0.6 ],
                [-2, 2, 6.245, 0.8 ],
                [-1, 2, 6.245, 0.2 ],
                # [-1, 2, 6.245, 0.6 ],
                [-1, 0, 6.245, 0.6 ],

            ]
        },
    },
    'bags/backpack_0.obj':{ # Dress-v0
        'waypoints': {
            'a': [
                [-0.8019, 0.9734, 4.4249, 1],
                [0.3 ,  1.7888,  6.9245, 2],
                [0.3 ,  0.0,    6.9245, 3 ],
                [0.8 ,  -0.5,  6.945, 1.2 ],

            ],
            'b': [
                [0.1823, 0.9618, 4.4351,1],
                [3.7, 1.7888, 6.945, 2 ],
                [3.7, -1, 6.945, 3 ],
                [0, -3, 7.245, 2 ],
                [-3, -1, 6.845, 0.6 ],
                [-2, 2, 6.845, 0.8 ],
                [-2, 1, 6.845, 0.6 ],

            ]
        },
    },
    'cloth/mask_0.obj':{ # Mask-v0 # TODO fix models and the creepy head
        'waypoints': {
            'a': [
                # [0.4332, 1.9885, 6.1941],
                [0.5, 0, 7.3, 2],
                [0.8, -1, 7.1, 0.5],
                [0.3, -1, 7.1, 0.5],
                [0.3, -1, 7.1, 0.5],
                [0.3, -1, 6.7, 0.5],
            ],
            'b': [
                # [-0.8332 , 1.9885 , 6.1941]
                [-0.9, 0, 7.3, 2 ],
                [-1.3, -1, 7.1, 0.5 ],
                [-0.6, -1.2, 7.1, 0.5 ],
[-0.6, -1.2, 6.8, 0.5 ],

            ]
        },
    },
    'ropes/lasso3d_0.obj':{
        'waypoints': {
            'a': [
                # [1.1346, 3.3335, 6.1546],
                [1.1346, 3.3335, 6.1546, 1.1],
                [1, 1.5,6, 0.5],
                [-0.5, -0.6, 4.7, 1.1],
                [-0.5, -0.6, 1, 0.5],

            ],
            'b': [
                # [-0.8025 , 1.6467 , 5.9768]
                [-0.8025 , 1.6467 , 5.9768, 1.1],
                [-1 , -1 , 4.9768, 1.5],
                [-1 , -1 , 1, 0.5],

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

        traj_a = build_traj(env, preset_wp, 'a', anchor_idx=0, ctrl_freq=args.ctrl_freq)
        traj_b = build_traj(env, preset_wp, 'b', anchor_idx=1, ctrl_freq=args.ctrl_freq)
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
def build_traj(env, preset_wp, left_or_right, anchor_idx, ctrl_freq):
    anc_id = list(env.anchors.keys())[anchor_idx]
    init_anc_pos = env.anchors[anc_id]['pos']
    print(f'init_anc_pos {left_or_right}', init_anc_pos)
    wp = np.array(preset_wp[left_or_right])
    # Traditional wp
    steps = (wp[:, -1] * ctrl_freq).round().astype(np.int32)  # seconds -> ctrl steps
    pos = create_trajectory(init_anc_pos, wp[:, :3], steps, ctrl_freq)  # [:,3:]
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
