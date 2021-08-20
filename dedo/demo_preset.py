"""
A simple demo with preset trajectories.

TODO(Yonk): clean up this file when trajectories are finalized.
tODO(Yonk): remove old control code, use simple trajectory interpolation.

@pyshi

"""
import matplotlib.pyplot as plt
from matplotlib import interactive

interactive(True)
import numpy as np

import gym

from dedo.utils.args import get_args
from dedo.utils.anchor_utils import create_anchor_geom
from dedo.utils.waypoint_utils import create_trajectory, interpolate_waypts
import os
import cv2
WRITE_TO_VID = True
preset_traj = {
    # TODO add constraint to scene name
    'cloth/apron_0.obj': {  # HangCloth-v0, 600 steps
        'waypoints': {
            'a': [
                # [ x, y, z, seconds(time)]
                # [2, 3.5, 11, 1],  # waypoint 0
                # [2, 2, 10.5, 1],  # waypoint 0
                [2, -0.5, 9.5, 1],
                [2, -1, 8, 1],
            ],
            'b': [
                # [ x, y, z, seconds(time)]
                # [-2, 3.5, 11, 1],  # waypoint 0
                # [-2, 2, 10.5, 1],  # waypoint 0
                [-2, -0.5, 9.5, 1],
                [-2, -1, 8, 1],
            ],
        },
    },
    'cloth/shirt_0.obj': {  # HangCloth-v5, 1500 steps
        'waypoints': {
            'a': [
                # [ x, y, z, seconds(time)]
                # [0.173 , 2.4621 ,6.5115, 1],
                [2.5 ,  -0.5 ,9, 3],
                [1 ,  -1 ,10.5, 1],
[1 ,  -1 ,8, 1],


            ],
            'b': [
                # [ x, y, z, seconds(time)]
                # [-0.1283 , 2.4621 , 6.5115, 1]
                [0, 0.5 , 9, 3],
                [-2, 0 , 10.5, 1],
                [-4, 0, 10.5, 3],
[-3, -1, 10.5, 0.5],
[-1, -1, 10.5, 1],
[-1, -1, 9, 0.5],


            ]
        },
    },
    'cloth/button_cloth.obj': {  # ButtonSimple-v0, 800 steps
        'waypoints': {
            'a': [
                # [ x, y, z, timesteps]
                [2.9, 00, 3.6, 1],
[2.9, 0, 3.6, 3],

            ],
            'b': [
                # [ x, y, z, timesteps]
                [2.9, 0, 1.6, 1],
[2.9, 0, 1.6, 3],


            ]
        },
    },
    'bags/bags_zehang/bag1_0.obj': {  # HangBag-v0, 1500 steps
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
    'cloth/cardigan_0.obj': {  # Dress-v5 #
        'waypoints': {
            'a': [
                # [-0.278 ,  1.7888,  6.245 ],
                [0.6, 1.7888, 6.245, 0.6],
                # [0.6 ,  1.1,  6.245, 100 ],
                # [0.6 ,  0.8,  6.245, 100 ],
                [0.6, 0.0, 6.245, 0.8],
                [0, 0, 6.445, 1.2],

            ],
            'b': [
                # [0.3004, 1.7888, 6.245 ]
                [2.3, 1.7888, 6.245, 0.6],
                # [2.3, 0.5, 6.245, 100 ],
                # [2.3, -0.3, 6.245, 100 ],
                [2.8, -1, 6.245, 0.8],
                [-0.5, -4, 6.245, 1.2],
                [-3, -1, 6.245, 0.6],
                [-2, 2, 6.245, 0.8],
                [-1, 2, 6.245, 0.2],
                # [-1, 2, 6.245, 0.6 ],
                [-1, 0, 6.245, 0.6],

            ]
        },
    },
    'bags/backpack_0.obj': {  # Dress-v0
        'waypoints': {
            'a': [
                [-0.8019, 0.9734, 4.0249, 1],
                [0.3, 1.7888, 6.5245, 2],
                [0.3, 0.0, 6.9245, 3],
                [0.8, -0.5, 6.945, 1.2],

            ],
            'b': [
                [0.1823, 0.9618, 4.4351, 1],
                [3.7, 1.7888, 6.945, 2],
                [3.7, -1, 6.945, 3],
                [0, -3, 7.245, 2],
                [-3, -1, 6.845, 0.6],
                [-2, 2, 6.845, 0.8],
                [-2, 1, 6.845, 0.6],

            ]
        },
    },
    'cloth/mask_0.obj': {  # Mask-v0
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
                [-0.9, 0, 7.3, 2],
                [-1.3, -1, 7.1, 0.5],
                [-0.6, -1.2, 7.1, 0.5],
                [-0.6, -1.2, 6.8, 0.5],

            ]
        },
    },
    'ropes/lasso3d_0_fast.obj': {
        'waypoints': {
            'a': [
                # [ [-2.8518, -0.2436 , 5.9087]],
                [-2.8518, -0.2436, 5.9087, 3],
                [-0.3518, -0.0, 8, 3],
                [2, 0, 8, 1],
                [2, 0, 1, 1],

            ],
            'b': [
                # [-3.6025, -0.3533,  5.9768]
                [-3.6025, -0.3533, 5.9768, 3],
                [-1.1025, -0.0, 8, 3],
                [1.3, 0, 8, 1],
                [1.3, 0, 1, 1],

            ]
        },
    },
    'ropes/hoop3d_0.obj': {
        'waypoints': {
            'a': [
                # [1.5708, 1.9639, 1.1152],
                [1.5708, 1.9639, 6, 1.1],
                [-0.5, -0.3, 6, 1.1],
                [-0.5, -0.3, 1, 1.1],
            ],
            'b': [
                # [2.2903, 1.9168, 5.9768]
                [2.2903, 1.9168, 6, 1.1],
                [-0.2, -0.5, 6, 1.1],
                [-0.2, -0.5, 1, 1.1],

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
    assert deform_obj in preset_traj, \
        f'The environment name / deform obj "{deform_obj}" does not exist  ' \
        f'in presets. Available keys: {preset_traj.keys()}'
    preset_wp = preset_traj[deform_obj]['waypoints']

    if WRITE_TO_VID:
        savepath = os.path.join(args.logdir, f'{args.env}.mp4')
        vidwriter = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc(*'mp4v'), 24, (640, 480))
        print('saving to ', savepath)

    for epsd in range(num_episodes):
        print('------------ Play episode ', epsd, '------------------')
        obs = env.reset()
        if args.debug:
            viz_waypoints(env.sim, preset_wp['a'], (1, 0, 0, 1))
            viz_waypoints(env.sim, preset_wp['b'], (1, 0, 0, 0.5))
        # Need to step to get low-dim state from info.
        step = 0
        ctrl_freq = args.sim_freq / args.sim_steps_per_action
        traj_a = build_traj(env, preset_wp, 'a', anchor_idx=0, ctrl_freq=ctrl_freq)
        traj_b = build_traj(env, preset_wp, 'b', anchor_idx=1, ctrl_freq=ctrl_freq)
        # traj_b = np.zeros_like(traj_b)
        traj = merge_traj(traj_a, traj_b)
        gif_frames = []
        while True:
            assert (not isinstance(env.action_space, gym.spaces.Discrete))

            act = traj[step] if step < len(traj) else np.zeros_like(traj[0])

            next_obs, rwd, done, info = env.step(act, unscaled_velocity=True)
            if WRITE_TO_VID:
                obs = env.render(mode='rgb_array', width=640, height=480)
                bgr_obs = obs[...,::-1]
                vidwriter.write(bgr_obs)
            # gif_frames.append(obs)
            if step > len(traj) + 50: break;
            # if done: break;
            obs = next_obs
            step += 1
        if WRITE_TO_VID:
            vidwriter.release()

def merge_traj(traj_a, traj_b):
    if traj_a.shape[0] != traj_b.shape[0]:  # padding is required
        n_pad = np.abs(traj_a.shape[0] - traj_b.shape[0])
        zero_pad = np.zeros((n_pad, traj_a.shape[1]))
        if traj_a.shape[0] > traj_b.shape[0]:  # pad b
            traj_b = np.concatenate([traj_b, zero_pad, ], axis=0)
        else:  # pad a
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
