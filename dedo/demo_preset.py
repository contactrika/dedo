"""
A simple demo with preset trajectories.

TODO(Yonk): clean up this file when trajectories are finalized.
tODO(Yonk): remove old control code, use simple trajectory interpolation.

@pyshi

"""
import os

import gym
from matplotlib import interactive
interactive(True)
import numpy as np

from dedo.utils.args import get_args
from dedo.utils.anchor_utils import create_anchor_geom
from dedo.utils.waypoint_utils import create_traj, create_traj_savgol
from dedo.utils.preset_info import preset_traj

WRITE_TO_VID = True
if WRITE_TO_VID:
    import cv2


def play(env, num_episodes, args):
    if args.task == 'ButtonProc':
        deform_obj = 'cloth/button_cloth.obj'
    elif args.task == 'HangProcCloth':
        deform_obj = 'cloth/apron_0.obj'
    else:
        deform_obj = env.deform_obj

    assert deform_obj in preset_traj, \
        f'deform_obj {deform_obj:s} not in presets {preset_traj.keys()}'
    preset_wp = preset_traj[deform_obj]['waypoints']

    if WRITE_TO_VID:
        savepath = os.path.join(args.logdir, f'{args.env}.mp4')
        vidwriter = cv2.VideoWriter(
            savepath, cv2.VideoWriter_fourcc(*'mp4v'), 24, (640, 480))
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
        rwds = []
        while True:
            assert (not isinstance(env.action_space, gym.spaces.Discrete))

            act = traj[step] if step < len(traj) else np.zeros_like(traj[0])

            next_obs, rwd, done, info = env.step(act, unscaled_velocity=True)
            rwds.append(rwd)
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


def viz_waypoints(sim, waypoints, rgba):
    waypoints = np.array(waypoints)
    for waypoint in waypoints:
        create_anchor_geom(sim, waypoint[:3], mass=0, rgba=rgba, use_collision=False)


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
    pos = create_traj(init_anc_pos, wp[:, :3], steps, ctrl_freq)  # [:,3:]
    traj = pos[:, 3:]
    #
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
    play(env, 1, args)
    env.close()


if __name__ == "__main__":
    main(get_args())
