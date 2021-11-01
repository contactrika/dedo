"""
A simple demo with preset trajectories.

python -m dedo.demo_preset --logdir=/tmp/dedo_preset --env=HangGarment-v1 \
  --max_episode_len=200 --robot anchor --cam_resolution 0 --viz


Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@yonkshi

"""
import os

import gym
from matplotlib import interactive
interactive(True)
import numpy as np

from dedo.utils.args import get_args
from dedo.utils.anchor_utils import create_anchor_geom
from dedo.utils.preset_info import preset_traj
import wandb
import cv2

from .utils.bullet_manipulator import convert_all

# TODO(yonkshi): remove this before release
from .internal.waypoint_utils import create_traj


def play(env, num_episodes, args):
    if args.task == 'ButtonProc':
        deform_obj = 'cloth/button_cloth.obj'
    elif args.task == 'HangProcCloth':
        deform_obj = 'cloth/apron_0.obj'
    elif args.task == 'FoodPacking':
        deform_obj = 'food'
    else:
        deform_obj = env.deform_obj

    assert deform_obj in preset_traj, \
        f'deform_obj {deform_obj:s} not in presets {preset_traj.keys()}'
    preset_wp = preset_traj[deform_obj]['waypoints']
    vidwriter = None
    if args.cam_resolution > 0 and args.logdir is not None:
        savepath = os.path.join(args.logdir, f'{args.env}_{args.seed}.mp4')
        vidwriter = cv2.VideoWriter(
            savepath, cv2.VideoWriter_fourcc(*'mp4v'), 24,
            (args.cam_resolution, args.cam_resolution))
        print('saving to ', savepath)
    if args.use_wandb:
        wandb.init(project='dedo', name=f'{args.env}-preset',
                   config={'env': f'{args.task}-preset'},
                   tags=['preset', args.env])

    for epsd in range(num_episodes):
        print('------------ Play episode ', epsd, '------------------')
        obs = env.reset()
        if args.cam_resolution > 0:
            img = env.render(mode='rgb_array', width=args.cam_resolution,
                             height=args.cam_resolution)
            vidwriter.write(img[..., ::-1])
        if args.debug:
            viz_waypoints(env.sim, preset_wp['a'], (1, 0, 0, 1))
            if 'b' in preset_wp:
                viz_waypoints(env.sim, preset_wp['b'], (1, 0, 0, 0.5))
        # Need to step to get low-dim state from info.
        step = 0
        ctrl_freq = args.sim_freq / args.sim_steps_per_action
        pos_traj, vel_traj = build_traj(
            env, preset_wp, 'a', anchor_idx=0, ctrl_freq=ctrl_freq)
        pos_traj_b, traj_b = None, None
        if 'b' in preset_wp:
            pos_traj_b, traj_b = build_traj(
                env, preset_wp, 'b', anchor_idx=1, ctrl_freq=ctrl_freq)
        if env.robot is None:
            traj = vel_traj
            if traj_b is not None:
                traj = merge_traj(traj, traj_b)
            last_action = np.zeros_like(traj[0])
        else:
            traj = pos_traj
            if pos_traj_b is not None:
                traj = merge_traj(pos_traj, pos_traj_b)
            last_action = traj[-1]
        traj_ori = preset_wp.get('a_theta', None)
        if traj_ori is not None:
            traj_ori = convert_all(np.array(traj_ori), 'theta_to_sin_cos')
            n_repeats = traj.shape[0]//len(traj_ori)
            traj_ori = np.repeat(traj_ori, n_repeats, axis=0)
            print('traj_ori', traj_ori.shape, traj_ori)
            assert(traj_ori.shape[0] == traj.shape[0])
            assert(traj_ori.shape[1] == 6)  # Euler sin,sin,sin,cos,cos,cos
            traj = np.hstack([traj, traj_ori])

        gif_frames = []
        rwds = []
        print(f'# {args.env}:')

        while True:
            assert (not isinstance(env.action_space, gym.spaces.Discrete))

            act = traj[step] if step < len(traj) else last_action

            next_obs, rwd, done, info = env.step(act, unscaled=True)
            rwds.append(rwd)
            if args.cam_resolution > 0:
                img = env.render(mode='rgb_array', width=args.cam_resolution,
                                 height=args.cam_resolution)
                vidwriter.write(img[..., ::-1])
            # gif_frames.append(obs)
            if done: break;
            # if step > len(traj) + 50: break;
            obs = next_obs

            step += 1

        print(f'episode reward: {env.episode_reward:.4f}')
        print('traj_length:', len(traj))
        if args.use_wandb:
            mean_rwd = np.sum(rwds)
            for i in range(31):
                wandb.log({'rollout/ep_rew_mean': mean_rwd, 'Step':i}, step=i)
        if args.cam_resolution > 0:
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
    if env.robot is None:
        anc_id = list(env.anchors.keys())[anchor_idx]
        init_anc_pos = env.anchors[anc_id]['pos']
    else:
        init_anc_pos = env.robot.get_ee_pos(left=anchor_idx>0)
    print(f'init_anc_pos {left_or_right}', init_anc_pos)
    wp = np.array(preset_wp[left_or_right])
    steps = (wp[:, -1] * ctrl_freq).round().astype(np.int32)  # seconds -> ctrl steps
    # TODO(yonkshi): replace create_traj with scipy.interpolate
    print('ATTENTION: Need to use scipy interpolate for preset trajs')
    exit(1)
    # WARNING: old code below.
    traj_pos_vel = create_traj(init_anc_pos, wp[:, :3], steps, ctrl_freq)
    pos_traj = traj_pos_vel[:, :3]
    vel_traj = traj_pos_vel[:, 3:]
    # plot_traj(pos_traj)
    from scipy.interpolate import interp1d
    # xi = interp1d(ids, waypoints[:, 0], kind='cubic')(interp_i)
    # yi = interp1d(ids, waypoints[:, 1], kind='cubic')(interp_i)
    # zi = interp1d(ids, waypoints[:, 2], kind='cubic')(interp_i)
    # traj = np.array([xi, yi, zi]).T
    return pos_traj, vel_traj


def plot_traj(traj):
    import matplotlib.pyplot as plt
    clrs = np.linspace(0, 1, traj.shape[0])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], c=clrs, cmap=plt.cm.jet)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_zlim3d(min(0, traj[:, 2].min()), traj[:, 2].max())
    plt.show()
    input('Continue')


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
