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
import time
import gym
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib import interactive

interactive(True)
import numpy as np

from dedo.utils.args import get_args
from dedo.utils.anchor_utils import create_anchor_geom
from dedo.utils.preset_info import preset_traj
from dedo.utils.pcd_utils import visualize_data, render_video
import wandb
import cv2

from dedo.utils.bullet_manipulator import convert_all


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
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
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
            if vidwriter is not None:
                vidwriter.write(img[..., ::-1])
        if args.debug:
            viz_waypoints(env.sim, preset_wp['a'], (1, 0, 0, 1))
            if 'b' in preset_wp:
                viz_waypoints(env.sim, preset_wp['b'], (1, 0, 0, 0.5))
        # Need to step to get low-dim state from info.
        step = 0
        ctrl_freq = args.sim_freq / args.sim_steps_per_action
        robot = None
        if hasattr(env, 'robot'):
            robot = env.robot
        pos_traj, vel_traj = build_traj(
            env, preset_wp, 'a', anchor_idx=0, ctrl_freq=ctrl_freq, robot=robot)
        pos_traj_b, traj_b = None, None
        if 'b' in preset_wp:
            pos_traj_b, traj_b = build_traj(env, preset_wp, 'b', anchor_idx=1,
                                            ctrl_freq=ctrl_freq, robot=robot)
        if robot is None:  # velocity control for anchors
            traj = vel_traj
            if traj_b is not None:
                traj = merge_traj(traj, traj_b)
            last_action = np.zeros_like(traj[0])
        else:  # position control for robot
            traj = pos_traj
            if pos_traj_b is not None:
                traj = merge_traj(pos_traj, pos_traj_b)
            last_action = traj[-1]
        traj_ori = preset_wp.get('a_theta', None)
        if traj_ori is not None:
            traj_ori = convert_all(np.array(traj_ori), 'theta_to_sin_cos')
            n_repeats = traj.shape[0] // len(traj_ori)
            traj_ori = np.repeat(traj_ori, n_repeats, axis=0)
            print('traj_ori', traj_ori.shape, traj_ori)
            assert (traj_ori.shape[0] == traj.shape[0])
            assert (traj_ori.shape[1] == 6)  # Euler sin,sin,sin,cos,cos,cos
            traj = np.hstack([traj, traj_ori])

        gif_frames = []
        rwds = []
        print(f'# {args.env}:')

        if args.pcd:
            pcd_fig = plt.figure(figsize=(10,5))
        while True:
            assert (not isinstance(env.action_space, gym.spaces.Discrete))

            act = traj[step] if step < len(traj) else last_action

            next_obs, rwd, done, info = env.step(act, unscaled=True)
            rwds.append(rwd)

            if done and vidwriter is not None:  # Record the internal steps after anchor drop
                for ob in info['final_obs'][1:]:
                    vidwriter.write(np.uint8(ob[..., ::-1] * 255))
            if args.cam_resolution > 0:
                img = env.render(mode='rgb_array', width=args.cam_resolution,
                                 height=args.cam_resolution)
                if vidwriter is not None:
                    vidwriter.write(img[..., ::-1])

                if args.pcd:
                    # Grab additional obs from the environment
                    pcd_obs = env.get_pcd_obs()
                    img, pcd, ids = pcd_obs.values()

                    os.makedirs(f"{args.logdir}/pcd", exist_ok=True) # tmpfolder
                    save_path = f'{args.logdir}/pcd/{step:06d}.png'
                    visualize_data(img, pcd, ids, fig=pcd_fig, 
                                        save_path=save_path)


            # gif_frames.append(obs)
            if done:
                break

            # if step > len(traj) + 50: break;
            obs = next_obs

            step += 1

        print(f'episode reward: {env.episode_reward:.4f}')
        print('traj_length:', len(traj))
        if args.use_wandb:
            mean_rwd = np.sum(rwds)
            for i in range(31):
                wandb.log({'rollout/ep_rew_mean': mean_rwd, 'Step': i}, step=i)
        if vidwriter is not None:
            vidwriter.release()

        if args.pcd:
            render_video(f'{args.logdir}/pcd', 
                                f'{args.logdir}/pcd_preset_test.mp4')            


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


def build_traj(env, preset_wp, left_or_right, anchor_idx, ctrl_freq, robot):
    if robot is not None:
        init_anc_pos = env.robot.get_ee_pos(left=anchor_idx > 0)
    else:
        anc_id = list(env.anchors.keys())[anchor_idx]
        init_anc_pos = env.anchors[anc_id]['pos']
    print(f'init_anc_pos {left_or_right}', init_anc_pos)
    wp = np.array(preset_wp[left_or_right])
    steps = (wp[:, -1] * ctrl_freq).round().astype(np.int32)  # seconds -> ctrl steps

    print('ATTENTION: Need to use scipy interpolate for preset trajs')
    # exit(1)
    # WARNING: old code below.

    from scipy.interpolate import interp1d
    wpt = np.concatenate([[init_anc_pos], wp[:, :3]], axis=0)
    ids = np.arange(wpt.shape[0])
    interp_type = 'linear'
    # Creates the respective time interval for each way point
    interp_i = []
    for i, num_step in enumerate(steps):
        interp_i.append(np.linspace(i, i + 1, num_step, endpoint=False))

    interp_i = np.concatenate(interp_i)
    # interp_i = np.linspace(0, 1, steps[0], endpoint=False) # np.arange(0, wpt.shape[0]-1, 0.1)
    xi = interp1d(ids, wpt[:, 0], kind=interp_type)(interp_i)
    yi = interp1d(ids, wpt[:, 1], kind=interp_type)(interp_i)
    zi = interp1d(ids, wpt[:, 2], kind=interp_type)(interp_i)

    traj = np.array([xi, yi, zi]).T

    dv = (traj[1:] - traj[:-1])  # * ctrl_freq

    # Calculating the avg velocity for each control step
    chunks = []
    chunk_size = int(np.round(ctrl_freq))
    start = 0
    for i in range(99999):

        if start + chunk_size > dv.shape[0]:
            # last chunk
            chunk_size = dv.shape[0] - start
        chunk = dv[start:start + chunk_size]
        mean_chunk = np.sum(chunk, axis=0, keepdims=True)
        mean_chunk = np.repeat(mean_chunk, chunk_size, axis=0, )  # scale back to original shape
        chunks.append(mean_chunk)
        start = start + chunk_size
        if start >= dv.shape[0]:
            break

    # Add the last point:
    chunks = chunks + [[chunks[-1][-1]]]
    velocities = np.concatenate(chunks, axis=0)

    return traj, velocities


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
