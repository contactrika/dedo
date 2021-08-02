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
    'cloth/apron_3_dense.obj': {
        'waypoints': {
            'left': [
                # [ x, y, z, timesteps]
                [-2, 2, 7, 100],  # waypoint 0
                [-2, 1, 7, 100],
                [-2, -1, 7, 200],
            ],
            'right': [
                # [ x, y, z, timesteps]
                [2, 2, 7, 100],  # waypoint 0
                [2, 1, 7, 100],
                [2, -1, 7, 200],
            ]
        },
    },
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
    for epsd in range(num_episodes):
        print('------------ Play episode ', epsd, '------------------')
        obs = env.reset()

        preset_wp = preset_traj['cloth/apron_3_dense.obj']['waypoints']
        viz_waypoints(env.sim, preset_wp['left'], (1,0,0,1))
        viz_waypoints(env.sim, preset_wp['right'], (1, 0, 0, 1))
        # Need to step to get low-dim state from info.
        step = 0

        traj_a = build_traj(env, preset_wp, 'left', anchor_idx=0, sim_freq=args.sim_frequency)
        traj_b = build_traj(env, preset_wp, 'right', anchor_idx=1, sim_freq=args.sim_frequency)
        # traj_b = np.zeros_like(traj_b)
        traj = np.concatenate([ traj_a,traj_b,], axis=-1) # TODO async trajectories


        while True:
            assert (not isinstance(env.action_space, gym.spaces.Discrete))

            act = traj[step]
            next_obs, rwd, done, info = env.step(act, unscaled_velocity=True)
            img = env.render()
            plt.imshow(img)
            plt.show()

            if args.viz and (args.cam_resolution is not None) and step % 100 == 0:
                img = next_obs
                print('img', img.shape)
                plt.imshow(img)
            if done:
                break
            obs = next_obs
            step += 1

        input('Episode ended; press enter to go on')


def build_traj(env, preset_wp, left_or_right, anchor_idx, sim_freq):
    anc_id = list(env.anchors.keys())[anchor_idx]
    init_anc_pos = env.anchors[anc_id]['pos']
    print('init_anc_pos 0', init_anc_pos)
    wp = np.array(preset_wp[left_or_right])
    # Traditional wp
    pos = create_trajectory(init_anc_pos, wp[:, :3], wp[:, -1].astype('int32'), sim_freq)# [:,3:]
    # traj = pos[:, :3]
    # traj = pos[1:, :3] - pos[:-1, :3]
    traj = pos[:,3:]

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
