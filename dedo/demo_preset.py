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

preset_traj = {
    # TODO add constraint to scene name
    'cloth/apron_3_dense.obj': {
        'waypoints': {
            'left': [
                # [ x, y, z, timesteps]
                [-0.2, 0.1, 0.7, 80],  # waypoint 0
                [-0.2, 0.2, 0.7, 200],  # waypoint 1
            ],
            'right': [
                # [ x, y, z, timesteps]
                [0.2, 0.1, 0.7, 80],  # waypoint 0
                [0.2, 0.2, 0.7, 200],  # waypoint 1
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

        # waypoint subcounts
        i_wp_l = 0
        i_wp_r = 0
        wp_l = 0
        wp_r = 0

        while True:
            assert (not isinstance(env.action_space, gym.spaces.Discrete))
            # act = env.action_space.sample()  # in [-1,1]
            # noise_act = 0.1 * act
            # act = policy_simple(obs, noise_act, args.task)
            if preset_wp['left'][wp_l][-1] <= i_wp_l:
                i_wp_l = 0
                wp_l += 1
            if preset_wp['right'][wp_r][-1] <= i_wp_r:
                i_wp_r = 0
                wp_r += 1
            wp_left = np.array(preset_wp['left'][wp_l])
            wp_right = np.array(preset_wp['right'][wp_r])
            tgt_vel_l = target_pos_to_velocity_controller(env.sim, env.anchor_ids[0], wp_left[:3], wp_left[-1] - i_wp_l)
            tgt_vel_r = target_pos_to_velocity_controller(env.sim, env.anchor_ids[0], wp_right[:3], wp_right[-1] - i_wp_r)
            tgt_vel = np.concatenate([tgt_vel_l, tgt_vel_r])

            # print(tgt_vel_l)
            next_obs, rwd, done, info = env.step(tgt_vel, absolute_velocity=True)

            if args.viz and (args.cam_resolution is not None) and step % 100 == 0:
                img = next_obs
                print('img', img.shape)
                plt.imshow(img)
            if done:
                break
            obs = next_obs
            step += 1
            i_wp_l += 1
            i_wp_r += 1
        input('Episode ended; press enter to go on')


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
