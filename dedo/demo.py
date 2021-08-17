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


def policy_simple(obs, act, task, step):
    act = act.reshape(2, 3)
    act = np.zeros_like(act) # TODO delete me
    # # act[:, 0] = 1  # z axis
    # act[0, 0] = 1  # z axis
    # act[1, 0] = -1  # z axis
    # # act[:, 1] = -0.2 # -0.2  # y axis
    # act[:, 2] = -1 # -0.01  # z axis
    # return act.reshape(-1)
    obs = obs.reshape(-1, 3)
    if task == 'Button':
        act[:, :] = 0.0
        if obs[0, 0] < 0.10:
            act[:, 0] = 0.10  # increase x
    elif task in ['HangCloth', 'HangProcCloth']:
        act[:, 1] = -0.2

    elif task in ['HangBag']:
        # Dragging T Shirt
        act[:, 1] = -0.5
        act[:, 2] = 0.6
    elif task in ['Dress']:
        act[:, 1] = -0.2
        act[:, 2] = 0.1
    elif task in ['Lasso', 'Hoop']:
        if obs[0, 1] > 0.0:
            act[:, 1] = -0.25  # decrease y
            act[:, 2] = -0.25  # decrease z
    # elif obs[0, 2] > 0.50:
        # act[:, 1] = -0.10  # decrease y
        # act[:, 2] = -0.06  # decrease z
    return act.reshape(-1)


def play(env, num_episodes, args):
    for epsd in range(num_episodes):
        print('------------ Play episode ', epsd, '------------------')
        obs = env.reset()
        # Need to step to get low-dim state from info.
        step = 0
        # input('Reset done; press enter to start episode')
        while True:
            assert(not isinstance(env.action_space, gym.spaces.Discrete))
            print('step', step)
            act = env.action_space.sample()  # in [-1,1]
            noise_act = 0.1*act
            act = policy_simple(obs, noise_act, args.task, step)
            next_obs, rwd, done, info = env.step(act)
            if args.viz and (args.cam_resolution is not None) and step%100==0:
                img = next_obs
                print('img', img.shape)
                plt.imshow(img)
            if done:
                break
            obs = next_obs
            step += 1

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
