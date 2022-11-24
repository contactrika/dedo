"""
A simple demo for envs (with random actions).

python -m dedo.demo --env=HangGarment-v1 --viz --debug
python -m dedo.demo --env=HangBag-v1 --viz --debug


Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika

"""
import matplotlib.pyplot as plt
import os
from matplotlib import interactive

interactive(True)
import numpy as np

import gym

from dedo.utils.args import get_args
from dedo.utils.pcd_utils import visualize_data, render_video

def policy_simple(obs, act, task, step):
    """A very simple default policy."""
    act = act.reshape(2, 3)
    obs = obs.reshape(-1, 3)
    if task == 'Button':
        act[:, :] = 0.0
        if obs[0, 0] < 0.10:
            act[:, 0] = 0.10  # increase x
    elif task in ['HangGarment', 'HangProcCloth']:
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
    elif obs[0, 2] > 0.50:
        act[:, 1] = -0.30  # decrease y
        act[:, 2] = 0.1  # decrease z
    return act.reshape(-1)


def play(env, num_episodes, args):
    for epsd in range(num_episodes):
        print('------------ Play episode ', epsd, '------------------')
        obs = env.reset()
        step = 0
        # input('Reset done; press enter to start episode')
        if args.pcd:
            pcd_fig = plt.figure(figsize=(10,5))
        while True:
            assert (not isinstance(env.action_space, gym.spaces.Discrete))
            print('step', step)
            act = env.action_space.sample()  # in [-1,1]
            noise_act = 0.1 * act
            act = policy_simple(obs, noise_act, args.task, step)

            next_obs, rwd, done, info = env.step(act)

            if args.pcd:
                # Grab additional obs from the environment
                pcd_obs = env.get_pcd_obs()
                img, pcd, ids = pcd_obs.values()
                if args.cam_resolution > 0:
                    os.makedirs(f"{args.logdir}/pcd", exist_ok=True) # tmpfolder
                    save_path = f'{args.logdir}/pcd/{step:06d}.png'
                    visualize_data(img, pcd, ids, fig=pcd_fig, save_path=save_path)

            if args.viz and (args.cam_resolution > 0) and step % 10 == 0:
                if not args.pcd: # Other visual for pcd mode
                    plt.imshow(next_obs)

            if done:
                break
            obs = next_obs
            step += 1
        input('Episode ended; press enter to go on')
        
        if args.pcd:
            render_video(f'{args.logdir}/pcd', f'{args.logdir}/pcd_test.mp4')            

        env.close()


def main(args):
    assert ('Robot' not in args.env), 'This is a simple demo for anchors only'
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
