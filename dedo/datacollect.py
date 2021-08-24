"""
A data colleciton script for storing trajectories into numpy files with a very basic preset trajectory

examples:
python -m dedo.vae.datacollect --env=ProcHangCloth-v0 --logdir=/tmp/
python -m dedo.vae.datacollect --cam_resolution=400 --env=ProcHangCloth-v0 --max_episode_len=999 --logdir=/tmp/  --dtype='float16'
python -m dedo.vae.datacollect --cam_resolution=400 --env=Sewing-v0 --max_episode_len=999 --logdir=/tmp/ --bundle_size=128 --action_noise=0.01 --dtype='float16'

@yonkshi

"""
from copy import deepcopy
from datetime import datetime
import os
from os.path import join
import platform
import argparse
import numpy as np
import torch
import cv2
import gym
from stable_baselines3 import A2C, DDPG, HER, PPO, SAC, TD3  # used dynamically
from stable_baselines3.common.env_util import (
    make_vec_env, DummyVecEnv, SubprocVecEnv)
import wandb

from dedo.utils.args import get_args_parser, args_postprocess
from dedo.utils.rl_utils import CustomCallback


def main(args):
    assert args.logdir is not None, '--logdir must be set for data collection'

    n_runs = int(np.ceil(args.max_episodes / args.num_envs))
    args.debug = False  # no debug during training
    args.viz = False  # no viz during training

    args.cam_viewmat = [6.2, 8.6, 328, -0.44, 1.37, 6.7]

    vec_env = make_vec_env(
        args.env, n_envs=args.num_envs,
        vec_env_cls=SubprocVecEnv if args.num_envs > 1 else DummyVecEnv,
        env_kwargs={'args': args})
    vec_env.seed(args.seed)
    bundle = None
    bundle_id = 0
    for i_run in range(n_runs):
        obs = vec_env.reset()



        # TODO Add fuzzy Camera position
        # TODO Add fuzzy Spawning  Noise
        act = np.zeros((args.num_envs, 6,))

        # Y-axis motion for both anchors
        act_noise = np.random.uniform(-args.action_noise,args.action_noise, args.num_envs)
        act[:, 1] += -0.3 + act_noise
        act[:, 4] += -0.3 + act_noise
        # Z-Axis motion
        act[:, 2] += 0.1 + act_noise
        act[:, 5] += 0.1 + act_noise
        done = False
        j = 0
        while not np.all(done):
            next_obs, rwd, done, info = vec_env.step(act)
            j+=1
            if j % 10  == 0 and j >= 70 and j < 130: # 90 - 130
                for i, ob in enumerate(next_obs):
                    ob = ob[...,::-1] * 255 # RGB2BGR
                    cv2.imwrite(join(args.logdir, f'{args.env}_obj{i_run * args.num_envs + i}_step{j}.png'), ob)
            if j>130:
                break

        continue
        next_obs = next_obs.astype(args.dtype)
        if bundle is None:
            bundle = next_obs
        else:
            bundle = np.concatenate((bundle, next_obs), axis=0)

        # Output
        if bundle.shape[0] >= args.bundle_size:
            # Clip bundle size
            bundle = bundle[:args.bundle_size]

            outfile = join(args.logdir, f'{args.env}_{bundle_id}.npy')
            print('Saving bundle:', outfile)
            np.save(outfile, bundle)

            bundle_id += 1
            bundle = None

        print(f'Runs: {i_run}/{n_runs}')

    vec_env.close()

def args_setup():
    args, main_parser = get_args_parser()
    parser = argparse.ArgumentParser(
        description="CollectData", parents=[main_parser], add_help=False)

    parser.add_argument('--bundle_size', type=int, default=128,
                        help='size of each numpy file bundle')
    parser.add_argument('--max_episodes', type=int, default=1000,
                        help='Maximum number episodes to collect')
    parser.add_argument('--dtype', type=str, default='float16',
                        help='Saved numpy file type')
    parser.add_argument('--action_noise', type=float, default=0,
                        help='Noise in the direction of force applied to each garment instance (We recommend a range around 0.01)')

    args, unknown = parser.parse_known_args()
    args_postprocess(args)
    return args
if __name__ == "__main__":
    main(args_setup())
