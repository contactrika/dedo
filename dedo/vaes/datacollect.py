"""
A simple demo with an example of RL training using Stable Baselines.

python -m dedo.rl_demo --env=HangGarment-v1 --rl_algo PPO --logdir=/tmp/dedo

tensorboard --logdir=/tmp/dedo --bind_all --port 6006


@contactrika

"""
from copy import deepcopy
from datetime import datetime
import os
import platform
if platform.system() == 'Linux':
    os.environ['IMAGEIO_FFMPEG_EXE'] = '/usr/bin/ffmpeg'

import numpy as np

import gym
from stable_baselines3 import A2C, DDPG, HER, PPO, SAC, TD3  # used dynamically
from stable_baselines3.common.env_util import (
    make_vec_env, DummyVecEnv, SubprocVecEnv)
import wandb

from dedo.utils.args import get_args
from dedo.utils.rl_utils import CustomCallback
import matplotlib.pyplot as plt


def main(args):
    np.set_printoptions(precision=4, linewidth=150, suppress=True)
    rl_tot_steps = int(1e6)
    if args.rl_algo is None:
        args.rl_algo = 'PPO'  # default to PPO if RL algo not specified
    logdir = None
    if args.logdir is not None:
        tstamp = datetime.strftime(datetime.today(), '%y%m%d_%H%M%S')
        subdir = '_'.join([args.rl_algo, tstamp, args.env])
        logdir = os.path.join(os.path.expanduser(args.logdir), subdir)
        if args.use_wandb:
            wandb.init(config=vars(args), project='dedo', name=logdir)
            wandb.init(sync_tensorboard=False)
            wandb.tensorboard.patch(tensorboardX=True, pytorch=True)
    # Stable baselines only support vectorized envs for on-policy algos.
    on_policy = args.rl_algo in ['A2C', 'PPO']
    n_envs = args.num_envs if on_policy else 1
    eval_env = gym.make(args.env, args=args)
    eval_env.seed(args.seed)
    train_args = deepcopy(args)
    train_args.debug = False  # no debug during training
    train_args.viz = False  # no viz during training
    vec_env = make_vec_env(
        args.env, n_envs=n_envs,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
        env_kwargs={'args': train_args})
    vec_env.seed(args.seed)
    obs = vec_env.reset()
    # Policy
    print('obs shape', obs.shape)
    vec_env.close()


if __name__ == "__main__":
    # TODO Add out dir
    # TODO Add options for camera angle shift
    # TODO Add options for start position varation
    main(get_args())
