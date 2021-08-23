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


def main(args):
    np.set_printoptions(precision=4, linewidth=150, suppress=True)
    if args.rl_algo is None:
        args.rl_algo = 'PPO'  # default to PPO if RL algo not specified
    if args.cam_resolution > 0:
        args.uint8_pixels = True  # for CnnPolicy
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
    print('Created', args.task, 'with observation_space',
          vec_env.observation_space.shape, 'action_space',
          vec_env.action_space.shape)
    rl_kwargs = {'learning_rate': args.rl_lr, 'device': args.device,
                 'tensorboard_log': logdir, 'verbose': 1}
    num_steps_between_play = 10000 if on_policy else 1000
    if not on_policy:
        if args.cam_resolution > 0:
            rl_kwargs['buffer_size'] = 10000  # storing RGB frames in replay
    policy_name = 'CnnPolicy' if args.cam_resolution > 0 else 'MlpPolicy'
    rl_agent = eval(args.rl_algo)(policy_name, vec_env, **rl_kwargs)
    cb = CustomCallback(eval_env, args.num_play_runs, logdir, n_envs, args,
                        num_steps_between_play=num_steps_between_play,
                        viz=args.viz, debug=args.debug)
    print('RL training start')
    rl_agent.learn(total_timesteps=args.rl_total_env_steps, callback=cb)
    vec_env.close()


if __name__ == "__main__":
    main(get_args())
