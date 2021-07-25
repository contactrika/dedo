"""
A simple demo with an example of PPO training using Stable Baselines RL.

python -m dedo.rl_demo --env=HangCloth-v0 --rl_algo TD3 \
    --logdir=/tmp/dedo --num_play_runs=3 --viz --debug

tensorboard --logdir=/tmp/dedo --bind_all --port 6006 \
  --samples_per_plugin images=1000


@contactrika

"""
from datetime import datetime
import os

import numpy as np

import gym
from stable_baselines3 import A2C, DDPG, HER, PPO, SAC, TD3  # used dynamically
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv

from dedo.utils.args import get_args
from dedo.utils.rl_utils import CustomCallback


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
    eval_debug = args.debug
    eval_viz = args.viz
    args.debug = False  # no debug during training
    args.viz = False  # no viz during training
    vec_env = make_vec_env(args.env, n_envs=8, vec_env_cls=SubprocVecEnv,
                           env_kwargs={'args': args})
    vec_env.seed(args.seed)
    args.debug = eval_debug
    args.viz = eval_viz
    eval_env = gym.make(args.env, args=args)
    eval_env.seed(args.seed)
    print('Created', args.task, 'with observation_space',
          vec_env.observation_space.shape, 'action_space',
          vec_env.action_space.shape)
    rl_agent = eval(args.rl_algo)('MlpPolicy', vec_env,
                                  tensorboard_log=logdir, verbose=1)
    cb = CustomCallback(eval_env, args.num_play_runs, logdir,
                        num_rollouts_between_play=2)
    rl_agent.learn(total_timesteps=rl_tot_steps, callback=cb)
    vec_env.close()


if __name__ == "__main__":
    main(get_args())
