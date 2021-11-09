"""
An example of RL training using StableBaselines3.

python -m dedo.run_rl_sb3 --env=HangGarment-v1 --rl_algo PPO --logdir=/tmp/dedo

tensorboard --logdir=/tmp/dedo --bind_all --port 6006

Play the saved policy (e.g. logged to PPO_210825_204955_HangGarment-v1):
python -m dedo.run_rl_sb3 --env=HangGarment-v1 --play \
    --load_checkpt=/tmp/dedo/PPO_210825_204955_HangGarment-v1


Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika

"""
from copy import deepcopy
import os
import pickle
from pathlib import Path

import gym
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3  # used dynamically
from stable_baselines3.common.env_util import (
    make_vec_env, DummyVecEnv, SubprocVecEnv)

from dedo.utils.args import get_args
from dedo.utils.rl_sb3_utils import CustomCallback, play
from dedo.utils.train_utils import init_train

import numpy as np
import torch


def do_play(args, num_episodes=1):
    checkpt = os.path.join(args.load_checkpt, 'agent.zip')
    run_name = str(Path(args.load_checkpt).stem)
    print('Loading RL agent checkpoint from', checkpt)
    logdir = args.logdir
    cam_resolution = args.cam_resolution
    args = pickle.load(open(os.path.join(args.load_checkpt, 'args.pkl'), 'rb'))
    args.debug = True
    args.viz = logdir is None  # viz if not logging a video
    args.replay_size=10
    args.num_envs=0
    print('args', args)
    eval_env = gym.make(args.env, args=args)
    eval_env.seed(args.seed)
    rl_agent = eval(args.rl_algo).load(checkpt, buffer_size=10) #, env=eval_env
    play(eval_env, num_episodes=num_episodes, rl_agent=rl_agent, debug=False,
         logdir=logdir, cam_resolution=cam_resolution, filename=run_name)


def main(args):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if args.play:
        do_play(args)
        return  # no training, just playing
    assert(args.rl_algo is not None), 'Please provide --rl_algo'
    assert(args.rl_algo in ['A2C', 'DDPG', 'PPO', 'SAC', 'TD3']), \
        f'{args.rl_algo:s} not tested with SB3 (try RLlib)'
    # Init RL training envs and agent.
    args.logdir, args.device = init_train(args.rl_algo, args)
    # Stable Baselines3 only supports vectorized envs for on-policy algos.
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
    rl_kwargs = {'learning_rate': args.lr, 'device': args.device,
                 'tensorboard_log': args.logdir, 'verbose': 1}
    num_steps_between_save = args.log_save_interval*10
    if on_policy:
        num_steps_between_save *= 10  # less frequent logging
    if not on_policy and args.cam_resolution > 0:
        rl_kwargs['buffer_size'] = args.replay_size
    policy_name = 'MlpPolicy'
    if args.cam_resolution > 0 and args.uint8_pixels:
        policy_name = 'CnnPolicy'
    rl_agent = eval(args.rl_algo)(policy_name, vec_env, **rl_kwargs)
    cb = CustomCallback(eval_env, args.logdir, n_envs, args,
                        num_steps_between_save=num_steps_between_save,
                        viz=args.viz, debug=args.debug)
    print('Start SB3 RL training')
    rl_agent.learn(total_timesteps=args.total_env_steps, callback=cb)
    vec_env.close()


if __name__ == "__main__":

    main(get_args())
