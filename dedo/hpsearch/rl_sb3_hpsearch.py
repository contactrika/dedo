"""
An example of RL training using StableBaselines3.

python -m dedo.run_rl_sb3 --env=HangGarment-v1 --rl_algo PPO --logdir=/tmp/dedo

tensorboard --logdir=/tmp/dedo --bind_all --port 6006

Play the saved policy (e.g. logged to PPO_210825_204955_HangGarment-v1):
python -m dedo.run_rl_sb3 --env=HangGarment-v1 \
    --play=/tmp/dedo/PPO_210825_204955_HangGarment-v1

@contactrika

"""
import sys
from copy import deepcopy
import os
import pickle
import argparse

import gym
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3  # used dynamically
from stable_baselines3.common.env_util import (
    make_vec_env, DummyVecEnv, SubprocVecEnv)

from dedo.utils.args import get_args, get_args_parser, args_postprocess
from dedo.utils.rl_sb3_utils import CustomCallback, play
from dedo.utils.train_utils import init_train

import numpy as np
import torch
import wandb


def do_play(args, num_episodes=10):
    checkpt = os.path.join(args.load_checkpt, 'agent.zip')
    print('Loading RL agent checkpoint from', checkpt)
    args = pickle.load(open(os.path.join(args.load_checkpt, 'args.pkl'), 'rb'))
    args.debug = True
    args.viz = True
    eval_env = gym.make(args.env, args=args)
    eval_env.seed(args.seed)
    rl_agent = eval(args.rl_algo).load(checkpt)
    play(eval_env, num_episodes=num_episodes, rl_agent=rl_agent, debug=False)


def main():

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    args.base_logdir = args.logdir
    if args.play:
        do_play(args)
        return  # no training, just playing
    assert(args.rl_algo is not None), 'Please provide --rl_algo'
    assert(args.rl_algo in ['A2C', 'DDPG', 'PPO', 'SAC', 'TD3']), \
        f'{args.rl_algo:s} not tested with SB3 (try RLlib)'

    # Init RL training envs and agent.
    print('debug:INIT', file=sys.stderr)
    args.logdir, args.device = init_train(args.rl_algo, args, tags=['SB3', 'HPSearch', args.rl_algo, args.env])

    # Hyperparameter search results
    args.seed = wandb.config['seed']
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    print('random_seed', args.seed)


    # Stable Baselines3 only supports vectorized envs for on-policy algos.
    on_policy = args.rl_algo in ['A2C', 'PPO']
    n_envs = args.num_envs if on_policy else 1
    print('debug:GYM', file=sys.stderr)
    eval_env = gym.make(args.env, args=args)
    eval_env.seed(args.seed)
    print('debug:DEEP COPY', file=sys.stderr)
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

    # HP search specific keys
    for key in wandb.config.keys():
        if key.startswith('HP_'):
            key_without_hp = key[3:]
            rl_kwargs[key_without_hp] = wandb.config[key]
    policy_name = 'MlpPolicy'
    if args.cam_resolution > 0 and args.uint8_pixels:
        policy_name = 'CnnPolicy'

    rl_agent = eval(args.rl_algo)(policy_name, vec_env, **rl_kwargs)
    cb = CustomCallback(eval_env, args.logdir, n_envs, args,
                        num_steps_between_save=num_steps_between_save,
                        viz=args.viz, debug=args.debug)
    rl_agent.learn(total_timesteps=args.total_env_steps, callback=cb)
    vec_env.close()
    wandb.finish()
    # reset logdir
    args.logdir = args.base_logdir


def args_setup():
    args, main_parser = get_args_parser()
    parser = argparse.ArgumentParser(
        description="AgentData", parents=[main_parser], add_help=False)

    parser.add_argument('--agent_id', type=str, required=True,
                        help='Agent ID For which wandb is used')

    args, unknown = parser.parse_known_args()
    args_postprocess(args)
    return args

if __name__ == "__main__":
    # args = args_setup()
    # wandb.agent(args.agent_id, main)
    # # main()
    args = get_args()
    main()
