"""
A simple demo for envs (with random actions).

python -m dedo.deform_env_demo --env_name=ClothScene-v0 --viz --debug

@contactrika
"""

import logging
import sys
import time

import numpy as np
np.set_printoptions(precision=4, linewidth=150, threshold=np.inf, suppress=True)

import gym
import pybullet

import dedo  # to register envs
from dedo.utils.args import get_args


def play(env, num_episodes, debug, viz):
    for epsd in range(num_episodes):
        print('------------ Play episode ', epsd, '------------------')
        obs = env.reset()
        # Need to step to get low-dim state from info.
        step = 0
        input('Reset done; press enter to start episode')
        while True:
            assert(not isinstance(env.action_space, gym.spaces.Discrete))
            act = np.random.rand(*env.action_space.shape)  # in [0,1]
            rng = env.action_space.high - env.action_space.low
            act = act*rng + env.action_space.low
            next_obs, rwd, done, info = env.step(act)
            print("next_obs.shape", next_obs.shape)
            print("act.shape", act.shape)

            if done:
                break
            obs = next_obs
            step += 1
        input('Episode ended; press enter to go on')


def main(args):
    print('env_name', args.env_name)
    assert('-v' in args.env_name)  # specify env version
    env = gym.make(args.env_name)
    env.seed(env.args.seed)
    print('Created env', args.env_name, 'with observation_space',
          env.observation_space.shape, 'action_space', env.action_space.shape)
    play(env, env.args.num_runs, env.args.debug, env.args.viz)
    env.close()


if __name__ == "__main__":
    main(get_args()[0])
