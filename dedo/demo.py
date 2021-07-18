"""
A simple demo for envs (with random actions).

python -m dedo.demo --task=HangBag --viz --debug
python -m dedo.demo --task=HangCloth --viz --debug

@contactrika

"""
import numpy as np

import gym

from dedo.utils.args import get_args


def policy_simple(obs):
    assert(2*3)
    obs = obs.reshape(-1, 3)
    act = np.random.rand(2, 3)  # in [0,1]
    if obs[0, 2] > 0.30:
        act[:, 1] = -0.13  # decrease y
        act[:, 2] = -0.1  # decrease z
    else:
        act[:] = 0.0  # rest
    return act.reshape(-1)


def play(env, num_episodes):
    for epsd in range(num_episodes):
        print('------------ Play episode ', epsd, '------------------')
        obs = env.reset()
        # Need to step to get low-dim state from info.
        step = 0
        input('Reset done; press enter to start episode')
        while True:
            assert(not isinstance(env.action_space, gym.spaces.Discrete))
            # act = np.random.rand(*env.action_space.shape)  # in [0,1]
            act = policy_simple(obs)
            rng = env.action_space.high - env.action_space.low
            act = act*rng + env.action_space.low
            next_obs, rwd, done, info = env.step(act)
            if done:
                break
            obs = next_obs
            step += 1
        input('Episode ended; press enter to go on')


def main(args):
    np.set_printoptions(precision=4, linewidth=150,
                        threshold=np.inf, suppress=True)
    version = 0  # TODO: make versions
    kwargs = {'version':version, 'args':args}
    env = gym.make(args.task+'-v'+str(version), **kwargs)
    env.seed(env.args.seed)
    print('Created', args.task, 'with observation_space',
          env.observation_space.shape, 'action_space', env.action_space.shape)
    play(env, env.args.num_runs)
    env.close()


if __name__ == "__main__":
    main(get_args()[0])
