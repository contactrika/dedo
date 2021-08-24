"""
A demo for training Sequential Variational Autoencoders.

python -m dedo.svae_demo --env=HangGarment-v1 --logdir=/tmp/dedo

tensorboard --logdir=/tmp/dedo --bind_all --port 6006


@contactrika

"""
from copy import deepcopy
from collections import deque
from datetime import datetime
from glob import glob
import os
import platform
if platform.system() == 'Linux':
    os.environ['IMAGEIO_FFMPEG_EXE'] = '/usr/bin/ffmpeg'

import gym
import numpy as np
from stable_baselines3.common.env_util import (
    make_vec_env, DummyVecEnv, SubprocVecEnv)
from tensorboardX import SummaryWriter
import torch
import wandb

from dedo.utils.args import get_args
from dedo.utils.rl_utils import object_to_str
from dedo.vaes.svae_advanced import SVAE
from dedo.vaes.svae_utils import do_logging
from dedo.vaes.svae_viz import viz_samples


def get_batch(env, rollout_len, device):
    x_1toT = []
    act_1toT = []
    for tmp_i in range(rollout_len):
        act01 = np.random.rand(env.num_envs, env.action_space.shape[-1])
        act = act01*2 - 1.0  # [0,1] -> [-1,1]
        obs, rwd, done, next = env.step(act)
        x_1toT.append(obs)
        act_1toT.append(act)
    x_1toT = torch.from_numpy(np.stack(x_1toT)).float().to(device)
    act_1toT = torch.from_numpy(np.stack(act_1toT)).float().to(device)
    x_1toT = x_1toT.transpose(0, 1).transpose(2, -1).transpose(-2, -1)
    act_1toT = act_1toT.transpose(0, 1)
    # TODO: add done masks
    return x_1toT, act_1toT


def main(args):
    np.set_printoptions(precision=4, linewidth=150, suppress=True)
    logdir = None
    if args.logdir is not None:
        tstamp = datetime.strftime(datetime.today(), '%y%m%d_%H%M%S')
        subdir = '_'.join([args.unsup_algo_params[7:], tstamp, args.env])
        logdir = os.path.join(os.path.expanduser(args.logdir), subdir)
        if args.use_wandb:
            wandb.init(config=vars(args), project='dedo', name=logdir)
            wandb.init(sync_tensorboard=False)
            wandb.tensorboard.patch(tensorboardX=True, pytorch=True)
    tb_writer = SummaryWriter(logdir)
    tb_writer.add_text('args', object_to_str(args))
    print('svae_demo with args:\n', args)
    eval_env = gym.make(args.env, args=args)
    eval_env.seed(args.seed)
    train_args = deepcopy(args)
    train_args.debug = False  # no debug during training
    train_args.viz = False  # no viz during training
    vec_env = make_vec_env(
        args.env, n_envs=args.num_envs,
        vec_env_cls=SubprocVecEnv if args.num_envs > 1 else DummyVecEnv,
        env_kwargs={'args': train_args})
    vec_env.seed(args.seed)
    print('Created', args.task, 'with observation_space',
          vec_env.observation_space.shape, 'action_space',
          vec_env.action_space.shape)
    #
    # Train unsupervised learner.
    #
    svae = eval(args.unsup_algo)(
        im_sz=args.cam_resolution, act_sz=vec_env.action_space.shape[-1],
        latent_sz=8, params_class=args.unsup_algo_params, device=args.device)
    optim = torch.optim.Adam(svae.parameters(), lr=args.lr)
    vec_env.reset()
    steps_done = 0
    epoch = 0
    while steps_done < args.total_env_steps:
        do_log_viz = epoch % 100 == 0
        x_1toT, act_1toT = get_batch(
            vec_env, svae.pr.past+svae.pr.pred, args.device)
        optim.zero_grad()
        loss, debug_dict = svae.loss(x_1toT, act_1toT, debug=do_log_viz)
        loss.backward()
        optim.step()
        if do_log_viz:
            do_logging(epoch, debug_dict, {}, tb_writer, 'train')
            viz_samples(svae, x_1toT, act_1toT, epoch, tb_writer, 'train')
            test_x_1toT, test_act_1toT = get_batch(
                vec_env, svae.pr.past+svae.pr.pred, args.device)
            loss, debug_dict = svae.loss(
                test_x_1toT, test_act_1toT, debug=do_log_viz)
            do_logging(epoch, debug_dict, {}, tb_writer, 'test')
            viz_samples(svae, test_x_1toT, test_act_1toT, epoch, tb_writer,
                        'test')
        epoch += 1
    #
    # Clean up.
    vec_env.close()


if __name__ == "__main__":
    main(get_args())
