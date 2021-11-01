"""
A demo for training Sequential Variational Autoencoders.

python -m dedo.run_svae --logdir=~/local/dedo --num_envs 12 --unsup_algo VAE

tensorboard --logdir=/tmp/dedo --bind_all --port 6006

--unsup_algo choices are: VAE, SVAE, PRED

Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika

"""
from copy import deepcopy

import gym
import numpy as np
from stable_baselines3.common.env_util import (
    make_vec_env, DummyVecEnv, SubprocVecEnv)
from tensorboardX import SummaryWriter
import torch

from dedo.utils.args import get_args
from dedo.utils.train_utils import init_train, object_to_str
from dedo.vaes.nets import ConvStack
from dedo.vaes.svae import SVAE  # used dynamically
from dedo.vaes.svae_utils import do_logging, fill_seq_bufs_from_rollouts
from dedo.vaes.svae_viz import viz_samples


def get_batch(env, rollout_len):
    x_1toT = []
    act_1toT = []
    mask_1toT = []
    for _ in range(rollout_len):
        act01 = np.random.rand(env.num_envs, env.action_space.shape[-1])
        act = act01*2 - 1.0  # [0,1] -> [-1,1]
        obs, rwd, done, next = env.step(act)
        masks = np.array([[0.0] if done_ else [1.0] for done_ in done])
        x_1toT.append(obs)
        act_1toT.append(act)
        mask_1toT.append(masks)
        if np.random.rand() > 0.9:  # reset envs randomly for data variety
            env_id = np.random.randint(env.num_envs)
            env.env_method('reset', indices=[env_id])
    x_1toT = torch.from_numpy(np.stack(x_1toT)).float()
    act_1toT = torch.from_numpy(np.stack(act_1toT)).float()
    mask_1toT = torch.from_numpy(np.stack(mask_1toT)).float()
    x_1toT = x_1toT.transpose(0, 1).transpose(2, -1).transpose(-2, -1)
    act_1toT = act_1toT.transpose(0, 1)   # put bsz 0th, time 1st
    mask_1toT = mask_1toT.transpose(0, 1)  # put bsz 0th, time 1st
    return x_1toT, act_1toT, mask_1toT


def main(args):
    assert(args.unsup_algo is not None), 'Please provide --unsup_algo'
    if args.cam_resolution not in ConvStack.IMAGE_SIZES:
        print(f'Setting cam_resolution to 512 (was {args.cam_resolution:d})')
        args.cam_resolution = 512  # set default image resolution if needed
    args.logdir, args.device = init_train(args.unsup_algo, args)
    tb_writer = SummaryWriter(args.logdir)
    tb_writer.add_text('args', object_to_str(args))
    print('svae_demo with args:\n', args)
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
    unsup_algo_params = 'PARAMS_'+args.unsup_algo
    unsup_algo_class = 'SVAE'
    svae = eval(unsup_algo_class)(
        im_sz=args.cam_resolution, act_sz=vec_env.action_space.shape[-1],
        params_class=unsup_algo_params, device=args.device)
    optim = torch.optim.Adam(svae.parameters(), lr=args.lr)
    seq_len = svae.pr.past+svae.pr.pred
    rlt_len = 50
    num_inner_epochs = 100 if args.unsup_algo == 'VAE' else 50
    mini_batch_size = 96 if args.unsup_algo == 'VAE' else 24
    if args.unsup_algo == 'PRED':
        mini_batch_size = 16
    vec_env.reset()
    steps_done = 0
    epoch = 0
    print('Getting test env data... ')
    test_x_1toT, test_act_1toT, test_mask_1toT = get_batch(vec_env, 300)
    print('got', test_x_1toT.shape)
    while steps_done < args.total_env_steps:
        print(f'Epoch {epoch:d}: getting train env data... ')
        all_x_1toT, all_act_1toT, all_mask_1toT = get_batch(vec_env, rlt_len)
        all_x_1toT.to(args.device)
        all_act_1toT.to(args.device)
        all_mask_1toT.to(args.device)
        print('got', all_x_1toT.shape)
        steps_done += rlt_len*args.num_envs
        for inner_epoch in range(num_inner_epochs):
            do_log_viz = inner_epoch+1 == num_inner_epochs
            x_1toT, act_1toT = fill_seq_bufs_from_rollouts(
                all_x_1toT, all_act_1toT, all_mask_1toT,
                mini_batch_size, seq_len, args.device)
            optim.zero_grad()
            loss, debug_dict = svae.loss(x_1toT, act_1toT, debug=do_log_viz)
            loss.backward()
            optim.step()
            if do_log_viz:
                do_logging(epoch, debug_dict, {}, tb_writer, 'train')
                viz_samples(svae, x_1toT, act_1toT, epoch, tb_writer, 'train')
                tmp_x_1toT, tmp_act_1toT = fill_seq_bufs_from_rollouts(
                    test_x_1toT, test_act_1toT, test_mask_1toT,
                    mini_batch_size, seq_len, args.device)
                steps_done += seq_len*args.num_envs
                loss, debug_dict = svae.loss(
                    tmp_x_1toT, tmp_act_1toT, debug=do_log_viz)
                do_logging(epoch, debug_dict, {}, tb_writer, 'test')
                viz_samples(svae, tmp_x_1toT, tmp_act_1toT, epoch, tb_writer,
                            'test')
        epoch += 1
    #
    # Clean up.
    vec_env.close()


if __name__ == "__main__":
    main(get_args())
