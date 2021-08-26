#
# RL with ray[rllib]
#
# @contactrika
#
# pip install ray[rllib]
# python -m dedo.rllib_utils --env_name=HangGarment-v1
#
import argparse
import glob
import io
import os
import pickle

import torch

import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.rollout import rollout
from ray.rllib.agents import a3c, impala, sac, ppo
from ray.rllib.agents.ddpg import apex, td3
from ray.tune.suggest.variant_generator import grid_search

from dedo.envs.deform_env import DeformEnv
from dedo.utils.args import get_args


class RllibDeformBulletEnv(DeformEnv):
    # ray.readthedocs.io/en/latest/rllib-env.html
    def __init__(self, env_config):
        super(RllibDeformBulletEnv, self).__init__(**env_config)


def deform_env_creator(env_config):
    return RllibDeformBulletEnv(env_config)


def get_agent_trainer(rl_algo):
    if rl_algo == 'ApexDDPG':
        rl_trainer_class = apex.ApexDDPGTrainer
    elif rl_algo == 'TD3':
        rl_trainer_class = td3.TD3Trainer
    else:
        rl_trainer_class = eval(rl_algo.lower()+'.'+rl_algo+'Trainer')
    return rl_trainer_class


def guess_checkpt(rl_algo, env_name):
    # Try to guess checkpoint path
    pfx = os.path.expanduser('~/ray_results/')
    if 'Viz' in env_name:
        parts = env_name.split('Viz')
        env_name = parts[0]+parts[1]
    if rl_algo=='ApexDDPG': rl_algo = 'APEX_DDPG'
    data_dir = pfx+rl_algo+'/'+rl_algo+'_'+env_name+'_*/'
    pth = data_dir+'checkpoint_*'
    options = glob.glob(pth)
    print('pth', pth, 'options', options)
    iter = max([int(res.split('_')[-1]) for res in options])
    load_pth = data_dir+'checkpoint_'+str(iter)+'/checkpoint-'+str(iter)
    print('Guessed path', load_pth)
    load_option = glob.glob(load_pth)
    assert(len(load_option) == 1)
    return load_option[0]


def play(args, rl_config):
    rl_config['num_workers'] = 0
    if args.load_checkpt is None:
        args.load_checkpt = guess_checkpt(args.rl_algo, args.env_name)
    print('Loading checkpoint', args.load_checkpt)
    play_algo = 'APEX_DDPG' if args.rl_algo=='ApexDDPG' else args.rl_algo
    cls = get_agent_class(play_algo)
    agent = cls(env=args.env_name, config=rl_config)
    agent.restore(os.path.expanduser(args.load_checkpt))
    rollout(agent, args.env_name, num_episodes=args.play,
            num_steps=1000, no_render=True)
    return  # just playing


def make_rl_config(args, num_gpus, use_pointnet=False):
    # github.com/ray-project/ray/blob/master/rllib/tuned_examples/walker2d-ppo.yaml
    if args.rl_algo=='ApexDDPG':
        rl_config = apex.APEX_DDPG_DEFAULT_CONFIG.copy()
    elif args.rl_algo=='TD3':
        # github.com/ray-project/ray/blob/master/rllib/agents/ddpg/td3.py
        rl_config = td3.TD3_DEFAULT_CONFIG.copy()
    else:
        rl_config = eval(args.rl_algo.lower()).DEFAULT_CONFIG.copy()
    env_parts = args.env_name.split('-v')
    state_v = 'Combo'
    if 'Fast' in env_parts[0]: state_v = 'Fast'
    if 'Topo' in env_parts[0]: state_v = 'Topo'
    if 'Pts' in env_parts[0]: state_v = 'Pts'
    scene_parts = env_parts[0].split(state_v)
    rl_config['env_config'] = {
        'max_episode_len':200,
        'version':int(env_parts[-1]),
        'scene_version_name':scene_parts[0],
        'state_version_name':state_v,
        'cam_and_cloth_args':get_args(),
        'viz':'Viz' in args.env_name, 'debug':'Debug' in args.env_name}
    bsz = args.rollout_len*max(1, args.ncpus)
    rl_config['train_batch_size'] = bsz
    rl_config['rollout_fragment_length'] = args.rollout_len  # aka sample_batch_size
    rl_config['soft_horizon'] = True  # don't reset episode after rollout
    rl_config['num_gpus'] = num_gpus
    rl_config['num_workers'] = args.ncpus
    rl_config['num_envs_per_worker'] = 1 if args.play else args.num_envs_per_worker
    rl_config['env'] = args.env_name
    rl_config['lr'] = args.rl_lr
    rl_config['clip_rewards'] = None
    rl_config['gamma'] = 0.995
    # rl_config['horizon'] = rollout_len  # seems to break things
    # Customize NN architecture and hidden layers.
    rl_config['model']['fcnet_activation'] = 'tanh'
    hsz = 128 if state_v=='Fast' else 512
    if state_v == 'Fast':
        rl_config['model']['fcnet_hiddens'] = [hsz, hsz, hsz]
    else:  # state_v in ['Pts', 'Combo']:
        if use_pointnet:
            rl_config['model']['custom_model'] = 'pointnet'
        else:
            rl_config['model']['custom_model_config'] = {
                'fcnet_activation': 'ReLU',
                'fcnet_hiddens':[1024,512,256],
                'no_final_linear':False,
                'vf_share_layers': True,
                'free_log_std':False,
            }
    if not args.use_tf: rl_config['framework'] = 'torch'
    if args.rl_algo == 'A3C' and args.use_tf:
        rl_config['sample_async'] = False
    if args.rl_algo == 'PPO':
        rl_config['kl_coeff'] = 1.0
        rl_config['num_sgd_iter'] = 100
        rl_config['sgd_minibatch_size'] = bsz//10
        # rl_config['vf_share_layers'] = True
        rl_config['entropy_coeff'] = 0.01   # low exploration noise
    if args.rl_algo == 'Impala':
        rl_config['num_sgd_iter'] = 50
        rl_config['replay_proportion'] = 0.5  # 0.5:1 proportion
        rl_config['replay_buffer_num_slots'] = 10000
    if args.rl_algo == 'ApexDDPG':
        rl_config['learning_starts'] = args.rollout_len
        rl_config['target_network_update_freq'] = 2*args.rollout_len
        rl_config['timesteps_per_iteration'] = 2*args.rollout_len
    return rl_config

