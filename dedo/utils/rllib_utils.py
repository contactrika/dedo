"""
RL with ray[rllib].

To in stall RLlib use:
pip install ray[rllib]

Example command to run RL training:
python -m dedo.rllib_utils --env_name=HangGarment-v1

Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika

"""

import glob
import os

from ray.rllib.agents.registry import get_agent_class
from ray.rllib.rollout import rollout
from ray.rllib.agents import a3c, impala, sac, ppo  # used dynamically
from ray.rllib.agents.ddpg import apex, td3

from dedo.envs.deform_env import DeformEnv


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


def guess_checkpt(load_checkpt_dir):
    # Try to guess checkpoint path
    pfx = os.path.join(os.path.expanduser(load_checkpt_dir),
                       'rllib', 'agent')
    pattern = os.path.join(pfx, 'checkpoint_*')
    options = glob.glob(pattern)
    # print('pattern', pattern, 'options', options)
    iter = max([int(res.split('_')[-1]) for res in options])
    load_pth = os.path.join(
        pfx, f'checkpoint_{iter:06d}', f'checkpoint-{iter:d}')
    print('Guessed path', load_pth)
    load_option = glob.glob(load_pth)
    assert(len(load_option) == 1)
    return load_option[0]


def play(args, rl_config, num_episodes=10):
    rl_config['num_workers'] = 0
    checkpt = guess_checkpt(args.load_checkpt)
    print('Loading play checkpoint', checkpt)
    assert(args.rl_algo in checkpt), \
        f'--rl_algo {args.rl_algo:s} must match checkpt algo'
    play_algo = 'APEX_DDPG' if args.rl_algo == 'ApexDDPG' else args.rl_algo
    cls = get_agent_class(play_algo)
    agent = cls(env=args.env, config=rl_config)
    agent.restore(checkpt)
    rollout(agent, args.env, num_episodes=num_episodes,
            num_steps=1000, no_render=True)
    return  # just playing


def make_rl_config(args, num_gpus):
    # github.com/ray-project/ray/blob/master/rllib/tuned_examples/walker2d-ppo.yaml
    if args.rl_algo == 'ApexDDPG':
        rl_config = apex.APEX_DDPG_DEFAULT_CONFIG.copy()
    elif args.rl_algo == 'TD3':
        # github.com/ray-project/ray/blob/master/rllib/agents/ddpg/td3.py
        rl_config = td3.TD3_DEFAULT_CONFIG.copy()
    else:
        rl_config = eval(args.rl_algo.lower()).DEFAULT_CONFIG.copy()
    rl_config['env_config'] = {'args': args}
    bsz = args.rollout_len * 2# *max(1, args.num_envs)
    rl_config['train_batch_size'] = bsz
    rl_config['rollout_fragment_length'] = args.rollout_len  # sample_batch_size
    rl_config['soft_horizon'] = True  # don't reset episode after rollout
    rl_config['num_gpus'] = num_gpus
    rl_config['num_workers'] = args.num_envs
    rl_config['num_envs_per_worker'] = 1
    rl_config['env'] = args.env
    rl_config['lr'] = args.lr
    rl_config['clip_rewards'] = None
    rl_config['gamma'] = 0.995
    # rl_config['horizon'] = rollout_len  # seems to break things
    # Customize NN architecture and hidden layers.
    # if args.cam_resolution > 0 is not None:
    #    im_sz = args.cam_resolution
    #    nflt = 32
    #    conv_filters = [[nflt, [4, 4], 2], [nflt*2, [4, 4], 2]]
    #    if im_sz >= 64:
    #        conv_filters.append([nflt*2, [4, 4], 2])
    #    if im_sz == 128:
    #        conv_filters.append([nflt*4, [4, 4], 2])
    #    conv_filters.append([nflt*4, [8, 8], 1])  # im_sz=32,64,128
    #    rl_config['model']['dim'] = args.cam_resolution
    #    rl_config['model']['conv_filters'] = conv_filters
    if args.cam_resolution > 0:
        rl_config['model']['fcnet_hiddens'] = [1024, 512, 128]
    else:
        rl_config['model']['fcnet_hiddens'] = [64, 64]
    if args.rllib_use_torch:
        rl_config['framework'] = 'torch'
    if args.rl_algo == 'A3C' and not args.rllib_use_torch:
        rl_config['sample_async'] = False
    if args.rl_algo == 'PPO':
        rl_config['kl_coeff'] = 1.0
        rl_config['num_sgd_iter'] = 100
        rl_config['sgd_minibatch_size'] = bsz//10
        # rl_config['vf_share_layers'] = True
        rl_config['entropy_coeff'] = 0.01   # low exploration noise
    if args.rl_algo == 'SAC':
        rl_config['buffer_size'] = args.replay_size
    if args.rl_algo == 'TD3':
        rl_config['buffer_size'] = args.replay_size
    if args.rl_algo == 'Impala':
        rl_config['num_sgd_iter'] = 50
        rl_config['replay_proportion'] = 0.5  # 0.5:1 proportion
        rl_config['replay_buffer_num_slots'] = args.replay_size
    if args.rl_algo == 'ApexDDPG':
        rl_config['learning_starts'] = args.rollout_len
        rl_config['target_network_update_freq'] = 2*args.rollout_len
        rl_config['timesteps_per_iteration'] = 2*args.rollout_len
    return rl_config
