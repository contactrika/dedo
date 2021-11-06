"""
An example of RL training using RLlib.

pip install tensorflow-gpu
pip install ray[rllib]
python -m dedo.run_rllib --env=HangGarment-v1 --rl_algo PPO --logdir=/tmp/dedo

tensorboard --logdir=/tmp/dedo --bind_all --port 6006

Play the saved policy (e.g. logged to PPO_210825_204955_HangGarment-v1):
python -m dedo.run_rl_sb3 --env=HangGarment-v1 \
    --play=/tmp/dedo/PPO_210825_204955_HangGarment-v1

To use Torch RLlib implementations:
python -m dedo.run_rllib --env=HangGarment-v1 --rl_algo PPO \
  --logdir=/tmp/dedo --rllib_use_torch


Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika

"""
import os

import ray

from dedo.utils.args import get_args
from dedo.utils.rllib_utils import (
    deform_env_creator, get_agent_trainer, make_rl_config, play)
from dedo.utils.train_utils import init_train


def run_with_args(args):
    assert(args.rl_algo is not None), 'Please provide --rl_algo'
    os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
    args.logdir, args.device = init_train(args.rl_algo, args)
    args.flat_obs = True
    ray.tune.registry.register_env(args.env, deform_env_creator)
    num_cpus = args.num_envs+1  # +1 for the controller
    num_gpus = 0
    if args.device != 'cpu':
        gpus = args.device[len('cuda:'):]  # expecting format: 'cuda:0,2,3'
        print('Using GPUs', gpus)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        num_gpus = len(args.device.split(','))
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus, local_mode=args.debug)
    rl_config = make_rl_config(args, num_gpus)
    # Play if requested.
    if args.play:
        play(args, rl_config)
        return  # no training, just play
    # Run training.
    ray.tune.run(get_agent_trainer(args.rl_algo), config=rl_config,
                 checkpoint_freq=args.log_save_interval,
                 local_dir=args.logdir, name='rllib',
                 trial_dirname_creator=lambda x: 'agent',
                 restore=args.load_checkpt, reuse_actors=True,
                 stop={'timesteps_total': args.total_env_steps},
                 )


if __name__ == "__main__":
    run_with_args(get_args())
