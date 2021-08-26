"""
An example of RL training using RLlib.

python -m dedo.run_rllib --env=HangGarment-v1 --rl_algo PPO --logdir=/tmp/dedo

tensorboard --logdir=/tmp/dedo --bind_all --port 6006

@contactrika

"""
import os
import ray

from dedo.utils.args import get_args
from dedo.utils.rllib_utils import (
    deform_env_creator, get_agent_trainer, make_rl_config, play)


def run_with_args(args):
    ray.tune.registry.register_env(args.env_name, deform_env_creator)
    num_gpus = 0
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        num_gpus = len(args.gpus.split(','))
    ray.init(num_cpus=args.ncpus+1, num_gpus=num_gpus, local_mode=args.debug)
    rl_config = make_rl_config(args, num_gpus)
    # Play if requested.
    if args.play is not None: play(args, rl_config); return
    # Run training.
    rl_trainer_class=get_agent_trainer(args.rl_algo)
    ray.tune.run(rl_trainer_class, config=rl_config,
                 checkpoint_freq=args.checkpt_save_interval,
                 restore=args.load_checkpt, reuse_actors=True)


if __name__ == "__main__":
    run_with_args(get_args())
