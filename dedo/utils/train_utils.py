"""
Common utilities for training.

@contactrika

"""
from datetime import datetime
import numpy as np

import os
import platform
import torch
import wandb


def object_to_str(obj):
    # Print all fields of the given object as text in tensorboard.
    text_str = ''
    for member in vars(obj):
        # Tensorboard uses markdown-like formatting, hence '  \n'.
        text_str += '  \n{:s}={:s}'.format(
            str(member), str(getattr(obj, member)))
    return text_str


def init_train(algo, args):
    np.set_printoptions(precision=4, linewidth=150, suppress=True)
    if platform.system() == 'Linux':
        os.environ['IMAGEIO_FFMPEG_EXE'] = '/usr/bin/ffmpeg'
    logdir = None
    if args.logdir is not None:
        tstamp = datetime.strftime(datetime.today(), '%y%m%d_%H%M%S')
        subdir = '_'.join([algo, tstamp, args.env])
        logdir = os.path.join(os.path.expanduser(args.logdir), subdir)
        if args.use_wandb:
            wandb.init(config=vars(args), project='dedo', name=logdir)
            wandb.init(sync_tensorboard=False)
            wandb.tensorboard.patch(tensorboardX=True, pytorch=True)
    device = args.device
    if not torch.cuda.is_available():
        device = 'cpu'
    return logdir, device
