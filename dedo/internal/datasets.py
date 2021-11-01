import sys
import time
import torch
from torch.utils.data import IterableDataset
import numpy as np

def worker_init_fn(worker_id):
    '''Helper function to launch each worker's env with a different seed'''

    gym = __import__('gym')
    worker_info = torch.utils.data.get_worker_info()
    ds = worker_info.dataset

    # np.random.seed(worker_info.seed)
    np.random.seed(np.random.randint(1212))

    args = ds.args
    args.seed = np.random.randint(1212)
    ds.env = gym.make(args.env, args=args)
    ds.env.reset()

    print('worker rand', worker_id, np.random.randint(1212))


class DeformEnvDataset(IterableDataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        print('random_number', np.random.randint(10000))
        worker_info = torch.utils.data.get_worker_info()

    def __iter__(self):

        # TODO Add Noise to Trajectory
        # TODO Add Noise to Camera Angle

        # TODO Check after each episode, seed changes
        worker_info = torch.utils.data.get_worker_info()
        print('worker_info iter',worker_info)
        args = self.args
        args.seed = worker_info.seed
        return self
    def __next__(self):
        print('random_number_iter', np.random.randint(10000))
        worker_info = torch.utils.data.get_worker_info()
        print('worker_info next',worker_info)
        next_obs, rwd, done, info = self.env.step(np.array([0,0,0,0,0,0], dtype=np.float16))
        if done:
            raise StopIteration
        return next_obs

