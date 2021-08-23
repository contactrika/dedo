import sys
import time
import torch
from torch.utils.data import IterableDataset
import numpy as np
import gym

class DeformEnvDataset(IterableDataset):
    def __init__(self, args):
        super().__init__()

        self.env = gym.make(args.env, args=args)
        print('random_number', np.random.randint(10000))
        obs = self.env.reset()

    
    def __iter__(self):
        # TODO Follow Rudimentary policy
        # TODO Add Noise to Trajectory
        # TODO Add Noise to Camera Angle

        return self
    def __next__(self):
        print('random_number_iter', np.random.randint(10000))
        next_obs, rwd, done, info = self.env.step(np.array([0,0,0,0,0,0], dtype=np.float16))
        if done:
            raise StopIteration
        return next_obs

