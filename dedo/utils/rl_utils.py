"""
Utilities for RL training and eval.

@contactrika

"""
import pickle
import os

import gym
import torch

from stable_baselines3 import A2C, DDPG, HER, PPO, SAC, TD3  # used dynamically
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import TensorBoardOutputFormat, Video

from dedo.utils.args import get_args


def play(env, num_episodes, rl_agent, debug=False):
    for epsd in range(num_episodes):
        if debug:
            print('------------ Play episode ', epsd, '------------------')
        obs = env.reset()
        step = 0
        while True:
            # rl_agent.predict() to get acts, not forcing deterministic.
            act, _states = rl_agent.predict(obs)
            next_obs, rwd, done, info = env.step(act)
            if done:
                break
            obs = next_obs
            step += 1
        # input('Episode ended; press enter to go on')


def object_to_str(obj):
    # Print all fields of the given object as text in tensorboard.
    text_str = ''
    for member in vars(obj):
        # Tensorboard uses markdown-like formatting, hence '  \n'.
        text_str += '  \n{:s}={:s}'.format(
            str(member), str(getattr(obj, member)))
    return text_str


class CustomCallback(BaseCallback):
    """
    A custom callback that runs eval and adds videos to Tensorboard.
    """
    def __init__(self, eval_env, num_play_episodes, logdir, num_train_envs,
                 args, num_steps_between_play=20000, viz=False, debug=False):
        super(CustomCallback, self).__init__(debug)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self._eval_env = eval_env
        self._num_play_episodes = num_play_episodes
        self._logdir = logdir
        self._num_train_envs = num_train_envs
        self._my_args = args
        self._num_steps_between_play = num_steps_between_play
        self._viz = viz
        self._debug = debug
        self._steps_since_play = num_steps_between_play  # play right away

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # Log args to tensorboard.
        self.logger.record('args', object_to_str(self._my_args))

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        self._steps_since_play += self._num_train_envs
        if self._steps_since_play >= self._num_steps_between_play:
            screens = []

            def grab_screens(_locals, _globals):
                screen = self._eval_env.render(
                    mode='rgb_array', width=300, height=300)
                # PyTorch uses CxHxW vs HxWxC gym (and TF) images
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model, self._eval_env, callback=grab_screens,
                n_eval_episodes=self._num_play_episodes, deterministic=False)
            self.logger.record(
                'trajectory/video', Video(torch.ByteTensor([screens]), fps=50),
                exclude=('stdout', 'log', 'json', 'csv'))

            self._steps_since_play = 0
            if self._logdir is not None:
                self.model.save(os.path.join(self._logdir, 'agent'))
                pickle.dump(self._my_args,
                            open(os.path.join(self._logdir, 'args.pkl'), 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


def main(args):
    # Example usage (if training logged to PPO_210822_104834_HangBag-v1):
    # python -m dedo.utils.rl_utils --logdir PPO_210822_104834_HangBag-v1
    checkpt = os.path.join(args.logdir, 'agent.zip')
    print('Loading checkpoint from', checkpt)
    args = pickle.load(open(os.path.join(args.logdir, 'args.pkl'), 'rb'))
    args.debug = True
    args.viz = True
    eval_env = gym.make(args.env, args=args)
    eval_env.seed(args.seed)
    rl_agent = eval(args.rl_algo).load(checkpt)
    play(eval_env, num_episodes=10, rl_agent=rl_agent, debug=False)


if __name__ == "__main__":
    main(get_args())
