from setuptools import setup

setup(name='dedo',
      version='0.1',
      description='Dynamic Environments with Deformable Objects.',
      packages=['dedo'],
      install_requires=[
            'numpy', 'gym==0.20.0', 'pybullet==3.1.7',
            'torch', 'stable_baselines3=1.2.0',
            'matplotlib', 'tensorboard', 'tensorboardX', 'moviepy', 'wandb',
            'pyaml', 'opencv-python',
      ])
