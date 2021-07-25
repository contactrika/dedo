from setuptools import setup

setup(name='dedo',
      version='0.1',
      description='Dynamic Environments with Deformable Objects.',
      packages=['dedo'],
      install_requires=[
            'numpy', 'gym', 'pybullet==3.1.7',
            'torch', 'stable_baselines3',
            'matplotlib', 'tensorboard',
      ])
