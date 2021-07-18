from setuptools import setup

setup(name='dedo',
      version='0.1',
      description='Dynamic Environments with Deformable Objects.',
      packages=['dedo'],
      install_requires=[
            'numpy', 'matplotlib', 'gym', 'pybullet==3.1.7',
      ])
