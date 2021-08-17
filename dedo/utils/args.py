#
# Command line arguments.
#
# @contactrika
#
import argparse
import logging
import sys

from .task_info import TASK_INFO


def get_args():
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)])
    parser = argparse.ArgumentParser(description='args', add_help=True)
    # Main/demo args.
    parser.add_argument('--env', type=str,
                        default='HangBag-v0', help='Env name')
    parser.add_argument('--max_episode_len', type=int,
                        default=400, help='Maximum time per episode (in seconds)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Path for logs')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Name of the device for training.')
    parser.add_argument('--rl_algo', type=str, default=None,
                        choices=['A2C', 'DDPG', 'HER', 'PPO', 'SAC', 'TD3'],
                        help='Name of RL algo from Stable Baselines to train')
    parser.add_argument('--num_envs', type=int, default=8,
                        help='Number of parallel envs (for A2C, PPO, SAC)')
    parser.add_argument('--num_play_runs', type=int, default=1,
                        help='Number of runs/episodes to play')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to enable logging to wandb.ai')
    parser.add_argument('--viz', action='store_true', help='Whether to visualize')
    parser.add_argument('--debug', action='store_true',
                        help='Whether to print debug info')

    # Simulation args. Note: turn up frequency when deform stiffness is high.
    parser.add_argument('--sim_gravity', type=float, default=-9.8,
                        help='Gravity constant for PyBullet simulation.')
    parser.add_argument('--sim_freq', type=int, default=500,  # 250-1K
                        help='PyBullet simulation frequency.')
    parser.add_argument('--sim_steps_per_action', type=int, default=4,
                        help='Number of sim steps per action.')
    # Anchor/grasping args.
    parser.add_argument('--anchor_init_pos', type=float, nargs=3,
                        default=[-0.04, 0.40, 0.70],
                        help='Initial position for an anchor')
    parser.add_argument('--other_anchor_init_pos', type=float, nargs=3,
                        default=[0.04, 0.40, 0.70],
                        help='Initial position for another anchors')
    # SoftBody args.
    parser.add_argument('--override_deform_obj', type=str, default=None,
                        help='Load custom deformable (note that you have to'
                             'fill in DEFORM_INFO entry for new items)')
    parser.add_argument('--deform_init_pos', type=float, nargs=3,
                        default=[0,0,0.42],
                        help='Initial pos for the center of the deform object')
    parser.add_argument('--deform_init_ori', type=float, nargs=3,
                        default=[0,0,0],
                        help='Initial orientation for deform (in Euler angles)')
    parser.add_argument('--deform_scale', type=float, default=1.0,
                        help='Scaling for the deform object')
    parser.add_argument('--deform_bending_stiffness', type=float, default=1.0,
                        help='deform spring elastic stiffness')  # 1.0-300.0
    parser.add_argument('--deform_damping_stiffness', type=float, default=0.1,
                        help='deform spring damping stiffness')
    parser.add_argument('--deform_elastic_stiffness', type=float, default=1.0,
                        help='deform spring elastic stiffness')  # 1.0-300.0
    parser.add_argument('--deform_friction_coeff', type=float, default=0.1,
                        help='deform friction coefficient')
    # Texture args
    parser.add_argument('--deform_texture_file', type=str, default="textures/deform/orange_pattern.png",
                        help='Texture file for the deformable objects')
    parser.add_argument('--rigid_texture_file', type=str, default="textures/rigid/red_marble.png",
                        help='Texture file for the rigid objects')
    parser.add_argument('--plane_texture_file', type=str, default="textures/plane/lightwood.jpg",
                        help='Texture file for the plane (floor)')
    parser.add_argument('--use_random_textures', action='store_true',
                        help='Randomly selecting a texture for the rigid obj, deformable obj and floor within the texture folder')
    # Camera args.
    parser.add_argument('--cam_resolution', type=int, default=None,
                        help='RGB camera resolution in pixels (both with and '
                             'height). Use none to get only anchor poses.')
    # TODO: move this flag to a separate utlity.
    parser.add_argument('--cam_viewmat', type=float, nargs=6,
                        default=None,
                        help='Generate the view matrix for rendering camera'
                             '(not the debug camera). '
                             '[distance, pitch, yaw, posX, posY, posZ')
    # Parse args and do sanity checks.
    args = parser.parse_args()
    env_parts = args.env.split('-v')
    assert(len(env_parts) == 2 and env_parts[1].isdigit()), \
        '--env=[Task]-v[Version] (e.g. HangCloth-v0)'
    args.task = env_parts[0]
    args.version = int(env_parts[1])
    if args.task not in TASK_INFO.keys():
        print('Supported tasks are', list(TASK_INFO.keys()), 'got', args.task)
        exit(1)
    assert(args.version < len(TASK_INFO[args.task])) or args.task in ['HangProcCloth'], 'env version too high'
    return args
