#
# Command line arguments.
#
# @contactrika
#
import argparse
import logging
import sys

from .task_info import TASK_TYPES


def get_args(parent=None):
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)])
    parser = argparse.ArgumentParser(description='args', add_help=False)
    # Main/demo args.
    parser.add_argument('--task', type=str,
                        default='Hang', help='Name of the task type',
                        choices=TASK_TYPES)
    parser.add_argument('--max_episode_len', type=int,
                        default=400, help='Number of simulation steps per task')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of runs/episodes to complete')
    parser.add_argument('--viz', action='store_true', help='Whether to visualize')
    parser.add_argument('--debug', action='store_true',
                        help='Whether to print debug info')
    # Simulation args. Note: turn up frequency when deform stiffness is high.
    parser.add_argument('--sim_frequency', type=int, default=500,
                        help='Number of simulation steps per second')  # 250-1K
    parser.add_argument('--sim_gravity', type=float, default=-9.8, help='Gravity')
    # Anchor/grasping args.
    parser.add_argument('--anchor_init_pos', type=float, nargs=3,
                        default=[-0.04, 0.40, 0.70],
                        help='Initial position for an anchor')
    parser.add_argument('--other_anchor_init_pos', type=float, nargs=3,
                        default=[0.04, 0.40, 0.70],
                        help='Initial position for another anchors')
    # SoftBody args.
    parser.add_argument('--deform_obj', type=str, default=None,
                        help='Obj file for deform item')
    parser.add_argument('--deform_init_pos', type=float, nargs=3,
                        default=[0,0,0.42],
                        help='Initial pos for the center of the deform object')
    parser.add_argument('--deform_init_ori', type=float, nargs=3,
                        default=[0,0,0],
                        help='Initial orientation for deform (in Euler angles)')
    parser.add_argument('--deform_scale', type=float, default=1.0,
                        help='Scaling for the deform object')
    parser.add_argument('--deform_noise', type=float, default=0.0,
                        help='Add noise to deform point cloud (0.01 ok)')
    parser.add_argument('--deform_bending_stiffness', type=float, default=30.0,
                        help='deform spring elastic stiffness (k)')  # 1.0-300.0
    parser.add_argument('--deform_damping_stiffness', type=float, default=1.0,
                        help='deform spring damping stiffness (c)')
    parser.add_argument('--deform_elastic_stiffness', type=float, default=30.0,
                        help='deform spring elastic stiffness (k)')  # 1.0-300.0
    parser.add_argument('--deform_friction_coeff', type=float, default=0.0,
                        help='deform friction coefficient')
    parser.add_argument('--deform_fuzz_stiffness', type=int, default=1,
                        help='Whether to randomize bending and elastic stiffness')
    # Camera args.
    parser.add_argument('--cam_resolution', type=int, default=None,
                        help='RGB camera resolution in pixels (both with and '
                             'height). Use none to get only anchor poses.')
    #
    args, unknown = parser.parse_known_args()
    return args, parser
