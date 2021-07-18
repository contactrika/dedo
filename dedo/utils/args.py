#
# Command line arguments for camera and deform utils.
#
# @contactrika
#
import argparse
import logging
import os
import sys


def get_args(parent=None):
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)])
    parser = argparse.ArgumentParser(
        description='utils',add_help=False)
    # Main/demo args.
    parser.add_argument('--task_name', type=str,
                        default='scooping', help='Name of the task')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of runs/episodes to complete')
    parser.add_argument('--waypoints_noise', type=float, default=0.05,
                        help='Add uniform noise for point clouds')
    parser.add_argument('--viz', action='store_true', help='Whether to visualize')
    parser.add_argument('--debug', action='store_true',
                        help='Whether to print debug info')
    # Simulation args. Note: turn up frequency when deform stiffness is high.
    parser.add_argument('--sim_frequency', type=int, default=500,
                        help='Number of simulation steps per second') # 250-1K
    parser.add_argument('--sim_gravity', type=float, default=-9.8, help='Gravity')
    parser.add_argument('--floor_texture_fname', type=str, default=None,
                        help='File name for the floor texture from data/planes')
    # Rigid obj args.
    parser.add_argument('--rigid_custom_obj', type=str, default=None,
                        help='Obj file for a custom rigid object in the scene')
    parser.add_argument('--rigid_scale', type=float, nargs=3, default=[1,1,1],
                        help='Scaling for the custom rigid object')
    parser.add_argument('--rigid_init_pos', type=float, nargs=3, default=[0,0,0.51],
                        help='Initial pos for the center of custom rigid object')
    parser.add_argument('--rigid_init_ori', type=float, nargs=3, default=[0,0,0],
                        help='Initial orientation for custom rigid object '
                             '(in Euler angles)')
    # Anchor/grasping args.
    parser.add_argument('--direct_control', action='store_true',
                        help='Override anchor pos instead of '
                             'interpreting actions as forces')
    # deform/SoftBody obj args.
    parser.add_argument('--deform_obj', type=str, default=None,
                        help='Obj file for deform item')
    parser.add_argument('--deform_init_pos', type=float, nargs=3,
                        default=[0,0,0.65],
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
    # Camera args.
    parser.add_argument('--cam_outdir', type=str, default=None,
                        help='Directory for depth and RGB camera output')
    parser.add_argument('--cam_resolution', type=int, default=100,
                        help='Point cloud resolution')
    parser.add_argument('--cam_rec_interval', type=int, default=1,
                        help='How many steps to skip between each cam shot')
    # Lfd args.
    parser.add_argument('--demo_path', type=str, default=None,
                        help='Directory for demonstration file in npz format.')
    parser.add_argument('--model_prefix', type=str, default=None,
                        help='Directory name of transporter model being used.')
    args, unknown = parser.parse_known_args()
    return args, parser
