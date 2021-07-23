
"""
Collect sequences of topo features from PyBullet simulations.

python -m topo_latents.collect_traj_data   --cam_outdir=/tmp/tmp_traj \
  --cam_rec_interval=10 --num_cpus=4 --num_runs=4 --scene_version=3 \
  --cloth_obj=cloth/ts_cardigan_sparser.obj \
  --debug --viz

python -m topo_latents.collect_traj_data \
  --cam_outdir=/tmp/tmp_traj --cam_rec_interval=10 --num_cpus=16 --num_runs=100

@contactrika

"""
import argparse
from copy import copy
import logging
import os
import sys
import time
import multiprocessing as mp
import numpy as np
import pybullet
np.set_printoptions(precision=2, linewidth=150, threshold=10000, suppress=True)

from dedo.utils.args import get_args
# from gym_bullet_deform.utils.cloth_utils import *


from gym_bullet_deform.simulator import DeformSim
from gym_bullet_deform.utils.process_camera import ProcessCamera
from gym_bullet_deform.utils.topo_utils import (
    make_colors, track_loops, extract_topo_feats, TOPO_FEAT_COORDS_IDX)
from .base_collector import BaseCollector
from scipy import spatial
from OpenGL.GL import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import imageio

from gym_bullet_deform.utils.cloth_objects_dict import (
    CLOTH_OBJECTS_DICT, MAX_TRUE_LOCS_NLOOPS, MAX_TRUE_LOCS_SIZE)
from .topo_distr import centers_of_loops
# Gains and limits for a simple PD controller for the anchors.
CTRL_MAX_FORCE = 10.0
CTRL_PD_KP = 100.0
CTRL_PD_KD = 50.0
NSTEPS_PER_WPT = 50
# Number of simulation steps per waypoint.
RENDER_WINDOW_NAME = 'Bullet Physics ExampleBrowser using OpenGL3+ [btgl] Release build'
BUFFER_FILE_LOC = '/tmp/screen2.png'

def get_all_args():
    args, main_parser = get_args()
    parser = argparse.ArgumentParser(
        description="CollectTrajData", parents=[main_parser])
    parser.add_argument('--view-angle', type=int, default=None, required=False,
                        help='specify which view angle is used')
    parser.add_argument('--max_nloops', type=int, default=5,
                        help='Maximum number of topo loops to extract')
    parser.add_argument('--viz-only', action='store_true', help='skips topo features and saving')
    parser.add_argument('--viz_wp', action='store_true', help='visualizez way points')
    args, unknown = parser.parse_known_args()
    return args



class VizTrajData(BaseCollector):
    def gen_wpt(self):
        x = 0.06;
        y = 0.40;
        z = 0.65
        # We repeat the start point to let the object drop under gravity without
        # starting the anchor motion. This will make states from nsteps_per_waypt
        # onward look physical (as opposed to a mesh being in an inflated state
        # when it is just loaded into simulation).
        min_z = 0.42
        left = []
        strt_left = np.array([-x, y, z])  # start loc of 1st anchor
        strt_right = np.array([x, y, z])  # start loc of 2nd anchor
        midl_left = np.array([-x, y - 0.1, z - 0.1])
        midl_right = np.copy(midl_left)
        midl_right[0] += 2 * x
        end_waypoints_left = np.array([-x, y - 0.35, min_z])
        end_waypoints_right = np.copy(end_waypoints_left)
        end_waypoints_right[0] += 2 * x

        left = [strt_left, strt_left, midl_left, end_waypoints_left]
        right = [strt_right, strt_right, midl_right, end_waypoints_right]
        return left, right
    def _make_waypoints(self):
        # TODO Load way points from past
        sim = self.rs.sim

        left, right = self.gen_wpt()
        if self.has_preset:
            left, right= CLOTH_OBJECTS_DICT[args.cloth_obj]['traj_preset'][self.rs.scene_name]['waypoints']
            # Hack
            # right = np.copy(left)

        # Visualize wp
        for i, pos in enumerate(left):

            # left[i][-1] += 0.18
            pos = left[i]
            # self.rs.add_waypoint_viz(pos)
        for i, pos in enumerate(right):

            # right[i][-1] += 0.18
            pos = right[i]
            # self.rs.add_waypoint_viz(pos, rgba=[0.5, 0.2, 0.2, 1])

        waypoints_left = np.vstack(left)
        waypoints_right = np.vstack(right)
        nwpts = waypoints_left.shape[0]
        # if waypoints_noise > 0:
        #     waypoints_left[2:,:] += (np.random.rand(2,3)-0.5)*2*waypoints_noise
        #     waypoints_right[2:,:] += (np.random.rand(2,3)-0.5)*2*waypoints_noise
        # number of sim steps per waypoint transition: how many simulation
        # steps should it take to go from one waypoint to the next
        steps_per_waypoint = [NSTEPS_PER_WPT] * nwpts
        #
        # getMeshData returns a tuple of two items: res[0] is the total number
        # over vertices in the mesh; res[1] is the list of 3D positions of these
        # vertices in world coordinates.

        left_start_anc, right_start_anc = self.rs.get_closest_anchor(np.array([waypoints_left[0], waypoints_right[0]]))
        cloth_anchored_vertex_ids = self.rs.cloth_anchored_vertex_ids
        print('left_start = ', left_start_anc)
        print('right_Start = ', right_start_anc)
        print('cloth_anchored_vertex_ids = ', cloth_anchored_vertex_ids)
        if self.has_preset and 'start_immediately' in CLOTH_OBJECTS_DICT[args.cloth_obj]['traj_preset'][self.rs.scene_name]:
            waypoints_left[0:1] = left_start_anc
            waypoints_right[0:1] = right_start_anc
        else:
            waypoints_left[0:2] = left_start_anc
            waypoints_right[0:2] = right_start_anc

        # Compose waypoints.
        all_waypoints = [waypoints_left.tolist(), waypoints_right.tolist()]
        print('all_waypoints', np.array(all_waypoints))
        # sanity checks (do not modify these unless we update main code)
        assert (len(all_waypoints) == len(cloth_anchored_vertex_ids))
        for i in range(len(all_waypoints)):
            assert (len(all_waypoints[i]) == len(steps_per_waypoint))
            # assert ((np.array(cloth_anchored_vertex_ids) < num_mesh_vertices).all())

        return all_waypoints, steps_per_waypoint, cloth_anchored_vertex_ids


    def setup_feature_dicts(self, feature_state_dict):
        feature_state_dict['gif'] = []


    def get_features(self, step, feature_state_dict, main_obj_id):
        _, _, rgbaImg, _, _ = self.rs.sim.getCameraImage(800, 600, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        feature_state_dict['gif'].append(rgbaImg)


    def save_features(self, feature_state_dict):
        args = self.args
        gif = feature_state_dict['gif']
        cloth_name = os.path.basename(args.cloth_obj)
        fname = f'/topo_traj/viz_debug/out/simulator/{cloth_name}_s{args.scene_version}_seed{args.seed}_angle{args.view_angle}.gif'
        print('saving...',fname)
        imageio.mimwrite(
            fname,
            gif, fps=12)


    def set_camera(self, sim):

        campers = [
            [0.9, -15, -30],  # a view from the back from the top
            [0.9, 15, -30],  # from the back slightly side (for laundry bag)
            [0.8, 160, -30],  # from the front from the top (good for apron)
            [0.65, 88, -50],  # from the back from the top (good for apron)
            [1.1, 150, -35],  # from the side (good for backpack)
            [1.3, 200, -25],  # straight from the front (for paperbag)
            [1.0, 20, -50],  # from the back side (for apron)
        ]
        if self.args.view_angle is not None:
            campr = campers[self.args.view_angle]  # from the back from the top (good for apron)
        else:
            campr = campers[4]

        cam_tgt = (0.0, 0.0, 0.3)
        # If an object already has a preset trajectory, use that instead and set the camera properly
        scene_name = self.rs.scene_name
        if self.has_preset:
            preset_traj = CLOTH_OBJECTS_DICT[args.cloth_obj]['traj_preset'][scene_name]
            campr = preset_traj['camera_pos']
            cam_tgt = preset_traj['camera_target']


        sim.resetDebugVisualizerCamera(
        cameraDistance=campr[0], cameraYaw=campr[1], cameraPitch=campr[2],
        cameraTargetPosition=cam_tgt)

    def setup(self):
        super().setup()
        if self.args.scene_version == 5:
            fix_anc1 = [-0.11, 0.05, 0.1]
            fix_anc2 = [-0.11, 0.05, 0.3]
            fix_anc3 = [-0.11, 0.05, 0.23]
            fix_anc4 = [-0.11, 0.05, 0.15]
            self.rs.add_waypoint_viz(fix_anc3, rgba=[1, 1, 1, 1])
            anc_locs, vert_indices = self.rs.find_closest_anchor_points([fix_anc1, fix_anc2, fix_anc3, fix_anc4])
            print('anc_loc_vertices', vert_indices)
            # self.rs.create_anchor_geom(anc_loc[0])
            cloth_id = self.rs.main_obj_ids[-1]

            for i, anc_loc in enumerate(anc_locs):
                anc_id = self.rs.create_anchor_geom(anc_loc)
                for v in vert_indices[i]:
                    self.rs.sim.createSoftBodyAnchor(cloth_id, v, bodyUniqueId=anc_id)


        # self.rs.add_waypoint_viz(anc_loc[2], rgba=[0, 0, 0, 1])



def run_one(args,scene_version):

    args.scene_version = scene_version  # 0: pole, 1: hook ; 2: figure
    for k,v in CLOTH_OBJECTS_DICT[args.cloth_obj].items():
        assert(hasattr(args, k)), 'unknown arg '+k; setattr(args, k, v)

    rs = DeformSim(args, viz=args.viz)
    kwargs = {
        'sim_gravity': args.sim_gravity,
        'sim_frequency': args.sim_frequency,
    }
    rs.setup(**kwargs)
    rs.setup_scene_rigid_obj()
    rs.setup_camera()

    # Loading soft object
    cloth_obj_path = os.path.join(args.data_path, args.cloth_obj)
    # If using procedural, load directly from cloth_obj
    if 'procedural' in args.cloth_obj: cloth_obj_path = args.cloth_obj
    # Loading soft body
    cloth_id = rs.load_soft_object(
        cloth_obj_path, args.cloth_scale,
        args.cloth_init_pos, args.cloth_init_ori,
        args.cloth_bending_stiffness, args.cloth_damping_stiffness,
        args.cloth_elastic_stiffness, args.cloth_friction_coeff,
        args.cloth_fuzz_stiffness, args.debug, args.cloth_scale_noise)
    rs.add_main_object(cloth_id)

    all_waypoints, steps_per_waypoint, cloth_anchored_vertex_ids = make_all_waypoints(sim = rs.sim,
                                                                                      cloth_id = cloth_id,
                                                                                      cloth_obj_path = cloth_obj_path,
                                                                                      waypoints_noise = args.waypoints_noise,
                                                                                      )

    anchor_ids, trajs = rs.generate_anchor_traj(all_waypoints, steps_per_waypoint, cloth_anchored_vertex_ids)

    sim = rs.sim
    output_data_file_pfx = f'{rs.output_data_file_pfx}_seed{args.seed:05d}'
    main_loop_collect(rs,
                      args,
                      output_data_file_pfx,
                      steps_per_waypoint = steps_per_waypoint,
                      trajs = trajs,
                      anchor_ids = anchor_ids,
                      cloth_id = cloth_id,
                      cloth_anchored_vertex_ids = cloth_anchored_vertex_ids,
                      )

    print('args', args)
    np.random.seed(args.seed)  # legacy, but ok for our purposes
    print('Done seed', args.seed, 'scene', args.scene_version, 'cloth', args.cloth_obj)



def main(args):

    args.cloth_fuzz_stiffness = True
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    args.data_path = os.path.join(
        curr_dir, '../..', 'gym-bullet-deform', 'gym_bullet_deform', 'data')
    # Run cloth sim in parallel.
    # extract_args_from_filename(filename, args)
    # TODO Extract object properties from file name

    traj_runner = VizTrajData(args, args.scene_version, viz=True)
    traj_runner.start()
    # run_one(args,
    #         args.scene_version)


if __name__ == "__main__":
    args = get_all_args()
    # filename = os.path.basename(args.trajfile)
    # Trajfile = np.load(args.trajfile)
    main(args)
