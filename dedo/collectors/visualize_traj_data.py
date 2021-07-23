
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

from gym_bullet_deform.utils.args import get_args
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
        pass

    def get_features(self, step, feature_state_dict, main_obj_id):
        _, _, rgbaImg, _, _ = self.rs.sim.getCameraImage(800, 600, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        feature_state_dict['gif'].append(rgbaImg)
        pass

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


def make_all_waypoints(sim, cloth_id, cloth_obj_path,
                       waypoints_noise):

    x = 0.06;
    y = 0.40;
    z = 0.65
    min_z = 0.42
    strt_left = np.array([-x, y, z])  # start loc of 1st anchor
    strt_right = np.array([x, y, z])  # start loc of 2nd anchor
    midl_left = np.array([-x, y - 0.1, z - 0.1])
    midl_right = np.copy(midl_left)
    midl_right[0] += 2 * x
    end_waypoints_left = np.array([-x, y - 0.35, min_z])
    end_waypoints_right = np.copy(end_waypoints_left)
    end_waypoints_right[0] += 2 * x
    # We repeat the start point to let the object drop under gravity without
    # starting the anchor motion. This will make states from nsteps_per_waypt
    # onward look physical (as opposed to a mesh being in an inflated state
    # when it is just loaded into simulation).
    waypoints_left = np.vstack(
        [strt_left, strt_left, midl_left, end_waypoints_left])
    waypoints_right = np.vstack(
        [strt_right, strt_right, midl_right, end_waypoints_right])
    nwpts = waypoints_left.shape[0]
    # if waypoints_noise > 0:
    #     waypoints_left[2:,:] += (np.random.rand(2,3)-0.5)*2*waypoints_noise
    #     waypoints_right[2:,:] += (np.random.rand(2,3)-0.5)*2*waypoints_noise
    # number of sim steps per waypoint transition: how many simulation
    # steps should it take to go from one waypoint to the next
    steps_per_waypoint = [NSTEPS_PER_WPT]*nwpts
    #
    # getMeshData returns a tuple of two items: res[0] is the total number
    # over vertices in the mesh; res[1] is the list of 3D positions of these
    # vertices in world coordinates.
    mesh_info = sim.getMeshData(cloth_id)
    num_mesh_vertices = mesh_info[0]
    mesh_vetex_positions = mesh_info[1]
    # TODO: make this less hackish later.
    cloth_nm = '/'.join(cloth_obj_path.split('/')[-2:])
    print('cloth_nm', cloth_nm)
    if 'cloth_anchored_vertex_ids' in CLOTH_OBJECTS_DICT[cloth_nm].keys():
        cloth_anchored_vertex_ids = CLOTH_OBJECTS_DICT[cloth_nm][
            'cloth_anchored_vertex_ids']
        for anc, anchor_vertex_ids in enumerate(cloth_anchored_vertex_ids):
            accum = np.array([0.0,0.0,0.0])
            for v in anchor_vertex_ids:
                v_pos = mesh_vetex_positions[v]
                accum += np.array(v_pos)
            accum /= len(anchor_vertex_ids)
            if anc == 0:
                waypoints_left[0:2] = accum
            else:
                waypoints_right[0:2] = accum
        print('Adjusted anchors start: ', waypoints_left[0], waypoints_right[0])
    else:
        cloth_anchored_vertex_ids = get_closest(
            [waypoints_left[0], waypoints_right[0]], mesh_vetex_positions)
        print('Attached anchors at: ', cloth_anchored_vertex_ids)
    # Compose waypoints.
    all_waypoints = [waypoints_left.tolist(), waypoints_right.tolist()]
    print('all_waypoints', np.array(all_waypoints))
    # sanity checks (do not modify these unless we update main code)
    assert(len(all_waypoints)==len(cloth_anchored_vertex_ids))
    for i in range(len(all_waypoints)):
        assert(len(all_waypoints[i]) == len(steps_per_waypoint))
        assert((np.array(cloth_anchored_vertex_ids)<num_mesh_vertices).all())

    return all_waypoints, steps_per_waypoint, cloth_anchored_vertex_ids



def main_loop_collect(rs, args, output_data_file_pfx,
                      steps_per_waypoint, trajs, anchor_ids,
                      cloth_id, cloth_anchored_vertex_ids):
    anchored_steps = sum(steps_per_waypoint[:-1])
    topo_generators = []
    topo_seqs = []
    ptclouds = []
    ancors_trajs = []
    true_locs = []
    nsteps = anchored_steps+steps_per_waypoint[-1]
    nanchors = len(anchor_ids)
    max_viz_clrs = 3
    COLORS = make_colors(max_viz_clrs)
    cloth_noise = 0.0
    cloth_args = CLOTH_OBJECTS_DICT[args.cloth_obj]
    sim = rs.sim
    gif = []
    if 'cloth_noise' in cloth_args.keys():
        cloth_noise = cloth_args['cloth_noise']
    if True:
        campers = [
            [0.9, -15, -30],  # a view from the back from the top
            [0.9, 15, -30],  # from the back slightly side (for laundry bag)
            [0.6, 130, -30],  # from the front from the top (good for apron)
            [0.65, 88, -50],  # from the back from the top (good for apron)
            [1.1, 150, -35],  # from the side (good for backpack)
            [1.3, 200, -25],  # straight from the front (for paperbag)
            [1.0, 20, -50],  # from the back side (for apron)
        ]
        if args.view_angle is not None:
            campr = campers[args.view_angle]  # from the back from the top (good for apron)
        else:
            campr = campers[4]

        cam_tgt = (0.35, 0.0, 0.3)
        # campr = [1.1, 50, -30]   # from the back and bottom
        # cam_tgt = (0.145, 0.179, -0.179)

        sim.resetDebugVisualizerCamera(
            cameraDistance=campr[0], cameraYaw=campr[1], cameraPitch=campr[2],
            cameraTargetPosition=cam_tgt)

    # topo_seqs_coords = Trajfile['topo_seqs']
    # # Extrac coordinates
    # topo_seqs_coords = topo_seqs_coords[..., TOPO_FEAT_COORDS_IDX]
    # _shp = topo_seqs_coords.shape
    # # Extract simplex and neighbours from the last dimension.
    # topo_seqs_coords = topo_seqs_coords.reshape(_shp[0], _shp[1], -1, 9)
    for step in range(nsteps):
        if step < anchored_steps:
            for i in range(nanchors):
                # sim.resetBasePositionAndOrientation(
                #     anchor_ids[i], trajs[i][step], quat)
                anc_pos, anc_ori = sim.getBasePositionAndOrientation(anchor_ids[i])
                anc_linvel, anc_angvel = sim.getBaseVelocity(anchor_ids[i])
                tgt_pos_vel = trajs[i][step]
                pos_diff = tgt_pos_vel[0:3] - np.array(anc_pos)
                vel_diff = tgt_pos_vel[3:6] - np.array(anc_linvel)
                force = CTRL_PD_KP*pos_diff + CTRL_PD_KD*vel_diff
                force = np.clip(force, -1.0*CTRL_MAX_FORCE, CTRL_MAX_FORCE)
                sim.resetBaseVelocity(
                    anchor_ids[i], linearVelocity=tgt_pos_vel[3:6],
                    angularVelocity=[0,0,0])
                sim.applyExternalForce(anchor_ids[i], -1, force.tolist(),
                                       [0,0,0], pybullet.LINK_FRAME)
                # Visualize deviations from target trajectory.
                if args.viz:
                    sim.addUserDebugLine(anc_pos, tgt_pos_vel[0:3], [1,0,0])
        elif step == anchored_steps:
            if args.debug: print('Releasing anchors at step', step)
            for i in range(len(anchor_ids)):
                sim.changeDynamics(anchor_ids[i], -1, mass=0.00001)
                sim.changeVisualShape(anchor_ids[i], -1, rgbaColor=[0.5,0.5,0.5,1])



        if args.viz and True: # step >= NSTEPS_PER_WPT:
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RENDERING, 0)
            sim.removeAllUserDebugItems()

        sim.stepSimulation()  # this advances physics sim


        #if step < NSTEPS_PER_WPT:
        #    continue  # we let the cloth fall under gravity firsts
        # topo_seq = topo_seqs_coords[int(step * PLAYBACK_SPEED)]
        rec_step = step
        if rec_step%args.cam_rec_interval == 0 or rec_step+1 == nsteps:
            print(rec_step)
            # Compute locations of true loops vertices and their convex hulls.
            _, vert_locs = sim.getMeshData(cloth_id)
            curr_true_locs = np.zeros(
                (MAX_TRUE_LOCS_NLOOPS, MAX_TRUE_LOCS_SIZE, 3))
            true_hulls = []
            cloth_obj_dict = CLOTH_OBJECTS_DICT[args.cloth_obj]

            # !!! Below are for plotting ground truth hull, not really needed for viz I don't think
            # if 'cloth_true_loop_vertices' in cloth_obj_dict.keys():
            #     true_loop_vs_lst = cloth_obj_dict['cloth_true_loop_vertices']
            #     for true_loop_i, true_loop_vs in enumerate(true_loop_vs_lst):
            #         for tmp_i, true_v in enumerate(true_loop_vs):
            #             if tmp_i >= MAX_TRUE_LOCS_SIZE:
            #                 print('WARNING: discarding true v at index', tmp_i)
            #                 break
            #             curr_true_locs[true_loop_i,tmp_i,:] = vert_locs[true_v]
            #
            #     true_hulls = []  # only for debugging visualization here
            #     for true_loop_i in range(len(true_loop_vs_lst)):
            #         tmp_nvs = len(true_loop_vs_lst[true_loop_i])
            #         tmp_vs = curr_true_locs[true_loop_i,:tmp_nvs]
            #         true_hull = spatial.ConvexHull(tmp_vs)
            #         true_hulls.append(true_hull)
            #         for s in true_hull.simplices:
            #             tmp_smplx = curr_true_locs[true_loop_i,s]
            #             for tmp_i in range(3):
            #                 sim.addUserDebugLine(
            #                     tmp_smplx[tmp_i], tmp_smplx[(tmp_i+1)%3],
            #                     (0.7,0.7,0), lifeTime=3.0)
            #     true_locs.append(curr_true_locs)


            # Do sim step.
            # if args.viz and step%10 == 0:
            #      res = sim.getDebugVisualizerCamera()
            #      print('DebugVisualizerCamera yaw pitch dist target', res[-4:])
            ptcloud, _ = ProcessCamera.render_soft(
                sim, [cloth_id], debug=args.debug)
            if cloth_noise > 0:
                rnd = np.random.rand(*ptcloud.shape)
                ptcloud = ptcloud + (rnd-0.5)*2*cloth_noise

            get_features(COLORS, anchor_ids, ancors_trajs, args, max_viz_clrs, nanchors, output_data_file_pfx, ptcloud,
                         ptclouds, rec_step, sim, topo_generators, topo_seqs, true_hulls)

        # # Loop through the 5 tracked loops
        # for loop_idx, topo in enumerate(topo_seq):
        #     clr = COLORS[(loop_idx) % max_viz_clrs][:3]
        #     # Loop through the simplex triangle and neighbour hood triangles
        #     for triangel in topo:
        #         # pt1 -> pt2
        #         pt0 = triangel[:3]
        #         pt1 = triangel[3:6]
        #         pt2 = triangel[6:]
        #         sim.addUserDebugLine(pt0, pt1, clr)
        #         sim.addUserDebugLine(pt0, pt1, clr)
        #         sim.addUserDebugLine(pt0, pt1, clr)

            # if args.viz:  # turn viz back on and pause to look
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RENDERING, 1)
            ### Yonk's addition
            # HACKCY HACK WAY TO CAPTURE USING IMAGEMAGICK
            if not args.viz_only:
                os.system(f' magick import -window "{RENDER_WINDOW_NAME}" {BUFFER_FILE_LOC}')
                cam_rgb = imageio.imread(BUFFER_FILE_LOC)
                gif.append(cam_rgb)
                print(step)
            else:
                if step % 4 == 0: input('presss any key to continue...')
                pass
        ###

                # time.sleep(1)
    # Done. Write output to file if needed.
    if not args.viz_only:
        cloth_name = os.path.basename(args.cloth_obj)
        imageio.mimwrite(f'/topo_traj/viz_debug/out/simulator/{cloth_name}_s{args.scene_version}_seed{args.seed}_angle{args.view_angle}.gif', gif, fps=6)

def get_features(COLORS, anchor_ids, ancors_trajs, args, max_viz_clrs, nanchors, output_data_file_pfx, ptcloud,
                 ptclouds, rec_step, sim, topo_generators, topo_seqs, true_hulls):
    if not args.viz_only and False:  # disabling topo features
        topo_generators = track_loops(
            ptcloud, rec_step, output_data_file_pfx, topo_generators,
            nloops=args.max_nloops, sim_freq=args.sim_frequency,
            viz=args.viz, debug=args.debug)
        topo_feats = extract_topo_feats(
            topo_generators, args.max_nloops)
        topo_seqs.append(topo_feats)
        ptclouds.append(ptcloud)
        anchor_poses = []
        for tmpi in range(nanchors):
            pos, _ = sim.getBasePositionAndOrientation(anchor_ids[tmpi])
            anchor_poses.append(pos)
        ancors_trajs.append(anchor_poses)

        lifetimes = []
        nbest = 3  # viz loops with 3 largest lifetimes
        for g in topo_generators:
            lifetimes.append(g.death - g.birth)
        best_lts = np.array(lifetimes).argsort()[::-1][:nbest]
        for idx, g in enumerate(topo_generators):
            lt = g.death - g.birth
            clr = COLORS[(idx) % max_viz_clrs][:3]
            if args.debug: print(f'lt {lt:0.6f}')
            if idx not in best_lts or lt * 100 < 1.0: continue
            # draw triangles as user debug lines
            if g is not None and g.loop.shape[0] >= 3:
                g_center = np.array(g.loop[0:3]).mean(axis=0)
                center_in_hull = False
                for hull in true_hulls:
                    if point_in_hull(g_center, hull):
                        center_in_hull = True
                for i in range(3):
                    sim.addUserDebugLine(
                        g.loop[i], g.loop[(i + 1) % 3], clr)
                    if center_in_hull:
                        sim.addUserDebugLine(
                            g.loop[i], g_center, (1, 0, 1))
                for neigh_idx, neigh in enumerate(g.neighborhood):
                    # print(neigh.points, neigh.filtration_value)
                    for tmp_i in range(3):
                        sim.addUserDebugLine(
                            neigh.points[tmp_i, :],
                            neigh.points[(tmp_i + 1) % 3, :], clr)
        # pybullet.configureDebugVisualizer(
        #     pybullet.COV_ENABLE_RENDERING, 1)

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
