import numpy as np
import pybullet
import os
import sys

from ..sim import DeformSim
from dedo.utils.task_info import DEFORM_INFO

class BaseCollector:
    CTRL_MAX_FORCE = 10.0
    CTRL_PD_KP = 100.0
    CTRL_PD_KD = 50.0
    NSTEPS_PER_WPT = 200
    def __init__(self, args,  scene_version, viz=False, output_data_file_pfx=None):
        self.args = args
        self.scene_version = scene_version
        self.viz = viz
        self.ft = {}
        self.output_data_file_pfx = output_data_file_pfx

        self.setup()


    def _make_waypoints(self, *kwargs):
        raise NotImplementedError('Make Waypoint method has not been implemented ')

    def setup_feature_dicts(self, feature_state_dict):
        raise NotImplementedError('setup_feature_dicts method has not been implemented')

    def get_features(self, step, feature_state_dict, main_obj_id):
        raise NotImplementedError('get_features method has not been implemented')

    def save_features(self, feature_state_dict,):
        raise NotImplementedError('save_features method has not been implemented')

    def start(self):
        feature_state_dict = {}
        self.setup_feature_dicts(feature_state_dict)
        anchored_steps = sum(self.steps_per_waypoint[:-1])
        nsteps = anchored_steps + self.steps_per_waypoint[-1]#  + 100
        sim = self.rs.sim
        self.set_camera(sim)

        # topo_seqs_coords = Trajfile['topo_seqs']
        # # Extrac coordinates
        # topo_seqs_coords = topo_seqs_coords[..., TOPO_FEAT_COORDS_IDX]
        # _shp = topo_seqs_coords.shape
        # # Extract simplex and neighbours from the last dimension.
        # topo_seqs_coords = topo_seqs_coords.reshape(_shp[0], _shp[1], -1, 9)
        for step in range(nsteps):
            if step < anchored_steps:
                for i in range(len(self.anchor_ids)):
                    self.move_anchor(i, sim, step)
            elif step == anchored_steps:
                if self.args.debug: print('Releasing anchors at step', step)
                self.release_anchors(sim)

            # clean up rendering
            if self.args.viz and True:  # step >= NSTEPS_PER_WPT:
                pybullet.configureDebugVisualizer(
                    pybullet.COV_ENABLE_RENDERING, 0)
                sim.removeAllUserDebugItems()



            rec_step = step
            if rec_step % self.args.cam_rec_interval == 0 or rec_step + 1 == nsteps:

                self.get_features(step, feature_state_dict, self.main_obj_id)
                  # turn viz back on and pause to look
                # TODO For some applications we always render
                # if args.viz:
                pybullet.configureDebugVisualizer(
                    pybullet.COV_ENABLE_RENDERING, 1)
                ### Yonk's addition
            ###
            sim.stepSimulation()  # this advances physics sim
            print(rec_step)
            # time.sleep(1)
        self.save_features(feature_state_dict)
        # Done. Write output to file if needed.

    def release_anchors(self, sim):
        for i in range(len(self.anchor_ids)):
            sim.removeConstraint(self.anchor_ids[i])
            sim.changeVisualShape(self.anchor_ids[i], -1, rgbaColor=[0.5, 0.5, 0.5, 1])

    def set_camera(self, sim):
        campers = [
            [0.9, -15, -30],  # a view from the back from the top
            [0.9, 15, -30],  # from the back slightly side (for laundry bag)
            [1.3, 160, -30],  # from the front from the top (good for apron)
            [0.65, 88, -50],  # from the back from the top (good for apron)
            [1.1, 150, -35],  # from the side (good for backpack)
            [1.3, 200, -25],  # straight from the front (for paperbag)
            [1.0, 20, -50],  # from the back side (for apron)
        ]
        if self.args.view_angle is not None:
            campr = campers[self.args.view_angle]  # from the back from the top (good for apron)
        else:
            campr = campers[4]

        cam_tgt = (0.35, 0.0, 0.3)
        # campr = [1.1, 50, -30]   # from the back and bottom
        # cam_tgt = (0.145, 0.179, -0.179)

        sim.resetDebugVisualizerCamera(
            cameraDistance=campr[0], cameraYaw=campr[1], cameraPitch=campr[2],
            cameraTargetPosition=cam_tgt)

    def move_anchor(self, i, sim, step):
        # sim.resetBasePositionAndOrientation(
        #     anchor_ids[i], trajs[i][step], quat)
        anc_pos, anc_ori = sim.getBasePositionAndOrientation(self.anchor_ids[i])
        anc_linvel, anc_angvel = sim.getBaseVelocity(self.anchor_ids[i])
        tgt_pos_vel = self.trajs[i][step]
        pos_diff = tgt_pos_vel[0:3] - np.array(anc_pos)
        vel_diff = tgt_pos_vel[3:6] - np.array(anc_linvel)
        force = self.CTRL_PD_KP * pos_diff + self.CTRL_PD_KD * vel_diff
        force = np.clip(force, -1.0 * self.CTRL_MAX_FORCE, self.CTRL_MAX_FORCE)
        sim.resetBaseVelocity(
            self.anchor_ids[i], linearVelocity=tgt_pos_vel[3:6],
            angularVelocity=[0, 0, 0])
        sim.applyExternalForce(self.anchor_ids[i], -1, force.tolist(),
                               [0, 0, 0], pybullet.LINK_FRAME)
        # Visualize deviations from target trajectory.
        if self.args.viz:
            sim.addUserDebugLine(anc_pos, tgt_pos_vel[0:3], [1, 0, 0])

    def setup(self):
        self._setup_scene()
        self._setup_trajectory()
    @property
    def has_preset(self):
        return self.args.deform_obj in DEFORM_INFO and 'traj_preset' in DEFORM_INFO[
            self.args.deform_obj] and self.rs.scene_name in \
        DEFORM_INFO[self.args.deform_obj]['traj_preset']

    def _setup_scene(self):
        args = self.args
        print('args', args)
        print('Done seed', args.seed, 'scene', args.scene_version, 'cloth', args.deform_obj)

        np.random.seed(args.seed)  # legacy, but ok for our purposes
        scene_version = args.scene_version  # 0: pole, 1: hook ; 2: figure
        if args.deform_obj in DEFORM_INFO:
            for k, v in DEFORM_INFO[args.deform_obj].items():
                if not hasattr(args, k): print( 'unknown arg ' + k)
                setattr(args, k, v)

        self.rs = self._setup_sim(args)
        # TODO Setup scene

        main_obj_id, main_obj_filepath = self._setup_main_object(args, self.rs)
        #
        # anchor_ids, cloth_anchored_vertex_ids, steps_per_waypoint, trajs = self.setup_trajectory(args, main_obj_id,
        #                                                                                          main_obj_filepath, rs)

        self.main_obj_id = main_obj_id
        self.main_obj_filepath = main_obj_filepath
        # output_data_file_pfx = f'{self.rs.output_data_file_pfx}_seed{args.seed:05d}'



    def _setup_trajectory(self):
        waypoint_noise = self.args.waypoints_noise
        all_waypoints, steps_per_waypoint, cloth_anchored_vertex_ids = self._make_waypoints()
        anchor_ids, trajs = self.rs.generate_anchor_traj(all_waypoints, steps_per_waypoint, cloth_anchored_vertex_ids)

        self.anchor_ids = anchor_ids
        self.trajs = trajs
        self.steps_per_waypoint = steps_per_waypoint


    def _setup_main_object(self, args, rs):
        '''Loads the main object from the simulator'''
        # Loading soft object
        deform_obj_path = os.path.join(args.data_path, args.deform_obj)
        # If using procedural, load directly from deform_obj
        if 'procedural' in args.deform_obj: deform_obj_path = args.deform_obj
        # Loading soft body
        cloth_id = rs.load_soft_object(
            deform_obj_path, args.cloth_scale,
            args.cloth_init_pos, args.cloth_init_ori,
            args.cloth_bending_stiffness, args.cloth_damping_stiffness,
            args.cloth_elastic_stiffness, args.cloth_friction_coeff,
            args.cloth_fuzz_stiffness, args.debug, args.cloth_scale_noise)

        rs.add_main_object(cloth_id)

        return cloth_id, deform_obj_path

    def _setup_sim(self, args):
        rs = DeformSim(args, viz=args.viz)
        kwargs = {
            'sim_gravity': args.sim_gravity,
            'sim_frequency': args.sim_frequency,
        }
        rs.setup(**kwargs)
        rs.setup_scene_rigid_obj()
        rs.setup_camera()
        return rs