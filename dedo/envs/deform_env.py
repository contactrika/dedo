#
# Dynamic environments with deformable objects.
#
# @contactrika, @pyshi
#
import os
import time

import numpy as np
import gym
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bclient

from ..utils.anchor_utils import (
    create_anchor, attach_anchor, create_anchor_geom, command_anchor_velocity,
    pin_fixed)
from ..utils.init_utils import (
    load_deform_object, load_rigid_object, reset_bullet, get_preset_properties)
from ..utils.mesh_utils import get_mesh_data
from ..utils.task_info import CAM_INFO, DEFORM_INFO, SCENE_INFO, TASK_INFO


class DeformEnv(gym.Env):
    MAX_OBS_VEL = 20.0  # max vel (in m/s) for the anchor observations
    MAX_ACT_VEL = 10.0  # max vel (in m/s) for the anchor actions
    NUM_ANCHORS = 2
    WORKSPACE_BOX_SIZE = 20.0  # workspace box limits (needs to be >=1)

    def __init__(self, args):
        self.args = args
        self.n_max_episode_steps = int(args.max_episode_len * args.ctrl_freq)
        self.cam_on = args.cam_resolution is not None
        self.cam_args = {
            'cameraDistance': 11.4,
            'cameraPitch': -22.4,
            'cameraYaw': 257,
            'cameraTargetPosition': np.array([-0.08, -0.29, 1.8])
        }
        # Initialize sim and load objects.
        self.sim = bclient.BulletClient(
            connection_mode=pybullet.GUI if args.viz else pybullet.DIRECT)
        reset_bullet(args, self.sim, self.cam_on, self.cam_args, args.debug)
        self.rigid_ids, self.deform_id, self.deform_obj, self.goal_pos = \
            self.load_objects(self.sim, self.args)
        # setting cam_args again after loading preset data
        if self.args.cam_viewmat is not None:
            dist, pitch, yaw, pos_x, pos_y, pos_z = self.args.cam_viewmat
            self.cam_args = {
                'cameraDistance': dist,
                'cameraPitch': pitch,
                'cameraYaw': yaw,
                'cameraTargetPosition': np.array([pos_x, pos_y, pos_z])
            }
        # Define sizes of observation and action spaces.
        self.anchor_lims = np.tile(np.concatenate(
            [DeformEnv.WORKSPACE_BOX_SIZE * np.ones(3),  # 3D pos
             np.ones(3)]), DeformEnv.NUM_ANCHORS)        # 3D linvel/MAX_OBS_VEL
        if args.cam_resolution is None:
            self.observation_space = gym.spaces.Box(
                -1.0 * self.anchor_lims, self.anchor_lims)
        else:  # RGB
            self.observation_space = gym.spaces.Box(
                np.zeros((args.cam_resolution, args.cam_resolution, 3)),
                np.ones((args.cam_resolution, args.cam_resolution, 3)))
        self.action_space = gym.spaces.Box(  # [-1,1]
            -1.0 * np.ones(DeformEnv.NUM_ANCHORS * 3),
            np.ones(DeformEnv.NUM_ANCHORS * 3))
        # Loading done, turn on visualizer if needed
        if self.args.viz:
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

    @property
    def anchor_ids(self):
        return list(self.anchors)

    def load_objects(self, sim, args):
        scene_name = self.args.task.lower()
        # if scene_name.startswith('hang'):
        #     scene_name = 'hang'  # same scene for 'HangBag', 'HangCloth'
        if scene_name.startswith('button'):
            scene_name = 'button'  # same human figure for dress and mask tasks
        data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
        sim.setAdditionalSearchPath(data_path)
        #
        # Load rigid objects.
        #
        rigid_ids = []
        for name, kwargs in SCENE_INFO[scene_name]['entities'].items():
            pth = os.path.join(data_path, name)
            rgba_color = kwargs['rgbaColor'] if 'rgbaColor' in kwargs else None
            id = load_rigid_object(
                sim, pth, kwargs['globalScaling'],
                kwargs['basePosition'], kwargs['baseOrientation'], rgba_color)
            rigid_ids.append(id)
        #
        # Load deformable object.
        #
        if args.override_deform_obj is not None:
            deform_obj = args.override_deform_obj
        else:
            assert (args.task in TASK_INFO)  # already checked in args
            assert (args.version < len(TASK_INFO[args.task]))  # checked in args
            deform_obj = TASK_INFO[args.task][args.version]
            for arg_nm, arg_val in DEFORM_INFO[deform_obj].items():
                setattr(args, arg_nm, arg_val)

        texture_path = os.path.join(
            data_path, 'textures', 'blue_bright.png')
        deform_id = load_deform_object(
            sim, deform_obj, texture_path, args.deform_scale,
            args.deform_init_pos, args.deform_init_ori,
            args.deform_bending_stiffness, args.deform_damping_stiffness,
            args.deform_elastic_stiffness, args.deform_friction_coeff,
            args.debug)
        if scene_name == 'button':  # pin cloth edge for buttoning task
            assert ('deform_fixed_anchor_vertex_ids' in DEFORM_INFO[deform_obj])
            pin_fixed(sim, deform_id,
                      DEFORM_INFO[deform_obj]['deform_fixed_anchor_vertex_ids'])
        #
        # Mark the goal.
        #
        goal_poses = SCENE_INFO[scene_name]['goal_pos']
        if args.viz:
            for i, goal_pos in enumerate(goal_poses):
                alpha = 1 if i == 0 else 0.3  # primary vs secondary goal
                create_anchor_geom(sim, goal_pos, mass=0.0,
                                   rgba=(0, 1, 0, alpha), use_collision=False)
        return rigid_ids, deform_id, deform_obj, np.array(goal_poses)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.stepnum = 0
        self.episode_reward = 0.0
        self.anchors = {}
        reset_bullet(self.args, self.sim, self.cam_on, self.cam_args)

        self.rigid_ids, self.deform_id, self.deform_obj, self.goal_pos = \
            self.load_objects(self.sim, self.args)

        self.sim.stepSimulation()  # step once to get initial state

        if self.args.debug and self.args.viz:
            self.debug_viz_cent_loop()
        #
        # Setup dynamic anchors.
        for i in range(DeformEnv.NUM_ANCHORS):  # make anchors
            anchor_init_pos = self.args.anchor_init_pos if (i % 2) == 0 else \
                self.args.other_anchor_init_pos
            preset_dynamic_anchor_vertices = get_preset_properties(
                DEFORM_INFO, self.deform_obj, 'deform_anchor_vertices')
            _, mesh = get_mesh_data(self.sim, self.deform_id)
            anchor_id, anchor_pos, anchor_vertices = create_anchor(
                self.sim, anchor_init_pos, i,
                preset_dynamic_anchor_vertices, mesh)
            attach_anchor(self.sim, anchor_id, anchor_vertices, self.deform_id)
            self.anchors[anchor_id] = {'pos': anchor_pos,
                                       'vertices': anchor_vertices}
        #
        # Set up viz.
        if self.args.viz:  # loading done, so enable debug rendering if needed
            time.sleep(0.1)  # wait for debug visualizer to catch up
            self.sim.stepSimulation()  # step once to get initial state
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        obs, _ = self.get_obs()
        return obs

    def debug_viz_cent_loop(self):
        # DEBUG visualize true loop center
        if not hasattr(self.args, "deform_true_loop_vertices"): return
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        vv = np.array(vertex_positions)
        for i, true_loop_vertices in enumerate(self.args.deform_true_loop_vertices):
            cent_pos = vv[true_loop_vertices].mean(axis=0)
            alpha = 1 if i == 0 else 0.3  # Primary = solid, secondary = 50% transparent
            create_anchor_geom(self.sim, cent_pos, mass=0.0,
                               rgba=(0, 1, 0.8, alpha), use_collision=False)

    def step(self, action, unscaled_velocity=False):
        # action is num_anchors x 3 for 3D velocity for anchors/grippers;
        # assume action in [-1,1], we convert to [-MAX_ACT_VEL, MAX_ACT_VEL].
        if self.args.debug:
            print('action', action)
        if not unscaled_velocity:
            assert ((np.abs(action) <= 1.0).all()), 'action must be in [-1, 1]'
            action *= DeformEnv.MAX_ACT_VEL
        action = action.reshape(DeformEnv.NUM_ANCHORS, 3)
        # Step through physics simulation.
        for sim_step in range(self.args.sim_steps_per_action):
            for i in range(DeformEnv.NUM_ANCHORS):
                command_anchor_velocity(self.sim, self.anchor_ids[i], action[i])
            self.sim.stepSimulation()
        # Get next obs, reward, done.
        next_obs, done = self.get_obs()
        reward = self.get_reward()
        if done:  # if terminating early use reward from current step for rest
            reward *= (self.n_max_episode_steps - self.stepnum)
        self.episode_reward += reward
        done = (done or self.stepnum >= self.n_max_episode_steps)
        info = {}
        if self.args.debug and self.stepnum % 10 == 0:
            print(f'step {self.stepnum:d} reward {reward:0.4f}')
            if done:
                print(f'episode reward {self.episode_reward:0.4f}')
        self.stepnum += 1

        return next_obs, reward, done, info

    def get_obs(self):
        anc_obs = []
        done = False
        for i in range(DeformEnv.NUM_ANCHORS):
            pos, _ = self.sim.getBasePositionAndOrientation(self.anchor_ids[i])
            linvel, _ = self.sim.getBaseVelocity(self.anchor_ids[i])
            anc_obs.extend(pos)
            anc_obs.extend((np.array(linvel) / DeformEnv.MAX_OBS_VEL).tolist())
        anc_obs = np.array(anc_obs)
        # if (np.abs(anc_obs) > self.anchor_lims).any():
        #     if self.args.debug:
        #         print('clipping anchor obs', anc_obs)
        #     anc_obs = np.clip(anc_obs, -1.0*self.anchor_lims, self.anchor_lims)
        #     done = True
        if self.args.cam_resolution is None:
            obs = anc_obs
        else:
            w, h, rgba_px, _, _ = self.sim.getCameraImage(
                width=self.args.cam_resolution, height=self.args.cam_resolution,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
            obs = rgba_px[:, :, 0:3]
        return obs, done

    def get_reward(self):
        if not hasattr(self.args, 'deform_true_loop_vertices'):
            return 0.0  # not reward info without info about true loops
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        dist = 0
        # A simple solution for cases when there is a mismatch between
        # the number of goals and number of ground truth loops.
        num_holes_to_track = min(
            len(self.args.deform_true_loop_vertices), len(self.goal_pos))
        for i in range(num_holes_to_track):  # loop through goal vertices
            true_loop_vertices = self.args.deform_true_loop_vertices[i]
            goal_pos = self.goal_pos[i]
            vv = np.array(vertex_positions)
            cent_pos = vv[true_loop_vertices].mean(axis=0)
            dist += np.linalg.norm(cent_pos - goal_pos)
        rwd = -1.0 * dist / DeformEnv.WORKSPACE_BOX_SIZE
        return rwd

    def render(self, mode='rgb_array', width=300, height=300,
               dist=11.4, pitch=-22.4, yaw=257, tgt_pos=[-0.08, -0.29, 1.8]):
        #
        # TODO(Yonk): remove hard-coded numbers.
        #
        assert (mode == 'rgb_array')
        if self.args.cam_viewmat is None:
            dist, pitch, yaw, pos_x, pos_y, pos_z = [11.4, -22.4, 257, -0.08, -0.29, 1.8,]
        else:
            dist, pitch, yaw, pos_x, pos_y, pos_z = self.args.cam_viewmat
        cam = {
            'distance': dist,
            'pitch': pitch,
            'yaw': yaw,
            'cameraTargetPosition': [pos_x, pos_y, pos_z],
            'upAxisIndex':2,
            'roll':0,
        }
        view_mat = self.sim.computeViewMatrixFromYawPitchRoll(**cam)
        w, h, rgba_px, _, _ = self.sim.getCameraImage(
            width=width, height=height,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_mat, **CAM_INFO)
        # If getCameraImage() returns a tuple instead of numpy array that
        # means that pybullet was installed without numpy being present.
        # Uninstall pybullet, then install numpy, then install pybullet.
        assert (isinstance(rgba_px, np.ndarray)), 'Install numpy, then pybullet'
        img = rgba_px[:, :, 0:3]
        return img
