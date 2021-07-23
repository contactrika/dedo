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
    attach_anchor, create_anchor, command_anchor_velocity, pin_fixed)
from ..utils.init_utils import (
    load_deform_object, load_rigid_object, reset_bullet)
from ..utils.mesh_utils import get_mesh_data
from ..utils.task_info import DEFORM_INFO, SCENE_INFO, TASK_INFO


class DeformEnv(gym.Env):
    MAX_OBS_VEL = 20.0  # max vel (in m/s) for the anchor observations
    MAX_ACT_VEL = 10.0  # max vel (in m/s) for the anchor actions
    NUM_ANCHORS = 2
    WORKSPACE_BOX_SIZE = 2.0  # workspace box limits (needs to be >=1)

    def __init__(self, args):
        self.args = args
        self.max_episode_len = args.max_episode_len
        self.cam_on = args.cam_resolution is not None
        self.cam_args = {
            'cameraDistance': 1.2,
            'cameraYaw': 140,
            'cameraPitch': -40,
            'cameraTargetPosition': np.array([0.0, 0, 0])
        }
        # Initialize sim and load objects.
        self.sim = bclient.BulletClient(
            connection_mode=pybullet.GUI if args.viz else pybullet.DIRECT)
        reset_bullet(args, self.sim, self.cam_on, self.cam_args)
        self.rigid_ids, self.deform_id, self.goal_pos = self.load_objects(
            self.sim, self.args)
        # Define sizes of observation and action spaces.
        self.anchor_lims = np.hstack(  # 3D pos and 3D linvel/MAX_OBS_VEL
            [DeformEnv.WORKSPACE_BOX_SIZE*np.ones(DeformEnv.NUM_ANCHORS*3),
             np.ones(DeformEnv.NUM_ANCHORS*3)])
        if args.cam_resolution is None:
            self.observation_space = gym.spaces.Box(
                -1.0*self.anchor_lims, self.anchor_lims)
        else:  # RGB
            self.observation_space = gym.spaces.Box(
                np.zeros((args.cam_resolution, args.cam_resolution, 3)),
                np.ones((args.cam_resolution, args.cam_resolution, 3)))
        self.action_space = gym.spaces.Box(  # [-1,1]
            -1.0*np.ones(DeformEnv.NUM_ANCHORS*3),
            np.ones(DeformEnv.NUM_ANCHORS*3))
        # Loading done, turn on visualizer if needed
        if self.args.viz:
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

    def load_objects(self, sim, args):
        scene_name = self.args.task.lower()
        if scene_name.startswith('hang'):
            scene_name = 'hang'  # same scene for 'HangBag', 'HangCloth'
        elif scene_name.startswith('mask'):
            scene_name = 'dress'  # same human figure for dress and mask tasks
        data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
        sim.setAdditionalSearchPath(data_path)
        #
        # Load rigid objects.
        #
        rigid_ids = []
        for name, kwargs in SCENE_INFO[scene_name]['entities'].items():
            pth = os.path.join(data_path, name)
            id = load_rigid_object(
                sim, pth, kwargs['globalScaling'],
                kwargs['basePosition'], kwargs['baseOrientation'])
            rigid_ids.append(id)
        #
        # Load deformable object.
        #
        if args.override_deform_obj is not None:
            deform_obj = args.override_deform_obj
        else:
            assert(args.task in TASK_INFO)  # already checked in args
            assert(args.version < len(TASK_INFO[args.task]))  # checked in args
            deform_obj = TASK_INFO[args.task][args.version]
            for arg_nm, arg_val in DEFORM_INFO[deform_obj].items():
                setattr(args, arg_nm, arg_val)
        texture_path = os.path.join(
            data_path, 'textures', 'blue_bright.png')
        deform_id = load_deform_object(
            sim, deform_obj,  texture_path, args.deform_scale,
            args.deform_init_pos, args.deform_init_ori,
            args.deform_bending_stiffness, args.deform_damping_stiffness,
            args.deform_elastic_stiffness, args.deform_friction_coeff,
            args.debug)
        if args.task == 'Button':  # pin cloth edge for buttoning task
            assert('deform_fixed_anchor_vertex_ids' in DEFORM_INFO[deform_obj])
            pin_fixed(sim, deform_id,
                      DEFORM_INFO[deform_obj]['deform_fixed_anchor_vertex_ids'])
        #
        # Mark the goal.
        #
        goal_pos = SCENE_INFO[scene_name]['goal_pos']
        if args.viz:
            create_anchor(sim, goal_pos, mass=0.0, radius=0.01,
                          rgba=(0,1,0,1), use_collision=True)

        return rigid_ids, deform_id, np.array(goal_pos)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.stepnum = 0
        self.episode_reward = 0.0
        self.anchor_ids = []
        self.topo_generators = []
        reset_bullet(self.args, self.sim, self.cam_on, self.cam_args)
        self.rigid_ids, self.deform_id, self.goal_pos = self.load_objects(
            self.sim, self.args)
        # Setup dynamic anchors.
        for i in range(DeformEnv.NUM_ANCHORS):  # make anchors
            anchor_init_pos = self.args.anchor_init_pos if (i%2)==0 else \
                self.args.other_anchor_init_pos
            anchor_id = create_anchor(self.sim, anchor_init_pos)
            attach_anchor(self.sim, anchor_id, self.deform_id)
            self.anchor_ids.append(anchor_id)
        # Set up viz.
        if self.args.viz:  # loading done, so enable debug rendering if needed
            time.sleep(0.1)  # wait for debug visualizer to catch up
            self.sim.stepSimulation()  # step once to get initial state
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        obs, _ = self.get_obs()
        return obs

    def step(self, action):
        # action is num_anchors x 3 for 3D velocity for anchors/grippers;
        # assume action in [-1,1], we convert to [-MAX_ACT_VEL, MAX_ACT_VEL].
        print('action', action)
        assert((np.abs(action) <= 1.0).all())
        action = action.reshape(DeformEnv.NUM_ANCHORS, 3)*DeformEnv.MAX_ACT_VEL
        for i in range(DeformEnv.NUM_ANCHORS):
            command_anchor_velocity(self.sim, self.anchor_ids[i], action[i])
        self.sim.stepSimulation()
        next_obs, done = self.get_obs()
        reward = self.get_reward(action)
        self.episode_reward += reward
        done = (done or self.stepnum >= self.max_episode_len)
        info = {}
        if (self.args.debug or self.args.viz) and self.stepnum%10==0:
            print(f'step {self.stepnum:d} reward {reward:0.4f}')
            if done: print(f'episode reward {self.episode_reward:0.4f}')
        self.stepnum += 1

        return next_obs, reward, done, info

    def get_obs(self):
        anc_obs = []
        done = False
        for i in range(DeformEnv.NUM_ANCHORS):
            pos, _ = self.sim.getBasePositionAndOrientation(self.anchor_ids[i])
            linvel, _ = self.sim.getBaseVelocity(self.anchor_ids[i])
            anc_obs.extend(pos)
            anc_obs.extend((np.array(linvel)/DeformEnv.MAX_OBS_VEL).tolist())
        anc_obs = np.array(anc_obs)
        if (np.abs(anc_obs) > self.anchor_lims).any():
            if self.args.debug:
                print('clipping anchor obs', anc_obs)
            anc_obs = np.clip(anc_obs, -1.0*self.anchor_lims, self.anchor_lims)
            done = True
        if self.args.cam_resolution is None:
            obs = anc_obs
        else:
            w, h, rgba_px, _, _ = self.sim.getCameraImage(
                width=self.args.cam_resolution,
                height=self.args.cam_resolution,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
            obs = rgba_px[:,:,0:3]
        return obs, done

    def get_reward(self, action):
        if not hasattr(self.args, 'deform_true_loop_vertices'):
            return 0.0  # not reward info without info about true loops
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        accum = np.zeros(3)
        true_loop_vertices = self.args.deform_true_loop_vertices[0]
        for v in true_loop_vertices:
            accum += np.array(vertex_positions[v])
        loop_centroid = accum/len(true_loop_vertices)
        dist = np.linalg.norm(loop_centroid-self.goal_pos)
        rwd = -1.0*dist
        return rwd
