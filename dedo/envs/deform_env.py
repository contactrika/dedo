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
    create_anchor, attach_anchor, create_anchor_geom, command_anchor_velocity, release_anchor,
    pin_fixed)
from ..utils.init_utils import (
    load_deform_object, load_rigid_object, reset_bullet, get_preset_properties)
from ..utils.mesh_utils import get_mesh_data
from ..utils.task_info import DEFAULT_CAM_PROJECTION, DEFORM_INFO, SCENE_INFO, TASK_INFO, DEFAULT_CAM
from ..utils.procedural_utils import gen_procedural_hang_cloth, gen_procedural_button_cloth


class DeformEnv(gym.Env):
    MAX_OBS_VEL = 20.0  # max vel (in m/s) for the anchor observations
    MAX_ACT_VEL = 10.0  # max vel (in m/s) for the anchor actions
    NUM_ANCHORS = 2
    WORKSPACE_BOX_SIZE = 20.0  # workspace box limits (needs to be >=1)
    STEPS_AFTER_DONE = 500 # Release anchor and let free fall to check if task is done correctly
    SUCESS_REWARD_TRESHOLD = 2.5

    def __init__(self, args):
        self.args = args
        self.max_episode_len = args.max_episode_len
        self.cam_on = args.cam_resolution is not None
        # Initialize sim and load objects.
        self.sim = bclient.BulletClient(
            connection_mode=pybullet.GUI if args.viz else pybullet.DIRECT)
        reset_bullet(args, self.sim, debug=args.debug)
        self.rigid_ids, self.deform_id, self.deform_obj, self.goal_pos = \
            self.load_objects(self.sim, self.args)

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
        if self.args.debug:
            print('Created DeformEnv with obs', self.observation_space, 'act',
                  self.action_space)

    @property
    def anchor_ids(self):
        return list(self.anchors)

    def get_texture_path(self, file_path):
        # Get either the prespecified texture file name or a random one, depends on settings
        if self.args.use_random_textures:
            parent = os.path.dirname(file_path)
            full_parent_path = os.path.join(self.args.data_path, parent)
            randfile = np.random.choice(list(os.listdir(full_parent_path)))
            file_path = os.path.join(parent,randfile)
        return file_path

    def load_objects(self, sim, args):
        scene_name = self.args.task.lower()
        # if scene_name.startswith('hang'):
        #     scene_name = 'hang'  # same scene for 'HangBag', 'HangCloth'
        if scene_name.startswith('button'):
            scene_name = 'button'  # same human figure for dress and mask tasks
        elif scene_name.startswith('hangproccloth'):
            scene_name = 'hangcloth'

        # Make v0 the random version
        if args.version == 0:
            args.use_random_textures = True


        data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
        args.data_path = data_path
        sim.setAdditionalSearchPath(data_path)

        # Setup Hangbag task's 100+ objects
        if args.version == 0:
            _do = np.random.choice(TASK_INFO[args.task])
        else:
            _do = TASK_INFO[args.task][args.version-1]
        if args.task == 'HangBag' and _do not in DEFORM_INFO:
            bn = os.path.basename(_do)
            if bn.startswith('bag1'):
                DEFORM_INFO[_do] = DEFORM_INFO['bags/bags_zehang/bag1_0.obj'].copy()
            elif bn.startswith('bag2'):
                DEFORM_INFO[_do] = DEFORM_INFO['bags/bags_zehang/bag2_0.obj'].copy()
            elif bn.startswith('bag3'):
                DEFORM_INFO[_do] = DEFORM_INFO['bags/bags_zehang/bag3_0.obj'].copy()


        if args.override_deform_obj is not None:
            deform_obj = args.override_deform_obj
        else:
            assert (args.task in TASK_INFO)  # already checked in args
            assert (args.version < len(TASK_INFO[args.task]) + 1)  # checked in args
            if args.version == 0:
                deform_obj = np.random.choice(TASK_INFO[args.task])
            else:
                deform_obj = TASK_INFO[args.task][args.version-1]
            # deform_obj = self.TASK_INFO[args.task][args.version]


            for arg_nm, arg_val in DEFORM_INFO[deform_obj].items():
                setattr(args, arg_nm, arg_val)

        # Procedural generation for hanging cloth
        if deform_obj == 'procedural_hang_cloth':
            args.node_density = 15
            if args.version == 1:
                args.num_holes = 1
            elif args.version == 2:
                args.num_holes = 2
            deform_obj = gen_procedural_hang_cloth(self.args, deform_obj, DEFORM_INFO)
            for arg_nm, arg_val in DEFORM_INFO[deform_obj].items():
                setattr(args, arg_nm, arg_val)

        # Procedural generation for buttoninig
        if self.args.task == 'ButtonProc':
            args.num_holes = 2
            args.node_density = 15
            deform_obj, hole_centers = gen_procedural_button_cloth(self.args, deform_obj, DEFORM_INFO)
            for arg_nm, arg_val in DEFORM_INFO[deform_obj].items():
                setattr(args, arg_nm, arg_val)

            h1, h2 = hole_centers

            # Move buttons according to match hole position so the task can complete
            # conversion
            h1 = (-h1[1], 0, h1[2]+2)
            h2 = (-h2[1], 0, h2[2] + 2)

            buttons = SCENE_INFO['button']
            buttons['entities']['urdf/button_fixed.urdf']['basePosition'] = (h1[0], 0.2, h1[2])
            buttons['entities']['urdf/button_fixed2.urdf']['basePosition'] = (h2[0], 0.2, h2[2])

            # goal pos
            buttons['goal_pos'] = [h1, h2]
            # TODO Add noise to the pos of buttons

        #
        # Load rigid objects.
        #
        rigid_ids = []
        for name, kwargs in SCENE_INFO[scene_name]['entities'].items():
            pth = os.path.join(data_path, name)
            rgba_color = kwargs['rgbaColor'] if 'rgbaColor' in kwargs else None
            texture_file = self.get_texture_path(args.rigid_texture_file) if 'useTexture' in kwargs and kwargs[
                'useTexture'] else None
            id = load_rigid_object(
                sim, pth, kwargs['globalScaling'],
                kwargs['basePosition'], kwargs['baseOrientation'], texture_file, rgba_color)
            rigid_ids.append(id)

        #
        # Load deformable object.
        #
        texture_path = os.path.join(
            data_path, self.get_texture_path(args.deform_texture_file))
        deform_id = load_deform_object(
            sim, deform_obj, texture_path, args.deform_scale,
            args.deform_init_pos, args.deform_init_ori,
            args.deform_bending_stiffness, args.deform_damping_stiffness,
            args.deform_elastic_stiffness, args.deform_friction_coeff, not args.disable_self_collision,
            args.debug)
        if scene_name == 'button':  # pin cloth edge for buttoning task
            assert ('deform_fixed_anchor_vertex_ids' in DEFORM_INFO[deform_obj])
            pin_fixed(sim, deform_id,
                      DEFORM_INFO[deform_obj]['deform_fixed_anchor_vertex_ids'])
        #
        # Mark the goal.
        #
        goal_poses = SCENE_INFO[scene_name]['goal_pos']
        if args.viz and args.debug:
            for i, goal_pos in enumerate(goal_poses):
                alpha = 1 if i == 0 else 0.3  # primary vs secondary goal
                print(f'goal_pos{i}', goal_pos)
                create_anchor_geom(sim, goal_pos, mass=0.0,
                                   rgba=(0, 1, 0, alpha), use_collision=False)
        return rigid_ids, deform_id, deform_obj, np.array(goal_poses)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.stepnum = 0
        self.episode_reward = 0.0
        self.anchors = {}
        plane_texture_path = os.path.join(
            self.args.data_path,  self.get_texture_path(self.args.plane_texture_file))
        reset_bullet(self.args, self.sim, plane_texture=plane_texture_path)
        self.rigid_ids, self.deform_id, self.deform_obj, self.goal_pos = \
            self.load_objects(self.sim, self.args)

        # Special case for Procedural Cloth V2 (two holes), reward is based on the closest hole
        if self.args.env == 'HangProcCloth-v2':
            self.goal_pos = np.vstack((self.goal_pos,self.goal_pos))

        self.sim.stepSimulation()  # step once to get initial state
        #
        if self.args.debug and self.args.viz:
            self.debug_viz_cent_loop()



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
        v = np.array(vertex_positions)
        for i, true_loop_vertices in enumerate(self.args.deform_true_loop_vertices):
            cent_pos = v[true_loop_vertices].mean(axis=0)

            alpha = 1 if i == 0 else 0.3  # Primary = solid, secondary = 50% transparent
            print('cent_pos', cent_pos)
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
        self.episode_reward += reward
        done = (done or self.stepnum >= self.max_episode_len)
        info = {}
        if self.args.debug and self.stepnum % 10 == 0:
            print(f'step {self.stepnum:d} reward {reward:0.4f}')
            if done:
                print(f'episode reward {self.episode_reward:0.4f}')

        # Compute final reward by releasing anchor and let fall
        if done:
            # Release anchors
            release_anchor(self.sim, self.anchor_ids[0])
            release_anchor(self.sim, self.anchor_ids[1])
            # if self.args.task.lower() == 'lasso':
            #     self.STEPS_AFTER_DONE *= 2
            for sim_step in range(self.STEPS_AFTER_DONE):
                # For lasso, pull the string at the end to prevent dropping lasso
                if self.args.task.lower() == 'lasso' and sim_step % self.args.sim_steps_per_action == 0:
                    action = [100,100,0] # pull towards the endge
                    for i in range(DeformEnv.NUM_ANCHORS):
                        command_anchor_velocity(self.sim, self.anchor_ids[i], action)
                # extend string
                self.sim.stepSimulation()
            # if terminating early use reward from current step for rest
            reward = self.get_reward() * 50
            # reward *= (self.max_episode_len - self.stepnum)

            info['is_success'] = np.abs(reward) < self.SUCESS_REWARD_TRESHOLD
            print('is sucess',  info['is_success'] )
            print('final reward -- ',reward)

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
        dist = []


        # A simple solution for cases when there is a mismatch between
        # the number of goals and number of ground truth loops.
        num_holes_to_track = min(
            len(self.args.deform_true_loop_vertices), len(self.goal_pos))
        for i in range(num_holes_to_track):  # loop through goal vertices
            true_loop_vertices = self.args.deform_true_loop_vertices[i]
            goal_pos = self.goal_pos[i]
            pts = np.array(vertex_positions)
            cent_pts = pts[true_loop_vertices]
            cent_pts = cent_pts[~np.isnan(cent_pts).any(axis=1)] # Nan guard
            # assert len(cent_pts) > 0, 'no valid center points left after NaN clean up. '
            # assert not np.isnan(cent_pts).any(), 'There are still Nan inside cent pts'
            if len(cent_pts) == 0 or np.isnan(cent_pts).any():
                #return a failure reward immediately
                dist = DeformEnv.WORKSPACE_BOX_SIZE*num_holes_to_track*50

                # save a screenshot for debug
                obs = self.render('rgb_array', 300, 300)
                fpath = f'{self.args.logdir}/nan_{self.args.env}_s{self.stepnum}.npy'
                np.save(fpath, obs)
                break
            cent_pos = cent_pts.mean(axis=0)
            dist.append( np.linalg.norm(cent_pos - goal_pos))

        if self.args.env == 'HangProcCloth-v2':
            dist = np.min(dist)
        else:
            dist = np.mean(dist)
        rwd = -1.0 * dist / DeformEnv.WORKSPACE_BOX_SIZE
        return rwd

    def render(self, mode='rgb_array', width=300, height=300,):
        #
        # TODO(Yonk): remove hard-coded numbers.
        #
        assert (mode == 'rgb_array')
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
            viewMatrix=view_mat, **DEFAULT_CAM_PROJECTION)
        # If getCameraImage() returns a tuple instead of numpy array that
        # means that pybullet was installed without numpy being present.
        # Uninstall pybullet, then install numpy, then install pybullet.
        assert (isinstance(rgba_px, np.ndarray)), 'Install numpy, then pybullet'
        img = rgba_px[:, :, 0:3]
        return img
