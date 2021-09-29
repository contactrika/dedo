#
# Env for Franka Emika pushing/collecting deformable YCB fruit and food
# packages as well as rigid YCB objects into a target area.
#
# This task is set to address the challenges that arise in the food handling
# and packing industry, where the objective is to pack the food and other items
# into boxes or bags while avoiding high deformations that can damage the items.
# This task is also applicable to a similar household version of packing food
# items for storage.
#
# @contactrika
#
#
import os
import numpy as np
import pybullet
from typing import Union

from bulb.envs.rearrange_env import RearrangeEnv
from ..utils.bullet_manipulator import BulletManipulator


class FrankaCollectBulletEnv(RearrangeEnv):
    MIN_MULT = 0.10   # 10% of original param values (e.g. mass or friction)
    MAX_MULT = 2.00   # 200% of original
    OBJECT_X = np.array([-0.3, -0.1, -0.3, -0.1])
    OBJECT_Y = np.array([0.10, -0.08, -0.08, 0.12])
    TARGET_POS = np.array([-0.27, -0.27, 0.02])
    TARGET_RGB = np.array([1, 0, 0])
    MAX_EE_Z = 0.3  # stop the episode if end effector gets too high

    def __init__(self,
                 obs_resolution: Union[int, None],
                 debug: bool = False,
                 visualize: bool = False):
        # Initialize robot and base env.
        cam_dist = 0.9; cam_yaw = 20; cam_pitch = -45; cam_tgt=(-0.3, 0, 0)
        robot = BulletManipulator(
            sim, os.path.join('franka_robot', 'franka_small_fingers.urdf'),
            control_mode='torque',
            ee_joint_name='panda_joint7', ee_link_name='panda_hand',
            base_pos=[-0.6,0,0],
            base_quat=pybullet.getQuaternionFromEuler([0, 0, 0]),
            global_scaling=1.0,
            rest_arm_qpos=[-0.0988, 0.719, 0.0948, -1.9623, -0.0733, 2.6373, 0.758],
            kp=([200.0]*7 + [1.0]*2), kd=([2.0]*7 + [0.1]*2),
            min_z=0.0)
        super(FrankaCollectBulletEnv, self).__init__(
            version=2, variant='Ycb', robot=robot, rnd_init_pos=False,
            obs_resolution=obs_resolution, obs_ptcloud=False, debug=debug)
        self._max_episode_steps = 1000  # override base with longer episodes
        data_folder = os.path.join(os.path.split(__file__)[0], '..', 'data')
        self._brick_id = self.robot.sim.loadURDF(
            os.path.join(data_folder, 'brick.urdf'), [-0.15, -0.28, 0.035],
            useFixedBase=True)
        self._ee_jid = self.robot.info.joint_names.tolist().index('panda_joint7')
        assert(self._ee_jid is not None)
        if visualize:
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
            self.robot.sim.resetDebugVisualizerCamera(  # was: cam dist=0.37
                cameraDistance=cam_dist, cameraYaw=cam_yaw,
                cameraPitch=cam_pitch, cameraTargetPosition=cam_tgt)
            dbg_len = 0.05
            tgt = np.copy(FrankaCollectBulletEnv.TARGET_POS)
            self.robot.sim.addUserDebugLine(
                tgt+np.array([-dbg_len,0,0]).tolist(),
                tgt+np.array([dbg_len,0,0]).tolist(),
                FrankaCollectBulletEnv.TARGET_RGB)  # tgt x axis
            self.robot.sim.addUserDebugLine(
                tgt+np.array([0,-dbg_len,0]).tolist(),
                tgt+np.array([0,dbg_len,0]).tolist(),
                FrankaCollectBulletEnv.TARGET_RGB)  # tgt y axis
        self._distr_obj_id = None
        self._init_dists = None

    @property
    def sim_params(self):
        return self._sim_params

    def reset(self):
        self.reset_aggregators()
        obs = super(FrankaCollectBulletEnv, self).reset()
        for i, obj_id in enumerate(self._object_ids):
            res = self.robot.sim.getBasePositionAndOrientation(obj_id)
            obj_pos = list(res[0]); obj_quat = list(res[1])
            obj_pos[0] = FrankaCollectBulletEnv.OBJECT_X[i]
            obj_pos[1] = FrankaCollectBulletEnv.OBJECT_Y[i]
            self.robot.sim.resetBasePositionAndOrientation(
                obj_id, obj_pos, obj_quat)
        self._init_dists = self.compute_dists()
        self.resample_sim_params()
        return obs

    def step(self, action):
        # Assume: robot is torque controlled and action is scaled to [0,1]
        torque = np.hstack(
            [action, np.zeros(self._max_torque.shape[0] - self._ndof)])
        torque = np.clip((torque-0.5)*2*self._max_torque,
                         -self._max_torque, self._max_torque)
        # Apply torque action to joints
        self.robot.apply_joint_torque(torque)  # advances sim inside
        # The last join is very jittery. Set its vel to 0.
        jpos = self.robot.get_qpos()[self._ee_jid]
        self.robot.reset_joint(self._ee_jid, jpos, 0.0)
        # Finish the step.
        next_obs = self.compute_obs()
        ee_ok = self.in_workspace()
        ee_pos = self.robot.get_ee_pos()
        if ee_pos[2] > FrankaCollectBulletEnv.MAX_EE_Z: ee_ok = False
        # Report reward starts and other info.
        rwd = self.compute_reward()
        done = not ee_ok
        done, info = self.update_aggregators(rwd, done)
        if self._debug:
            if self._stepnum%10==0:
                print('step', self._stepnum, 'action', action, 'torque', torque)
                if self._obs_resolution is None: print('state', next_obs)
            if done:
                print(f'rwd {rwd:.4f} tot_rwd {self._episode_rwd:.4f}')
        return next_obs, rwd, done, info

    def resample_sim_params(self, override_smpl=None):
        mn = FrankaCollectBulletEnv.MIN_MULT
        mx = FrankaCollectBulletEnv.MAX_MULT
        self._sim_params = self._distr.sample(mn, mx, override_smpl)
        mult = self._sim_params[0]  # the mass multiplier
        # Change the mass of each object.
        for i, obj_id in enumerate(self._object_ids):
            nm = self._object_names[i]
            orig = 0.2 if '_can' in nm else 0.05 if '_box' in nm else 0.1
            self.robot.sim.changeDynamics(
                self._object_ids[i], -1, mass=mult*orig)
            if self._debug:
                print(f'smpl {mult:0.4f} mass of {nm:s} <- {mult*orig:0.4f}')

    def compute_dists(self):
        dists = []
        for i, obj_id in enumerate(self._object_ids):
            obj_pos, _ = self.robot.sim.getBasePositionAndOrientation(obj_id)
            tgt_pos = FrankaCollectBulletEnv.TARGET_POS
            dist = np.linalg.norm(np.array(obj_pos[:2]) - np.array(tgt_pos[:2]))
            dists.append(min(dist,1.0))   # max distance is 1 meter
        return np.array(dists)

    def compute_reward(self):
        curr_dists = self.compute_dists()
        diff = (self._init_dists - curr_dists)*100  # in cm
        rwd = diff.mean()
        return rwd

    def reset_ee_pos(self):
        ee_pos, ee_quat, _,  _ = self.robot.get_ee_pos_ori_vel()
        qpos = self.robot._ee_pos_to_qpos_raw(np.array([0,0,0.1]), ee_quat)
        print('qpos', qpos)
        self.robot.reset_to_qpos(qpos)
        self.robot.refresh_viz()
        input('cont')
