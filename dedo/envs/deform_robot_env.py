"""
Environment that derives from DeformEnv and uses robots instead of anchors
for grasping and control. This class is experimental, so might only be
appropriate for expert users.


Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika

"""
import os

import gym
import numpy as np
import pybullet
from math import fabs

from ..utils.bullet_manipulator import BulletManipulator
from ..utils.init_utils import get_preset_properties
from ..utils.mesh_utils import get_mesh_data
from ..utils.task_info import DEFORM_INFO, ROBOT_INFO

from .deform_env import DeformEnv


class DeformRobotEnv(DeformEnv):
    ORI_SIZE = 3 * 2  # 3D position + sin,cos for 3 Euler angles
    FING_DIST = 0.01  # default finger distance

    def __init__(self, args):
        super(DeformRobotEnv, self).__init__(args)
        act_sz = 3
        if self.food_packing:
            act_sz += DeformRobotEnv.ORI_SIZE
        self.action_space = gym.spaces.Box(  # [-1, 1]
            -1.0 * np.ones(self.num_anchors * act_sz),
            np.ones(self.num_anchors * act_sz))
        if self.args.debug:
            print('Wrapped as DeformEnvRobot with act', self.action_space)

    @staticmethod
    def unscale_pos(act, unscaled):
        if unscaled:
            return act
        return act * DeformEnv.WORKSPACE_BOX_SIZE

    def load_objects(self, sim, args, debug):
        # call to load_objects of super class
        res = super(DeformRobotEnv, self).load_objects(sim, args, debug)
        data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
        sim.setAdditionalSearchPath(data_path)

        # robot_info = ROBOT_INFO.get(f'franka{self.num_anchors:d}', None)
        robot_info = ROBOT_INFO.get('fetch', None)

        robot_path = os.path.join(data_path, 'robots',
                                  robot_info['file_name'])
        if debug:
            print('Loading robot from', robot_path)
        if robot_info is None:
            print('This robot is not yet supported:', self.args.robot)
            exit(1)
        self.robot = BulletManipulator(
            sim, robot_path, control_mode='velocity',
            ee_joint_name=robot_info['ee_joint_name'],
            ee_link_name=robot_info['ee_link_name'],
            base_pos=robot_info['base_pos'],
            base_quat=pybullet.getQuaternionFromEuler([0, 0, np.pi]),
            global_scaling=robot_info['global_scaling'],
            use_fixed_base=robot_info['use_fixed_base'],
            rest_arm_qpos=robot_info['rest_arm_qpos'],
            left_ee_joint_name=robot_info.get('left_ee_joint_name', None),
            left_ee_link_name=robot_info.get('left_ee_link_name', None),
            left_fing_link_prefix='panda_hand_l_', left_joint_suffix='_l',
            left_rest_arm_qpos=robot_info.get('left_rest_arm_qpos', None),
            debug=debug)

        return res

    def make_anchors(self):
        # incorrect name: not really making any new anchor
        preset_dynamic_anchor_vertices = get_preset_properties(
            DEFORM_INFO, self.deform_obj, 'deform_anchor_vertices')
        _, mesh = get_mesh_data(self.sim, self.deform_id)
        assert (preset_dynamic_anchor_vertices is not None)
        for i in range(self.num_anchors):  # make anchors 
            anchor_pos = np.array(mesh[preset_dynamic_anchor_vertices[i][0]]) # from the cloth definition (15,170)
            print("location of anchor {} attached to garment before glue :".format(i), i, anchor_pos)
            if not np.isfinite(anchor_pos).all():
                print('anchor_pos not sane:', anchor_pos)
                input('Press enter to exit')
                exit(1)

            link_id = self.robot.info.ee_link_id if i == 0 else \
                self.robot.info.left_ee_link_id
 
            # create a glue between the robot at given link position 
            # with the deform object at given vertex position
            # THUS THERE IS NO CONCEPT OF GRASPING
            # Already fixed - so now when end effector moves, the garment will move
            # self.sim.createSoftBodyAnchor(
            #     self.deform_id, preset_dynamic_anchor_vertices[i][0],
            #     self.robot.info.robot_id, link_id)

    def do_action(self, action, unscaled=False, final_wp = None):
        # Note: action is in [-1,1], so we unscale pos (ori is sin,cos so ok).
        action = action.reshape(self.num_anchors, -1)
        ee_pos, ee_ori, _, _ = self.robot.get_ee_pos_ori_vel()
        tgt_pos = DeformRobotEnv.unscale_pos(action[0, :3], unscaled)

        print('ee_pos',ee_pos)
        print('tgt_pos',tgt_pos)

        if final_wp is not None:
            final_wp = final_wp.reshape(self.num_anchors, -1)
            final_pos = DeformRobotEnv.unscale_pos(final_wp[0, :3], unscaled)

        # Calculate distance of robot base from final endpoint
        # Assumption: Way-points (if multiple) are progressively further away from the bot
        pos_rel_base, quat_rel_base = self.robot.get_relative_pose(pos = final_pos)

        # base_state = self.sim.getLinkState(
        #     self.robot.info.robot_id, 0, computeLinkVelocity=0)
        # base_pos = base_state[0]
        # print(np.linalg.norm(pos_rel_base[:2]), self.robot.in_range, base_pos[:2])

        print("error:", np.linalg.norm(ee_pos - final_pos))
        print("pos_rel_base", np.linalg.norm(pos_rel_base[0:2]))

        ################### FOR MOVEMENT OF BASE ##########################

        # If constraint created
        if self.robot.base_cid: 

            # StopGap solution: due to fixed final difference in ee_pos and final_pos
            error_th = 0.75

            # Empirical value to bring robot within reach of target
            robot_reach = 10.0 
            # Note: If IK solver is ideal - we would know that tgt pos is unreachable w/o
            # moving robot if joint angles returned are out of range

            if np.linalg.norm(pos_rel_base[0:2]) < robot_reach:
                self.robot.in_range = 1
            elif np.linalg.norm(pos_rel_base[0:2]) > robot_reach and self.robot.in_range is None:
                self.robot.move_base(tgt_pos)
                if (np.linalg.norm(ee_pos - final_pos) < error_th):
                    self.robot.in_range = 1

        ####################################################################
    
        tgt_ee_ori = ee_ori if action.shape[-1] == 3 else action[0, 3:]
        tgt_kwargs = {'ee_pos': tgt_pos, 'ee_ori': tgt_ee_ori,
                      'fing_dist': DeformRobotEnv.FING_DIST}

        if self.num_anchors > 1:  # dual-arm
            res = self.robot.get_ee_pos_ori_vel(left=True)
            left_ee_pos, left_ee_ori = res[0], res[1]
            left_tgt_pos = DeformRobotEnv.unscale_pos(action[1, :3], unscaled)
            left_tgt_ee_ori = left_ee_ori if action.shape[-1] == 3 else \
                action[1, 3:]
            tgt_kwargs.update({'left_ee_pos': left_tgt_pos,
                               'left_ee_ori': left_tgt_ee_ori,
                               'left_fing_dist': DeformRobotEnv.FING_DIST})
        # so far we have a dict of tgt ee_pos for both arms 
        # now do inverse kinematics
        tgt_qpos = self.robot.ee_pos_to_qpos(**tgt_kwargs)
        # tgt_qpos = self.robot.get_qpos()
        print("tgt_pose", tgt_qpos)
        print("current_pose", self.robot.get_qpos())
        n_slack = 1  # use > 1 if robot has trouble reaching the pose
        sub_i = 0
        ee_th = 0.01
        diff = np.abs(self.robot.get_qpos() - tgt_qpos)

        while (diff > ee_th).any():
            self.robot.move_to_qpos(
                tgt_qpos, mode=pybullet.POSITION_CONTROL, kp=0.05, kd=7.0) # 0.05, 7.0
            self.sim.stepSimulation()
            diff = np.abs(self.robot.get_qpos() - tgt_qpos)
            sub_i += 1
            if sub_i >= n_slack:
                diff = np.zeros_like(diff)  # set while loop to done

    def make_final_steps(self):
        ee_pos, ee_ori, *_ = self.robot.get_ee_pos_ori_vel()
        final_action = np.hstack([ee_pos, ee_ori]).reshape(1, -1)
        if self.num_anchors > 1:  # dual-arm
            left_ee_pos, left_ee_ori, *_ = \
                self.robot.get_ee_pos_ori_vel(left=True)
            final_left_action = np.hstack(
                [left_ee_pos, left_ee_ori]).reshape(1, -1)
            final_action = np.vstack([final_action, final_left_action])
        if self.args.debug:
            print('final_action', final_action)
        info = {'final_obs': []}
        for sim_step in range(DeformEnv.STEPS_AFTER_DONE):
            self.do_action(final_action, unscaled=True, final_wp = final_action)
            self.sim.stepSimulation()
            if sim_step % self.args.sim_steps_per_action == 0:
                next_obs, _ = self.get_obs()
                info['final_obs'].append(next_obs)
        return info

    def get_grip_obs(self):
        grip_obs = []
        ee_pos, _, ee_linvel, _ = self.robot.get_ee_pos_ori_vel()
        grip_obs.extend(ee_pos)
        grip_obs.extend((np.array(ee_linvel) / DeformEnv.MAX_OBS_VEL))
        if self.num_anchors > 1:  # EE pos, vel of left arm
            left_ee_pos, _, left_ee_linvel, _ = \
                self.robot.get_ee_pos_ori_vel(left=True)
            grip_obs.extend(left_ee_pos)
            grip_obs.extend((np.array(left_ee_linvel) / DeformEnv.MAX_OBS_VEL))

        return grip_obs

    def get_reward(self):
        if self.food_packing:
            return self.get_food_packing_reward()
        else:
            return super(DeformRobotEnv, self).get_reward()

    def get_food_packing_reward(self):
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        # rigid_ids[1] is the box, rigid_ids[2] is the can
        box_pos, _ = self.sim.getBasePositionAndOrientation(self.rigid_ids[1])
        can_pos, _ = self.sim.getBasePositionAndOrientation(self.rigid_ids[2])
        vertex_cent = np.mean(vertex_positions, axis=0)
        dist1 = np.linalg.norm(vertex_cent - box_pos)
        dist2 = np.linalg.norm(vertex_cent - can_pos)

        dist = np.mean([dist1, dist2])
        rwd = -1.0 * dist / DeformEnv.WORKSPACE_BOX_SIZE

        # Squish penalty (to protect the fruit)
        vertices = np.array(vertex_positions)
        relative_dist = np.linalg.norm(vertices - vertices[[0]], axis=1)

        current_shape = relative_dist[self.deform_shape_sample_idx]
        penalty_rwd = np.linalg.norm(current_shape - self.deform_init_shape)
        rwd = rwd + penalty_rwd
        return rwd
