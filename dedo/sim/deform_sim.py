import os

import numpy as np
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bclient
from gym_bullet_deform.utils.process_camera import ProcessCamera
from gym_bullet_deform.utils.cloth_objects_dict import (
    CLOTH_OBJECTS_DICT, SCENE_OBJECTS_PRESETS, )

class DeformSim():
    def __init__(self, args, viz=False):
        self.args = args
        self.viz = viz
        self.is_sim_init = False

        # House keeping
        self.main_obj_ids = []
        self.anchor_ids = []
        self.debug = self.args.debug
        self.setup_required_kwargs = []
        self.waypoint_viz = []

    def setup(self, **kwargs):
        self.setup_required_kwargs.extend(['sim_gravity', 'sim_frequency'])
        for kwarg in self.setup_required_kwargs:
            assert kwarg in kwargs, f'{kwarg} is a required setup kwarg'
        self._sim_init(**kwargs)
        if self.viz: self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

    def render_soft(self, soft_body_id, debug=False ):
        return ProcessCamera.render_soft(
            self.sim, soft_body_id, debug=debug)

    def _sim_init(self, **kwargs):
        ''' Setting up the floor and scene of the simulator'''
        sim = self._init_bullet(**kwargs)
        self.is_sim_init = True
        self.sim = sim
        self.build_floor()
        return sim

    def build_floor(self):
        # Load floor plane and rigid objects
        self.sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        floor_id = self.sim.loadURDF('plane.urdf')
        # assert (floor_id == 0)  # camera rendering assumes ground has bullet id 0

    def get_closest(self, pts, vertices, max_dist=None):
        ''' Get the lcosest vertices ids to the points (e.g. anchors)'''
        closest_vertex_ids = []
        pts = np.array(pts)
        vertices = np.array(vertices)
        num_pins_per_anchor = max(1, vertices.shape[0]//50)
        num_to_pin = min(vertices.shape[0], num_pins_per_anchor)
        for i in range(pts.shape[0]):
            dists = np.linalg.norm(vertices - pts[[i],:], axis=1)
            to_pin_ids = np.argpartition(dists, num_to_pin)[0:num_to_pin]
            if max_dist is not None:
                to_pin_ids = to_pin_ids[dists[to_pin_ids] <= max_dist]
            closest_vertex_ids.append(to_pin_ids.tolist())
            #closest_vertex_ids.append(np.argmin(dists))
        #print('num_pins_per_anchor', num_pins_per_anchor)
        #print('closest_vertex_ids', closest_vertex_ids)
        return closest_vertex_ids

    def _init_bullet(self, sim=None, **kwargs,):
        # TODO Finish this method
        if self.viz:
            if sim is None:
                sim = bclient.BulletClient(connection_mode=pybullet.GUI)
            # disable aux menus in the gui
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            # don't render during init
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
            # Keep camera distance and target same as defaults for ProcessCamera,
            # so that debug views look similar to those recorded by the camera pans.
            # Camera pitch and yaw can be set as desired, since they don't
            # affect ProcessCamera panning strategy.
            sim.resetDebugVisualizerCamera(
                cameraDistance=ProcessCamera.CAM_DIST, cameraYaw=70.0,
                cameraPitch=-30, cameraTargetPosition=ProcessCamera.CAM_TGT)
        else:
            if sim is None:
                sim = bclient.BulletClient(connection_mode=pybullet.DIRECT)
        sim.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
        sim.setGravity(0, 0, kwargs['sim_gravity'])
        sim.setTimeStep(1.0 / kwargs['sim_frequency'])
        return sim

    def create_dynamic_anchor(self, pos, mass=0.0, radius=0.005, rgba=(1, 0, 1, 1.0)):
        # Create a small visual object at the provided pos in world coordinates.
        # If mass==0: this object does not collide with any other objects
        # and only serves to show grip location.
        # input: sim (pybullet sim), pos (list of 3 coords for anchor in world frame)
        # output: anchorId (long) --> unique bullet ID to refer to the anchor object
        anchorGeomId = self.create_anchor_geom(pos, mass, radius, rgba)
        self.anchor_ids.append(anchorGeomId)
        return anchorGeomId

    def create_anchor_geom(self, pos, mass=0.0, radius=0.005, rgba=(1, 1, 1, 1.0)):
        # Create a small visual object at the provided pos in world coordinates.
        # If mass==0: this object does not collide with any other objects
        # and only serves to show grip location.
        # input: sim (pybullet sim), pos (list of 3 coords for anchor in world frame)
        # output: anchorId (long) --> unique bullet ID to refer to the anchor object
        anchorVisualShape = self.sim.createVisualShape(
            pybullet.GEOM_SPHERE, radius=radius * 1.5, rgbaColor=rgba)
        if mass > 0:
            anchorCollisionShape = self.sim.createCollisionShape(
                pybullet.GEOM_SPHERE, radius=radius)
        else:
            anchorCollisionShape = -1
        anchorId = self.sim.createMultiBody(baseMass=mass, basePosition=pos,
                                       baseCollisionShapeIndex=anchorCollisionShape,
                                       baseVisualShapeIndex=anchorVisualShape)
        return anchorId

    # def create_trajectory(self, waypoints, steps_per_waypoint, frequency):
    #     # Create a smoothed trajectory through the given waypoints.
    #     assert (len(waypoints) == len(steps_per_waypoint))
    #     num_wpts = len(waypoints)
    #     tot_steps = sum(steps_per_waypoint[:-1])
    #     dt = 1.0 / frequency
    #     traj = np.zeros([tot_steps, 3 + 3])  # 3D pos , 3D vel
    #     prev_pos = waypoints[0]  # start at the 0th waypoint
    #     t = 0
    #     for wpt in range(1, num_wpts):
    #         tgt_pos = waypoints[wpt]
    #         dur = steps_per_waypoint[wpt - 1]
    #         Y, Yd, Ydd = plan_min_jerk_trajectory(prev_pos, tgt_pos, dur * dt, dt)
    #         traj[t:t + dur, 0:3] = Y[:]
    #         traj[t:t + dur, 3:6] = Yd[:]  # vel
    #         # traj[t:t+dur,6:9] = Ydd[:]  # acc
    #         t += dur
    #         prev_pos = tgt_pos
    #     if t < tot_steps: traj[t:, :] = traj[t - 1, :]  # set rest to last entry
    #     # print('create_trajectory(): traj', traj)
    #     return traj

    def load_rigid_object(self, obj_file_name,  basePosition, baseOrientation, globalScaling=1.0, rgbaColor=None):
        assert(obj_file_name.endswith('.obj'))  # assume mesh info
        scaling_mat = [globalScaling,globalScaling,globalScaling]
        viz_shape_id = self.sim.createVisualShape(
            shapeType=pybullet.GEOM_MESH, rgbaColor=rgbaColor,
            fileName=obj_file_name, meshScale=scaling_mat)
        col_shape_id = self.sim.createCollisionShape(
            shapeType=pybullet.GEOM_MESH,
            fileName=obj_file_name, meshScale=scaling_mat)
        rigid_custom_id = self.sim.createMultiBody(
            baseMass=0.0,  # freefloating if baseMass > 0
            basePosition=basePosition,
            baseCollisionShapeIndex=col_shape_id,
            baseVisualShapeIndex=viz_shape_id,
            baseOrientation=pybullet.getQuaternionFromEuler(baseOrientation))
        return rigid_custom_id

    def load_custom_rigid_obj(self):
        # TODO rewire args
        args = self.args
        rigid_obj_path = os.Storingpath.join(args.data_path, args.rigid_custom_obj)
        rigid_id = self.load_rigid_object(
            rigid_obj_path,
            args.rigid_init_pos, args.rigid_init_ori,  args.rigid_scale,)
        rigid_ids.append(rigid_id)

    def load_rigid_urdf(self, urdf_filepath, **kwargs):
        assert (urdf_filepath.endswith('.urdf'))

        if 'baseOrientation' in kwargs:
            kwargs['baseOrientation'] = pybullet.getQuaternionFromEuler([0, 0, np.pi / 2])
        return self.sim.loadURDF(urdf_filepath, useFixedBase=1, **kwargs)


    def load_soft_object(self, obj_file_name, scale, init_pos, init_ori,
                         bending_stiffness, damping_stiffness, elastic_stiffness,
                         friction_coeff, fuzz_stiffness, debug=False, scale_gaussian_noise=0.0):
        # Load cloth from obj file
        # TODO: why does setting mass produce strange behavior?
        # https://github.com/bulletphysics/bullet3/blob/
        # 9ac1dd6194978525ac261f574b8289285318c6c7/examples/SharedMemory/
        # b3RobotSimulatorClientAPI_NoDirect.cpp#L1192
        # TODO: it is possible to set cloth textures/colors?
        if fuzz_stiffness:
            elastic_stiffness += (np.random.rand()-0.5)*2*20
            bending_stiffness += (np.random.rand()-0.5)*2*20
            friction_coeff += (np.random.rand()-0.5)*2*0.3
            if elastic_stiffness < 10.0:
                elastic_stiffness = 10.0
            if bending_stiffness < 10.0:
                bending_stiffness = 10.0

            scale += (np.random.rand() - 0.5) * 2 * scale_gaussian_noise
            scale = np.clip(scale, 0.1, 1.5)
            #toDo: Ioanna, here is where you change stifness
            el_stiff_fact = 1.5
            bend_stiff_fact = 1.1
            print('fuzzed', f'elastic_stiffness {el_stiff_fact*elastic_stiffness:0.4f}',
                  f'bending_stiffness {bend_stiff_fact*bending_stiffness:0.4f}',
                  f'friction_coeff {friction_coeff:0.4f} scale {scale:0.4f}')
        cloth_id = self.sim.loadSoftBody(
            scale=scale,  # mass=0.05,
            fileName=obj_file_name, basePosition=init_pos,
            baseOrientation=pybullet.getQuaternionFromEuler(init_ori),
            collisionMargin=0.002, useMassSpring=1, useBendingSprings=1,
            springElasticStiffness=el_stiff_fact*elastic_stiffness,
            springDampingStiffness=damping_stiffness,
            springBendingStiffness=bend_stiff_fact*bending_stiffness,
            frictionCoeff=friction_coeff, useSelfCollision=1)
        num_mesh_vertices, _ = self.sim.getMeshData(cloth_id)
        if self.debug:
            print('Loading obj_file_name', obj_file_name)
            print('Loaded cloth_id', cloth_id, 'with',
                  num_mesh_vertices, 'mesh vertices')
        # Pybullet will struggle with very large meshes, so we should keep mesh
        # sizes to a limited number of vertices and faces.
        # Large meshes will load on Linux/Ubuntu, but sim will run too slowly.
        # Meshes with >2^13=8196 verstices will fail to load on OS X due to shared
        # memory limits, as noted here:
        # https://github.com/bulletphysics/bullet3/issues/1965
        assert(num_mesh_vertices < 2**13), f'The mesh (n={num_mesh_vertices}) is too big' # make sure mesh has less than ~8K verts
        return cloth_id

    def add_main_object(self, obj_id):
        ''' Add an object as a main object (to be manipulated) '''
        self.main_obj_ids.append(obj_id)


    def setup_scene_rigid_obj(self, scene_version_name=None):
        assert self.is_sim_init, 'simulator not initialize! Call sim_init() first!'
        args = self.args
        self.sim.setAdditionalSearchPath(os.path.join(args.data_path, 'urdf'))
        rigid_ids = []

        # TODO Scene rigid object setup  ===

        if scene_version_name is not None:
            scene_name = scene_version_name.lower()
        else:
            scene_name = list(SCENE_OBJECTS_PRESETS.keys())[args.scene_version]

        self.scene_name = scene_name

        # Load scene entities
        for name, kwargs in SCENE_OBJECTS_PRESETS[scene_name]['entities'].items():
            if name.endswith('.obj'):
                pth = os.path.join(args.data_path,name)
                id = self.load_rigid_object(pth,  **kwargs)
                rigid_ids.append(id)
            elif name.endswith('.urdf'):
                id = self.load_rigid_urdf(name, **kwargs)
                rigid_ids.append(id)

        self.rigid_ids = rigid_ids
        return rigid_ids

    def add_waypoint_viz(self, pos, rgba=[1,0,1, 1]):
        ''' Visualize way points (helps with debugging) '''
        wpVisualShape = self.sim.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=[0.01,0.01,0.01], rgbaColor=rgba)

        wp_obj_id = self.sim.createMultiBody( basePosition=pos,
                                            baseVisualShapeIndex=wpVisualShape)
        self.waypoint_viz.append(wp_obj_id)

    def use_preset_args(self):
        # if not self.args.cloth_obj in CLOTH_OBJECTS_DICT: return
        # Convert from absolute path
        cloth_obj = self.cloth_obj_name
        if cloth_obj in CLOTH_OBJECTS_DICT:
            for k, v in CLOTH_OBJECTS_DICT[cloth_obj].items():
                setattr(self.args, k, v)

        return self.args

    @property
    def cloth_obj_name(self):
        cloth_obj_full_path = self.args.cloth_obj
        parent_dir = os.path.basename(os.path.dirname(cloth_obj_full_path))
        filename = os.path.basename(cloth_obj_full_path)
        cloth_obj = os.path.join(parent_dir, filename)
        return cloth_obj

    @property
    def cloth_anchored_vertex_ids(self):
        if self.cloth_obj_name not in CLOTH_OBJECTS_DICT.keys():
            return None
        if 'cloth_anchored_vertex_ids' in CLOTH_OBJECTS_DICT[self.cloth_obj_name].keys(): #
            return CLOTH_OBJECTS_DICT[self.cloth_obj_name]['cloth_anchored_vertex_ids']
        return None

    @property
    def cloth_fixed_anchor_vertex_ids(self):
        if self.cloth_obj_name not in CLOTH_OBJECTS_DICT.keys():
            return None
        if 'cloth_fixed_anchor_vertex_ids' in CLOTH_OBJECTS_DICT[self.cloth_obj_name].keys(): #
            return CLOTH_OBJECTS_DICT[self.cloth_obj_name]['cloth_fixed_anchor_vertex_ids']
        return None

    def set_cloth_anchored_vertex_ids(self, anchored_vertex_ids):
        if self.cloth_obj_name not in CLOTH_OBJECTS_DICT:
            CLOTH_OBJECTS_DICT[self.cloth_obj_name] = {}
        CLOTH_OBJECTS_DICT[self.cloth_obj_name]['cloth_anchored_vertex_ids'] = anchored_vertex_ids

    def anchors_grasp(self):
        cloth_id = self.main_obj_ids[-1]
        for i, anchor_id in enumerate(self.anchor_ids):
            for v in self.cloth_anchored_vertex_ids[i]:
                self.sim.createSoftBodyAnchor(cloth_id, v, bodyUniqueId=anchor_id)

    def get_closest_anchor(self, anchors):
        soft_obj_id = self.main_obj_ids[-1] # hardcode single soft object only
        mesh_info = self.sim.getMeshData(soft_obj_id)
        mesh_vetex_positions = np.array(mesh_info[1])
        cloth_obj = self.cloth_obj_name
        ret_anchors = []
        if self.cloth_anchored_vertex_ids is not None:
            for anc, anchor_vertex_ids in enumerate(self.cloth_anchored_vertex_ids):
                anc_pos = mesh_vetex_positions[anchor_vertex_ids].mean(axis=0)
                ret_anchors.append(anc_pos)
        else:
            anchor_vert_indices = self.get_closest(
                anchors, mesh_vetex_positions)
            self.set_cloth_anchored_vertex_ids(anchor_vert_indices)
            for anc_pts in anchor_vert_indices:
                anc_pos = mesh_vetex_positions[anc_pts].mean(axis=0)
                ret_anchors.append(anc_pos)
        return ret_anchors

    def set_fixed_anchors(self, anchors=None):
        if anchors is not None:
            raise NotImplementedError('Not done yet')
        soft_obj_id = self.main_obj_ids[-1]
        mesh_info = self.sim.getMeshData(soft_obj_id)
        mesh_vetex_positions = np.array(mesh_info[1])
        if self.cloth_fixed_anchor_vertex_ids is not None:
            for anc, anchor_vertex_ids in enumerate(self.cloth_fixed_anchor_vertex_ids):
                anc_pos = mesh_vetex_positions[anchor_vertex_ids].mean(axis=0)
                anc_id = self.create_anchor_geom(anc_pos)
                for v in anchor_vertex_ids:
                    self.sim.createSoftBodyAnchor(soft_obj_id, v, bodyUniqueId=anc_id)

    def find_closest_anchor_points(self, anchors):
        soft_obj_id = self.main_obj_ids[-1] # hardcode single soft object only
        mesh_info = self.sim.getMeshData(soft_obj_id)
        mesh_vetex_positions = np.array(mesh_info[1])
        ret_anchors = []
        anchor_vert_indices = self.get_closest(
            anchors, mesh_vetex_positions)
        for anc_pts in anchor_vert_indices:
            anc_pos = mesh_vetex_positions[anc_pts].mean(axis=0)
            ret_anchors.append(anc_pos)
        return ret_anchors, anchor_vert_indices

    def get_reward_target_pos(self, scene_version_name, target_type, ):
        scene_name = scene_version_name.lower()
        assert scene_name in SCENE_OBJECTS_PRESETS, f'Unknown scene version name {scene_version_name}'
        if scene_name == 'basic':
            return None
        elif target_type == 'hard':
            return SCENE_OBJECTS_PRESETS[scene_name]['hard_target_pos']
        elif target_type == 'easy':
            return SCENE_OBJECTS_PRESETS[scene_name]['easy_target_pos']

    def setup_camera(self):
        # Set up camera if needed.
        assert self.rigid_ids is not None, 'setup_scene_rigid_obj() must be called first! '
        cam = None;
        output_data_file_pfx = '';
        cam_object_ids = None
        args = self.args
        if args.cam_outdir is not None:
            outdir = os.path.expanduser(args.cam_outdir)
            if not os.path.exists(outdir): os.makedirs(outdir)
            if args.rigid_custom_obj is not None:
                nm = os.path.splitext(os.path.basename(args.rigid_custom_obj))[0]
                sfx = 'rigid_' + nm
            else:
                sfx = 'scene' + str(args.scene_version)
            sfx += '_' + os.path.splitext(os.path.basename(args.cloth_obj))[0]
            output_data_file_pfx = os.path.join(outdir, sfx)
            # cam_object_ids = rigid_ids
            # ProcessCamera.init(args.viz)

        self.output_data_file_pfx = output_data_file_pfx

    # def generate_anchor_traj(self, all_waypoints, steps_per_waypoint, cloth_anchored_vertex_ids ):
    #     # kwargs = {'scene_version': args.scene_version,
    #     #           'waypoints_noise': args.waypoints_noise}
    #     # res = make_all_waypoints_fxn(sim, cloth_id, cloth_obj_path, **kwargs)
    #     # all_waypoints, steps_per_waypoint, cloth_anchored_vertex_ids = res
    #
    #     args = self.args
    #
    #     if args.debug:
    #         print('Anchored at cloth_anchored_vertex_ids', cloth_anchored_vertex_ids)
    #     anchor_ids = [];
    #     trajs = []
    #     cloth_id = self.main_obj_ids[0] # hardcoded for now
    #     for i in range(len(cloth_anchored_vertex_ids)):  # Loop through {left, right} anchors
    #         anchor_id = self.create_dynamic_anchor(
    #             all_waypoints[i][0], mass=0.05, radius=0.007)
    #         for v in cloth_anchored_vertex_ids[i]:
    #             self.sim.createSoftBodyAnchor(cloth_id, v, bodyUniqueId=anchor_id)
    #         traj = self.create_trajectory(
    #             all_waypoints[i], steps_per_waypoint, frequency=args.sim_frequency)
    #         anchor_ids.append(anchor_id);
    #         trajs.append(traj)
    #
    #     self.anchor_ids = anchor_ids
    #     self.trajs = trajs
    #     return anchor_ids, trajs