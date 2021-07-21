import os

import numpy as np
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bclient

class DeformSim():
    def __init__(self, args,  object_preset_dict=None, scene_preset_dict=None, viz=False, ):
        self.args = args
        self.viz = viz
        self.sim = None


        # House keeping
        self.deform_obj_id = None

        # self.anchor_ids = []
        self.debug = self.args.debug
        self.setup_required_kwargs = []
        self.waypoint_viz = []

        # Presets for scene and deform objects
        self.object_preset_dict = {} if object_preset_dict is None else object_preset_dict
        self.scene_preset_dict = {} if scene_preset_dict is None else scene_preset_dict


        self.anchors = {} # list of all anchors (both dynamic and fixed)
        self.dynamic_anchors = {} # keeping track of dynamic anchors
        self.fixed_anchors = {} # keeping track of fixed anchors


    def viz_render(self):
        '''Renders a visual GUI'''
        self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

    def stepSimulation(self):
        self.sim.stepSimulation()

    def render_soft(self, soft_body_id, debug=False ):
        return ProcessCamera.render_soft(
            self.sim, soft_body_id, debug=debug)

    def find_closest_vertices(self, pts, vertices=None, max_dist=None, cam_args={}):
        ''' Get the lcosest vertices ids to the points (e.g. anchors)'''
        closest_vertex_ids = []
        pts = np.array(pts)
        vertices = np.array(vertices)
        num_pins_per_anchor = max(1, vertices.shape[0]//50)
        num_to_pin = min(vertices.shape[0], num_pins_per_anchor)
        dists = np.linalg.norm(vertices - pts, axis=1)
        to_pin_ids = np.argpartition(dists, num_to_pin)[0:num_to_pin]
        if max_dist is not None:
            to_pin_ids = to_pin_ids[dists[to_pin_ids] <= max_dist]

            #closest_vertex_ids.append(np.argmin(dists))
        #print('num_pins_per_anchor', num_pins_per_anchor)
        #print('closest_vertex_ids', closest_vertex_ids)
        return to_pin_ids.tolist()

    def init_bullet(self, cam_on=False, cam_args={}, ):

        curr_dir = os.path.dirname(os.path.realpath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        self.args.data_path = os.path.join(parent_dir, 'data')
        sim = self.sim
        if self.viz:
            if self.sim is None:
                sim = bclient.BulletClient(connection_mode=pybullet.GUI)
            # disable aux menus in the gui
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, cam_on)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, cam_on)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, cam_on)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, cam_on)
            # don't render during init
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, cam_on)
            # Keep camera distance and target same as defaults for ProcessCamera,
            # so that debug views look similar to those recorded by the camera pans.
            # Camera pitch and yaw can be set as desired, since they don't
            # affect ProcessCamera panning strategy.
            sim.resetDebugVisualizerCamera(**cam_args)
        else:
            if self.sim is None:
                sim = bclient.BulletClient(connection_mode=pybullet.DIRECT)

        sim.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
        sim.setGravity(0, 0, self.args.sim_gravity)
        sim.setTimeStep(1.0 / self.args.sim_frequency)

        sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        floor_id = sim.loadURDF('plane.urdf')

        self.sim = sim
        return sim

    @property
    def deform_mesh(self):
        assert self.deform_obj_id is not None, "deform object hasn't been loaded yet!"
        soft_obj_id = self.deform_obj_id # hardcode single soft object only
        kwargs = {}
        if hasattr(pybullet, 'MESH_DATA_SIMULATION_MESH'):
            kwargs['flags'] = pybullet.MESH_DATA_SIMULATION_MESH
        mesh_info = self.sim.getMeshData(soft_obj_id, **kwargs)
        mesh_vetex_positions = np.array(mesh_info[1])
        return mesh_vetex_positions


    #   ==================================
    #   ========   Anchor stuff  =========
    #   ==================================

    def create_dynamic_anchor(self,  anchor_pos, dynamic_anchor_idx, mass=0.0, radius=0.005, rgba=(1, 0, 1, 1.0), use_preset=True, use_closest=True):
        # Create a small visual object at the provided pos in world coordinates.
        # If mass==0: this object does not collide with any other objects
        # and only serves to show grip location.
        anchor_vertices = None
        if use_preset and self.preset_dynamic_anchor_vertices is not None:
            anchor_vertices = self.preset_dynamic_anchor_vertices[dynamic_anchor_idx]
            anchor_pos = self.deform_mesh[anchor_vertices].mean(axis=0)

        elif use_closest:
            anchor_pos, anchor_vertices = self.find_closest_anchor_pos(anchor_pos)


        anchorGeomId = self.create_anchor_geometry(anchor_pos, mass, radius, rgba)
        self.anchors[anchorGeomId] = {'anchor_pos': anchor_pos, 'anchor_vertices': anchor_vertices}
        self.dynamic_anchors[anchorGeomId] = {'anchor_pos': anchor_pos, 'anchor_vertices': anchor_vertices}
        return anchorGeomId

    def create_fixed_anchor(self, anchor_pos, fixed_anchor_idx, mass=0.0, radius=0.005, rgba=(1, 0, 1, 1.0), use_preset=True,
                             use_closest=True):
        # Create a small visual object at the provided pos in world coordinates.
        # If mass==0: this object does not collide with any other objects
        # and only serves to show grip location.
        anchor_vertices = None
        if use_preset and self.preset_fixed_anchor_vertices is not None:
            anchor_vertices = self.preset_fixed_anchor_vertices[fixed_anchor_idx]
            anchor_pos = self.deform_mesh[anchor_vertices].mean(axis=0)

        elif use_closest:
            anchor_pos, anchor_vertices = self.find_closest_anchor_pos(anchor_pos)

        anchorGeomId = self.create_anchor_geometry(anchor_pos, mass, radius, rgba)
        self.anchors[anchorGeomId] = {'anchor_pos': anchor_pos, 'anchor_vertices': anchor_vertices}
        self.fixed_anchors[anchorGeomId] = {'anchor_pos': anchor_pos, 'anchor_vertices': anchor_vertices}
        return anchorGeomId

    def create_anchor_geometry(self, pos, mass=0.0, radius=0.005, rgba=(1, 1, 1, 1.0)):
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

    @property
    def preset_dynamic_anchor_vertices(self):
        ''' Gets the corresponding vertices'''
        if self.object_preset_dict is None or self.deform_obj_name not in self.object_preset_dict.keys():
            return None
        if 'cloth_anchored_vertex_ids' in self.object_preset_dict[self.deform_obj_name].keys(): #
            return self.object_preset_dict[self.deform_obj_name]['cloth_anchored_vertex_ids']
        return None

    @property
    def preset_fixed_anchor_vertices(self):
        if self.object_preset_dict is None or self.deform_obj_name not in self.object_preset_dict.keys():
            return None
        if 'cloth_fixed_anchor_vertex_ids' in self.object_preset_dict[self.deform_obj_name].keys(): #
            return self.object_preset_dict[self.deform_obj_name]['cloth_fixed_anchor_vertex_ids']
        return None

    def find_closest_anchor_pos(self, init_pos):
        ''' Find closest vertices for an anchor'''
        mesh_vetex_positions = self.deform_mesh
        anchor_vert_indices = self.find_closest_vertices(init_pos, mesh_vetex_positions)
        new_anc_pos = mesh_vetex_positions[anchor_vert_indices].mean(axis=0)
        return new_anc_pos, anchor_vert_indices

    def anchor_grasp(self, anchor_id):
        self.sim.changeVisualShape(
            anchor_id, -1, rgbaColor=(1, 0, 1, 1))
        v = self.anchors[anchor_id]['anchor_vertices'][0]
        self.sim.createSoftBodyAnchor(self.deform_obj_id, v, anchor_id, -1)


    #
    #
    # def dynamic_anchor_grasp(self, anchor_id, anchor_entity_id):
    #     ''' Grasp onto deform obj after both have been initialized'''
    #     cloth_id = self.deform_obj_ids[-1]
    #     for v in self.preset_dynamic_anchor_vertices[anchor_id]:
    #         self.sim.createSoftBodyAnchor(cloth_id, v, bodyUniqueId=anchor_entity_id)
    #
    # #   ==== Fixed anchor stuff
    # def fixed_anchor_grasp(self, anchors=None):
    #     ''' Set and grasp anchors '''
    #     if anchors is not None:
    #         raise NotImplementedError('Not done yet')
    #     soft_obj_id = self.deform_obj_ids
    #     mesh_vetex_positions = self.deform_mesh
    #
    #     if self.preset_fixed_anchor_vertices is not None:
    #         for anc, anchor_vertex_ids in enumerate(self.preset_fixed_anchor_vertices):
    #             anc_pos = mesh_vetex_positions[anchor_vertex_ids].mean(axis=0)
    #             anc_id = self.create_anchor_geom(anc_pos)
    #             for v in anchor_vertex_ids:
    #                 self.sim.createSoftBodyAnchor(soft_obj_id, v, bodyUniqueId=anc_id)

    # def set_cloth_anchored_vertex_ids(self, anchored_vertex_ids):
    #     if self.cloth_obj_name not in CLOTH_OBJECTS_DICT:
    #         CLOTH_OBJECTS_DICT[self.cloth_obj_name] = {}
    #     CLOTH_OBJECTS_DICT[self.cloth_obj_name]['cloth_anchored_vertex_ids'] = anchored_vertex_ids

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

    def load_rigid_urdf(self, urdf_filepath, **kwargs):
        assert (urdf_filepath.endswith('.urdf'))

        if 'baseOrientation' in kwargs:
            kwargs['baseOrientation'] = pybullet.getQuaternionFromEuler([0, 0, np.pi / 2])
        return self.sim.loadURDF(urdf_filepath, useFixedBase=1, **kwargs)


    def load_deform_object(self, obj_file_name, texture_file_name, scale, init_pos, init_ori,
                         bending_stiffness, damping_stiffness, elastic_stiffness,
                         friction_coeff, fuzz_stiffness, debug=False, scale_gaussian_noise=0.0):
        # Load cloth from obj file
        # TODO: why does setting mass produce strange behavior?
        # https://github.com/bulletphysics/bullet3/blob/
        # 9ac1dd6194978525ac261f574b8289285318c6c7/examples/SharedMemory/
        # b3RobotSimulatorClientAPI_NoDirect.cpp#L1192

        # Note: do not set very small mass (e.g. 0.01 causes instabilities).
        deform_id = self.sim.loadSoftBody(
            mass=2.0,
            fileName=obj_file_name,
            scale=scale,
            basePosition=init_pos,
            baseOrientation=pybullet.getQuaternionFromEuler(init_ori),
            springElasticStiffness=elastic_stiffness,
            springDampingStiffness=damping_stiffness,
            springBendingStiffness=bending_stiffness,
            frictionCoeff=friction_coeff,
            collisionMargin=0.05,
            useSelfCollision=1,
            springDampingAllDirections=1,
            useFaceContact=1,
            useNeoHookean=0,
            useMassSpring=1,
            useBendingSprings=1,
        )

        # Loading texture
        texture_id = self.sim.loadTexture(texture_file_name)
        kwargs = {}
        if hasattr(pybullet, 'VISUAL_SHAPE_DOUBLE_SIDED'):
            kwargs['flags'] = pybullet.VISUAL_SHAPE_DOUBLE_SIDED
        self.sim.changeVisualShape(
            deform_id, -1, textureUniqueId=texture_id, **kwargs)

        num_mesh_vertices, _ = self.sim.getMeshData(deform_id)
        if self.debug:
            print('Loading obj_file_name', obj_file_name)
            print('Loaded cloth_id', deform_id, 'with',
                  num_mesh_vertices, 'mesh vertices')
        # Pybullet will struggle with very large meshes, so we should keep mesh
        # sizes to a limited number of vertices and faces.
        # Large meshes will load on Linux/Ubuntu, but sim will run too slowly.
        # Meshes with >2^13=8196 verstices will fail to load on OS X due to shared
        # memory limits, as noted here:
        # https://github.com/bulletphysics/bullet3/issues/1965
        assert(num_mesh_vertices < 2**13), f'The mesh (n={num_mesh_vertices}) is too big' # make sure mesh has less than ~8K verts

        self.deform_obj_id = deform_id

        return deform_id


    def load_scene(self, scene_version_name=None):
        # assert self.is_sim_init, 'simulator not initialize! Call sim_init() first!'
        args = self.args
        self.sim.setAdditionalSearchPath(os.path.join(args.data_path, 'urdf'))
        rigid_ids = []

        # TODO Scene rigid object setup  ===

        if scene_version_name is not None:
            # Gym env
            scene_name = scene_version_name.lower()
        else:
            # Collection script
            scene_name = list(self.scene_preset_dict.keys())[args.scene_version]

        self.scene_name = scene_name

        # Load scene entities
        for name, kwargs in self.scene_preset_dict[scene_name]['entities'].items():
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
        cloth_obj = self.deform_obj_name
        if cloth_obj in self.object_preset_dict:
            for k, v in self.object_preset_dict[cloth_obj].items():
                setattr(self.args, k, v)

        return self.args

    @property
    def deform_obj_name(self):
        cloth_obj_full_path = self.args.deform_obj
        parent_dir = os.path.basename(os.path.dirname(cloth_obj_full_path))
        filename = os.path.basename(cloth_obj_full_path)
        deform_obj = os.path.join(parent_dir, filename)
        return deform_obj


    def get_reward_target_pos(self, scene_version_name, target_type, ):
        scene_name = scene_version_name.lower()
        assert scene_name in self.scene_preset_dict, f'Unknown scene version name {scene_version_name}'
        if scene_name == 'basic':
            return None
        elif target_type == 'hard':
            return self.scene_preset_dict[scene_name]['hard_target_pos']
        elif target_type == 'easy':
            return self.scene_preset_dict[scene_name]['easy_target_pos']

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