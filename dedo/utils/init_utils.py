#
# Utilities for deform sim in PyBullet.
#
# @contactrika
#
import os

import numpy as np
np.set_printoptions(precision=2, linewidth=150, threshold=10000, suppress=True)
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bclient

from ..utils.bullet_manipulator import BulletManipulator

from .multi_camera import MultiCamera


def load_rigid_object(sim, obj_file_name, scale, init_pos, init_ori):
    """Load a rigid object from file, create visual and collision shapes."""
    assert(obj_file_name.endswith('.obj'))  # assume mesh info
    viz_shape_id = sim.createVisualShape(
        shapeType=pybullet.GEOM_MESH, rgbaColor=None,
        fileName=obj_file_name, meshScale=scale)
    col_shape_id = sim.createCollisionShape(
        shapeType=pybullet.GEOM_MESH,
        fileName=obj_file_name, meshScale=scale)
    rigid_custom_id = sim.createMultiBody(
        baseMass=0.0,  # mass==0 => fixed at the position where it is loaded
        basePosition=init_pos,
        baseCollisionShapeIndex=col_shape_id,
        baseVisualShapeIndex=viz_shape_id,
        baseOrientation=pybullet.getQuaternionFromEuler(init_ori))
    return rigid_custom_id


def create_spheres(id, radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]],
                   visual=True, collision=True, rgba=[0, 1, 1, 1]):
    """
    Reference: https://github.com/Healthcare-Robotics/assistive-gym/blob/
    41d7059f0df0976381b2544b6bcfc2b84e1be008/assistive_gym/envs/base_env.py#L127
    """
    sphere_collision = -1
    if collision:
        sphere_collision = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_SPHERE, radius=radius, physicsClientId=id)

    sphere_visual = pybullet.createVisualShape(
        shapeType=pybullet.GEOM_SPHERE, radius=radius,
        rgbaColor=rgba, physicsClientId=id) if visual else -1

    last_sphere_id = pybullet.createMultiBody(
        baseMass=mass, baseCollisionShapeIndex=sphere_collision,
        baseVisualShapeIndex=sphere_visual, basePosition=[0, 0, 0],
        useMaximalCoordinates=False, batchPositions=batch_positions,
        physicsClientId=id)

    spheres = []
    for body in list(range(last_sphere_id[-1]-len(batch_positions)+1,
                           last_sphere_id[-1]+1)):
        # sphere = Agent()
        # sphere.init(body, id, self.np_random, indices=-1)
        spheres.append(body)
    return spheres


def load_soft_object(sim, obj_file_name, scale, init_pos, init_ori,
                     bending_stiffness, damping_stiffness, elastic_stiffness,
                     friction_coeff, mass=1.0, collision_margin=0.002,
                     fuzz_stiffness=False, debug=False):
    """Load object from obj file with pybullet's loadSoftBody()."""
    if fuzz_stiffness:
        elastic_stiffness += (np.random.rand()-0.5)*2*20
        bending_stiffness += (np.random.rand()-0.5)*2*20
        friction_coeff += (np.random.rand()-0.5)*2*0.3
        scale += (np.random.rand()-0.5)*2*0.2
        if elastic_stiffness < 10.0:
            elastic_stiffness = 10.0
        if bending_stiffness < 10.0:
            bending_stiffness = 10.0
        scale = np.clip(scale, 0.6, 1.5)
        print('fuzzed', f'elastic_stiffness {elastic_stiffness:0.4f}',
              f'bending_stiffness {bending_stiffness:0.4f}',
              f'friction_coeff {friction_coeff:0.4f} scale {scale:0.4f}')
    # Note: do not set very small mass (e.g. 0.01 causes instabilities).
    deform_id = sim.loadSoftBody(
        scale=scale, mass=mass,
        fileName=obj_file_name, basePosition=init_pos,
        baseOrientation=pybullet.getQuaternionFromEuler(init_ori),
        collisionMargin=collision_margin,
        springElasticStiffness=elastic_stiffness,
        springDampingStiffness=damping_stiffness,
        springBendingStiffness=bending_stiffness,
        frictionCoeff=friction_coeff, useSelfCollision=1,
        useNeoHookean=0, useMassSpring=1, useBendingSprings=1
    )
    kwargs = {}
    if hasattr(pybullet, 'MESH_DATA_SIMULATION_MESH'):
        kwargs['flags'] = pybullet.MESH_DATA_SIMULATION_MESH
    num_mesh_vertices, _ = sim.getMeshData(deform_id, **kwargs)
    if debug:
        print('Loaded deform_id', deform_id, 'with',
              num_mesh_vertices, 'mesh vertices')
    # Pybullet will struggle with very large meshes, so we should keep mesh
    # sizes to a limited number of vertices and faces.
    # Large meshes will load on Linux/Ubuntu, but sim will run too slowly.
    # Meshes with >2^13=8196 verstices will fail to load on OS X due to shared
    # memory limits, as noted here:
    # https://github.com/bulletphysics/bullet3/issues/1965
    assert(num_mesh_vertices < 2**13)  # make sure mesh has less than ~8K verts
    return deform_id


def init_bullet_franka_robot(args, robot_desc_file, control_mode,
                             rest_qpos=None, cam_on=False):
    """Initialize pybullet simulation."""
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(curr_dir)
    args.data_path = os.path.join(parent_dir, 'data')
    # by default, the robot uses the franka_cup.urdf file
    robot_desc_path = os.path.join(os.path.split(__file__)[0], '..',
                                   'data', robot_desc_file)
    robot = BulletManipulator(
        robot_desc_path,
        control_mode=control_mode,
        ee_joint_name='panda_joint7', ee_link_name='panda_hand',
        base_pos=[0,0,0], rest_arm_qpos=rest_qpos,
        dt=1.0/args.sim_frequency,
        visualize=args.viz, cam_dist=0.8, cam_yaw=80, cam_pitch=-15,
        cam_target=(0.3, 0, 0.3))
    sim = robot.sim
    sim.setGravity(0, 0, args.sim_gravity)
    sim.setTimeStep(1.0/args.sim_frequency)

    # Load floor plane and rigid objects
    sim.setAdditionalSearchPath(pybullet_data.getDataPath())
    #floor_id = sim.loadURDF('plane.urdf')

    #assert(floor_id==0)  # camera rendering assumes ground has bullet id 0
    sim.setAdditionalSearchPath(os.path.join(args.data_path))
    
    # texture_id = sim.loadTexture(os.path.join(args.data_path, 'planes',
    #                                           'orange_pattern.png'))
    # pybullet.changeVisualShape(0, -1, rgbaColor=[1, 1, 1, 1],
    #                            textureUniqueId=texture_id)

    if args.viz:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, cam_on)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, cam_on)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, cam_on)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, cam_on)

    return sim, robot


def init_bullet(args, sim=None, cam_on=False, cam_configs={}):
    """Initialize pybullet simulation."""
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(curr_dir)
    args.data_path = os.path.join(parent_dir, 'data')
    if args.viz:
        if sim is None:
            sim = bclient.BulletClient(connection_mode=pybullet.GUI)
        # toggle aux menus in the gui
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, cam_on)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, cam_on)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, cam_on)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, cam_on)
        # don't render during init
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        # Keep camera distance and target same as defaults for MultiCamera,
        # so that debug views look similar to those recorded by the camera pans.
        # Camera pitch and yaw can be set as desired, since they don't
        # affect MultiCamera panning strategy.
        cam_args = {
            'cameraDistance': 0.85,
            'cameraYaw': -30,
            'cameraPitch': -70,
            'cameraTargetPosition': np.array([0.35, 0, 0])
        }
        cam_args.update(cam_configs)
        sim.resetDebugVisualizerCamera(**cam_args)
    else:
        if sim is None:
            sim = bclient.BulletClient(connection_mode=pybullet.DIRECT)
    sim.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
    sim.setGravity(0, 0, args.sim_gravity)
    sim.setTimeStep(1.0/args.sim_frequency)
    return sim


def init_task_sim(sim, args):
    """Init pybullet simulator and load objects based on paths in args."""
    init_bullet(args, cam_on=args.cam_outdir is not None, sim=sim)
    #

    """
    # Load ground.
        self.sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._aux_sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = self.sim.loadURDF('plane.urdf', [0,0,0])
        self._aux_sim.loadURDF('plane.urdf', [0,0,0])
        # Note: changing ground color doesn't work even for plane_transparent.urdf
        # pybullet.changeVisualShape(self.plane_id, -1, rgbaColor=[1, 1, 1, 1])


    """
    # Load floor plane and rigid objects
    sim.setAdditionalSearchPath(pybullet_data.getDataPath())
    floor_id = sim.loadURDF('plane.urdf')

    assert(floor_id==0)  # camera rendering assumes ground has bullet id 0
    sim.setAdditionalSearchPath(os.path.join(args.data_path))

    if args.floor_texture_fname is not None:
        fnm = os.path.join(args.data_path, 'planes', args.floor_texture_fname)
        print('loading', fnm)
        texture_id = sim.loadTexture(fnm)
        pybullet.changeVisualShape(floor_id, -1, rgbaColor=[1, 1, 1, 1],
                                   textureUniqueId=texture_id)

    # TODO: generalize to load multiple rigid and deformable objects.
    rigid_ids = []
    #
    # Load rigid objects.
    if args.rigid_custom_obj is not None:
        rigid_obj_path = os.path.join(args.data_path, args.rigid_custom_obj)
        rigid_id = load_rigid_object(
            sim, rigid_obj_path, args.rigid_scale,
            args.rigid_init_pos, args.rigid_init_ori)
        rigid_ids.append(rigid_id)
    #
    # Load deform objects.
    deform_ids = []
    if args.deform_obj is not None:
        deform_obj_path = os.path.join(args.data_path, args.deform_obj)
        if args.debug: print('Loading deform_obj_path', deform_obj_path)
        deform_id = load_soft_object(
            sim, deform_obj_path, args.deform_scale,
            args.deform_init_pos, args.deform_init_ori,
            args.deform_bending_stiffness, args.deform_damping_stiffness,
            args.deform_elastic_stiffness, args.deform_friction_coeff,
            fuzz_stiffness=False, debug=args.debug)
        deform_ids.append(deform_id)
    if args.viz:  # loading done, turn on visualizer if needed
        sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

    return rigid_ids, deform_ids
