#
# Utilities for deform sim in PyBullet.
#
# @contactrika
#
import os
import time

import numpy as np
import pybullet
import pybullet_data

from .mesh_utils import get_mesh_data
from .anchor_utils import create_anchor_geom, pin_fixed
from .task_info import DEFORM_INFO, SCENE_INFO, TASK_INFO


def get_preset_properties(object_preset_dict, deform_obj_name, key):
    if object_preset_dict is None or \
            deform_obj_name not in object_preset_dict.keys():
        return None
    if key in object_preset_dict[deform_obj_name].keys():
        return object_preset_dict[deform_obj_name][key]


def load_objects(sim, scene_name, args):
    if scene_name.startswith('hang'):
        scene_name = 'hang'  # same scene for 'HangBag', 'HangCloth'
    elif scene_name.startswith('mask'):
        scene_name = 'dress'  # same human figure for dress and mask tasks
    elif scene_name.startswith('button'):
        scene_name = 'button'  # same human figure for dress and mask tasks
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
    goal_pos = SCENE_INFO[scene_name]['goal_pos']
    if args.viz:
        create_anchor_geom(sim, goal_pos, mass=0.0, radius=0.01,
                           rgba=(0, 1, 0, 1), use_collision=True)
    pass


def load_rigid_object(sim, obj_file_name, scale, init_pos, init_ori,
                      rgba_color=None):
    """Load a rigid object from file, create visual and collision shapes."""
    if obj_file_name.endswith('.obj'):  # mesh info
        xyz_scale = [scale, scale, scale]
        viz_shape_id = sim.createVisualShape(
            shapeType=pybullet.GEOM_MESH, rgbaColor=rgba_color,
            fileName=obj_file_name, meshScale=xyz_scale)
        col_shape_id = sim.createCollisionShape(
            shapeType=pybullet.GEOM_MESH,
            fileName=obj_file_name, meshScale=xyz_scale)
        rigid_id = sim.createMultiBody(
            baseMass=0.0,  # mass==0 => fixed at the position where it is loaded
            basePosition=init_pos,
            # useMaximalCoordinates=1, # TODO Delete me
            baseCollisionShapeIndex=col_shape_id,
            baseVisualShapeIndex=viz_shape_id,
            baseOrientation=pybullet.getQuaternionFromEuler(init_ori))
    elif obj_file_name.endswith('.urdf'):  # URDF file
        rigid_id = sim.loadURDF(
            obj_file_name, init_pos, pybullet.getQuaternionFromEuler(init_ori),
            useFixedBase=1, globalScaling=scale)
    else:
        print('Unknown file extension', obj_file_name)
        assert(False), 'load_rigid_object supports only obj and URDF files'
    return rigid_id


def load_deform_object(sim, obj_file_name, texture_file_name,
                       scale, init_pos, init_ori,
                       bending_stiffness, damping_stiffness, elastic_stiffness,
                       friction_coeff, debug):
    """Load object from obj file with pybullet's loadSoftBody()."""
    if debug:
        print('Loading filename', obj_file_name)
    # Note: do not set very small mass (e.g. 0.01 causes instabilities).
    deform_id = sim.loadSoftBody(
        mass=1.0,  # 1kg is default; bad sim with lower mass
        fileName=obj_file_name,
        scale=scale,
        basePosition=init_pos,
        baseOrientation=pybullet.getQuaternionFromEuler(init_ori),
        springElasticStiffness=elastic_stiffness,
        springDampingStiffness=damping_stiffness,
        springBendingStiffness=bending_stiffness,
        frictionCoeff=friction_coeff,
        collisionMargin=0.05,  # how far apart do two objects begin interacting
        useSelfCollision=0,
        springDampingAllDirections=1,
        useFaceContact=True,
        useNeoHookean=0,
        useMassSpring=True,
        useBendingSprings=True,
        # repulsionStiffness=10000000,
    )
    # PyBullet examples for loading and anchoring deformables:
    # https://github.com/bulletphysics/bullet3/examples/pybullet/examples/deformable_anchor.py
    sim.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
    texture_id = sim.loadTexture(texture_file_name)
    kwargs = {}
    if hasattr(pybullet, 'VISUAL_SHAPE_DOUBLE_SIDED'):
        kwargs['flags'] = pybullet.VISUAL_SHAPE_DOUBLE_SIDED
    sim.changeVisualShape(
        deform_id, -1, textureUniqueId=texture_id, **kwargs)
    num_mesh_vertices = get_mesh_data(sim, deform_id)[0]

    if debug:
        print('Loaded deform_id', deform_id, 'with',
              num_mesh_vertices, 'mesh vertices', 'init_pos', init_pos)
    # Pybullet will struggle with very large meshes, so we should keep mesh
    # sizes to a limited number of vertices and faces.
    # Large meshes will load on Linux/Ubuntu, but sim will run too slowly.
    # Meshes with >2^13=8196 vertices will fail to load on OS X due to shared
    # memory limits, as noted here:
    # https://github.com/bulletphysics/bullet3/issues/1965
    assert(num_mesh_vertices < 2**13)  # make sure mesh has less than ~8K verts
    return deform_id


def reset_bullet(args, sim, cam_on=False, cam_args={}, debug=False):
    """Reset/initialize pybullet simulation."""
    if args.viz:  # toggle aux menus in the GUI.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, cam_on)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, cam_on)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, cam_on)
        pybullet.configureDebugVisualizer(
            pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, cam_on)
        sim.resetDebugVisualizerCamera(**cam_args)
        if debug:
            res = sim.getDebugVisualizerCamera()
            print('Camera info for', cam_args)
            print('viewMatrix', res[2])
            print('projectionMatrix', res[3])
    # Note: using sim.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
    # would turn on FEM, which could be very tricky to tune, so we avoid it.
    # sim.resetSimulation()
    sim.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
    sim.setGravity(0, 0, args.sim_gravity)
    sim.setTimeStep(1.0/args.sim_freq)
    sim.setPhysicsEngineParameter(
        # numSubSteps=10,
        # allowedCcdPenetration=0.01,
        # erp=0.1,
    )
    # Load floor plane and rigid objects
    sim.setAdditionalSearchPath(pybullet_data.getDataPath())
    floor_id = sim.loadURDF('plane.urdf')
    assert(floor_id == 0)  # camera assumes floor/ground is loaded first
    return sim
