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
import pybullet_utils.bullet_client as bclient

from .mesh_utils import get_mesh_data


def load_rigid_object(sim, obj_file_name, scale, init_pos, init_ori):
    """Load a rigid object from file, create visual and collision shapes."""
    if obj_file_name.endswith('.obj'):  # mesh info
        viz_shape_id = sim.createVisualShape(
            shapeType=pybullet.GEOM_MESH, rgbaColor=None,
            fileName=obj_file_name, meshScale=scale)
        col_shape_id = sim.createCollisionShape(
            shapeType=pybullet.GEOM_MESH,
            fileName=obj_file_name, meshScale=scale)
        rigid_id = sim.createMultiBody(
            baseMass=0.0,  # mass==0 => fixed at the position where it is loaded
            basePosition=init_pos,
            baseCollisionShapeIndex=col_shape_id,
            baseVisualShapeIndex=viz_shape_id,
            baseOrientation=pybullet.getQuaternionFromEuler(init_ori))
    elif obj_file_name.endswith('.urdf'):  # URDF file
        rigid_id = sim.loadURDF(
                os.path.join('urdf', 'torso.urdf'),
                [0.3, 0.0, 0.15], useFixedBase=1, globalScaling=scale)
    else:
        print('Unknown file extension', obj_file_name)
        assert(False), 'load_rigid_object supports only obj and URDF files'
    return rigid_id


def load_soft_object(sim, obj_file_name, texture_file_name,
                     scale, init_pos, init_ori,
                     bending_stiffness, damping_stiffness, elastic_stiffness,
                     friction_coeff, mass=1.0, collision_margin=0.002,
                     fuzz_stiffness=False, debug=True):
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
    texture_id = sim.loadTexture(texture_file_name)
    sim.changeVisualShape(
        deform_id, -1, flags=pybullet.VISUAL_SHAPE_DOUBLE_SIDED,
        textureUniqueId=texture_id)
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


def init_bullet(args, sim=None, cam_on=False, cam_args={}):
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
        sim.resetDebugVisualizerCamera(**cam_args)
    else:
        if sim is None:
            sim = bclient.BulletClient(connection_mode=pybullet.DIRECT)
    # Note: using sim.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
    # would turn on FEM, which could be very tricky to tune, so we avoid it.
    sim.resetSimulation()
    sim.setGravity(0, 0, args.sim_gravity)
    sim.setTimeStep(1.0/args.sim_frequency)
    return sim