"""
Utilities for deform sim in PyBullet.


Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika

"""
from pathlib import Path  # automatically converts forward slashes if needed

import numpy as np
import os
import pybullet
import pybullet_data

from .mesh_utils import get_mesh_data


def get_preset_properties(object_preset_dict, deform_obj_name, key):
    if object_preset_dict is None or \
            deform_obj_name not in object_preset_dict.keys():
        return None
    if key in object_preset_dict[deform_obj_name].keys():
        return object_preset_dict[deform_obj_name][key]


def load_rigid_object(sim, obj_file_name, scale, init_pos, init_ori,
                      mass=0.0, texture_file=None, rgba_color=None):
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
            baseMass=mass,  # mass==0 => fixed at position where it is loaded
            basePosition=init_pos,
            baseCollisionShapeIndex=col_shape_id,
            baseVisualShapeIndex=viz_shape_id,
            baseOrientation=pybullet.getQuaternionFromEuler(init_ori))
    elif obj_file_name.endswith('.urdf'):  # URDF file
        rigid_id = sim.loadURDF(
            obj_file_name, init_pos, pybullet.getQuaternionFromEuler(init_ori),
            useFixedBase=True if mass <= 0 else False, globalScaling=scale)
    else:
        print('Unknown file extension', obj_file_name)
        assert(False), 'load_rigid_object supports only obj and URDF files'
    sim.changeDynamics(rigid_id, -1, mass, lateralFriction=1.0,
                       spinningFriction=1.0, rollingFriction=1.0,
                       restitution=0.0)
    n_jt = sim.getNumJoints(rigid_id)

    if texture_file is not None:
        texture_id = sim.loadTexture(texture_file)
        kwargs = {}
        if hasattr(pybullet, 'VISUAL_SHAPE_DOUBLE_SIDED'):
            kwargs['flags'] = pybullet.VISUAL_SHAPE_DOUBLE_SIDED

        if obj_file_name.endswith('figure_headless.urdf'):
            sim.changeVisualShape(  # only changing the body of the figure
                rigid_id, 0, rgbaColor=[1, 1, 1, 1],
                textureUniqueId=texture_id, **kwargs)
        else:
            for i in range(-1, n_jt):
                sim.changeVisualShape(
                    rigid_id, i, rgbaColor=[1,1,1,1],
                    textureUniqueId=texture_id, **kwargs)

    return rigid_id


def load_deform_object(sim, obj_file_name, texture_file_name,
                       scale, init_pos, init_ori,
                       bending_stiffness, damping_stiffness, elastic_stiffness,
                       friction_coeff, self_collision, debug):
    """Load object from obj file with pybullet's loadSoftBody()."""
    if debug:
        print('Loading filename', obj_file_name)
    # Note: do not set very small mass (e.g. 0.01 causes instabilities).
    deform_id = sim.loadSoftBody(
        mass=1,  # 1kg is default; bad sim with lower mass
        fileName=str(Path(obj_file_name)),
        scale=scale,
        basePosition=init_pos,
        baseOrientation=pybullet.getQuaternionFromEuler(init_ori),
        springElasticStiffness=elastic_stiffness,
        springDampingStiffness=damping_stiffness,
        springBendingStiffness=bending_stiffness,
        frictionCoeff=friction_coeff,
        # collisionMargin=0.003,  # how far apart do two objects begin interacting
        useSelfCollision=self_collision,
        springDampingAllDirections=1,
        useFaceContact=True,
        useNeoHookean=0,
        useMassSpring=True,
        useBendingSprings=True,
        # repulsionStiffness=10000000,
    )
    # PyBullet examples for loading and anchoring deformables:
    # github.com/bulletphysics/bullet3/examples/pybullet/examples/deformable_anchor.py
    sim.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
    texture_id = sim.loadTexture(str(Path(texture_file_name)))
    kwargs = {}
    if hasattr(pybullet, 'VISUAL_SHAPE_DOUBLE_SIDED'):
        kwargs['flags'] = pybullet.VISUAL_SHAPE_DOUBLE_SIDED
    sim.changeVisualShape(
        deform_id, -1, rgbaColor=[1,1,1,1], textureUniqueId=texture_id, **kwargs)
    num_mesh_vertices = get_mesh_data(sim, deform_id)[0]

    if debug:
        print('Loaded deform_id', deform_id, 'with',
              num_mesh_vertices, 'mesh vertices', 'init_pos', init_pos)
    # Pybullet will struggle with very large meshes, so we should keep mesh
    # sizes to a limited number of vertices and faces.
    # Large meshes will load on Linux/Ubuntu, but sim will run too slowly.
    # Meshes with >2^13=8196 vertices will fail to load on OS X due to shared
    # memory limits, as noted here:
    # github.com/bulletphysics/bullet3/issues/1965
    assert(num_mesh_vertices < 2**13)  # make sure mesh has less than ~8K verts
    return deform_id


def reset_bullet_legacy(args, sim, plane_texture=None, debug=False):
    """Reset/initialize pybullet simulation."""
    dist, pitch, yaw, pos_x, pos_y, pos_z = args.cam_viewmat
    cam_args = {
            'cameraDistance': dist,
            'cameraPitch': pitch,
            'cameraYaw': yaw,
            'cameraTargetPosition': np.array([pos_x, pos_y, pos_z])
    }
    if args.viz:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, False)
        sim.resetDebugVisualizerCamera(**cam_args)
        if debug:
            res = sim.getDebugVisualizerCamera()
            print('Camera info for', cam_args)
            print('viewMatrix', res[2])
            print('projectionMatrix', res[3])
    sim.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)  # FEM deform sim
    sim.setGravity(0, 0, args.sim_gravity)
    sim.setTimeStep(1.0/args.sim_freq)
    # Could experiment with physic engine parameters, but so far we have not
    # noticed a stark improvement from changing these.
    # sim.setPhysicsEngineParameter(numSubSteps=10, allowedCcdPenetration=0.01)
    #
    # Load floor plane and rigid objects
    #
    sim.setAdditionalSearchPath(pybullet_data.getDataPath())
    floor_id = sim.loadURDF('plane.urdf')
    if plane_texture is not None:
        if debug: print('texture file', plane_texture)
        texture_id = sim.loadTexture(plane_texture)
        sim.changeVisualShape(
            floor_id, -1, rgbaColor=[1,1,1,1], textureUniqueId=texture_id, )
    assert(floor_id == 0)  # camera assumes floor/ground is loaded first
    return sim


def reset_bullet(args, sim, plane_texture=None, debug=False):
    """Reset/initialize pybullet simulation."""
    dist, pitch, yaw, pos_x, pos_y, pos_z = args.cam_viewmat
    cam_args = {
            'cameraDistance': dist,
            'cameraPitch': pitch,
            'cameraYaw': yaw,
            'cameraTargetPosition': np.array([pos_x, pos_y, pos_z])
    }
    if args.viz:
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, False)
        sim.resetDebugVisualizerCamera(**cam_args)
        if debug:
            res = sim.getDebugVisualizerCamera()
            print('Camera info for', cam_args)
            print('viewMatrix', res[2])
            print('projectionMatrix', res[3])
    sim.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)  # FEM deform sim
    sim.setGravity(0, 0, args.sim_gravity)
    sim.setTimeStep(1.0/args.sim_freq)
    sim.setAdditionalSearchPath(pybullet_data.getDataPath())
    # sim.setRealTimeSimulation(1)
    return


def load_deformable(args, sim, deform_obj, data_path='deps/dedo/dedo/data/', debug=False):
    # Load deformable object.
    texture_path = args.deform_texture_file
    deform_id = load_deform_object(
        sim, os.path.join(data_path, deform_obj), os.path.join(data_path, texture_path), args.deform_scale,
        args.deform_init_pos, args.deform_init_ori,
        args.deform_bending_stiffness, args.deform_damping_stiffness,
        args.deform_elastic_stiffness, args.deform_friction_coeff,
        not args.disable_self_collision, debug)

    return deform_id


def load_floor(sim, plane_texture=None, debug=False):
    sim.setAdditionalSearchPath(pybullet_data.getDataPath())
    floor_id = sim.loadURDF('plane.urdf')
    if plane_texture is not None:
        if debug: print('texture file', plane_texture)
        texture_id = sim.loadTexture(plane_texture)
        sim.changeVisualShape(
            floor_id, -1, rgbaColor=[1,1,1,1], textureUniqueId=texture_id, )
    # assert(floor_id == 1)  # camera assumes floor/ground is loaded second, AFTER THE DEFORMABLE
    return sim
