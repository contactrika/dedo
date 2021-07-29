#
# Utilities for deform sim in PyBullet.
#
# @contactrika
#
import numpy as np
import pybullet

from .mesh_utils import get_mesh_data

ANCHOR_MIN_DIST = 0.02  # 2cm
ANCHOR_MASS = 0.100  # 100g
ANCHOR_RADIUS = 0.007  # 7mm
ANCHOR_RGBA_ACTIVE = (1, 0, 1, 1)  # magenta
ANCHOR_RGBA_INACTIVE = (0.5, 0.5, 0.5, 1)  # gray
ANCHOR_RGBA_PEACH = (0.9, 0.75, 0.65, 1)  # peach
# Gains and limits for a simple controller for the anchors.
CTRL_MAX_FORCE = 500 # 10
CTRL_PD_KD = 50.0 # 50


def get_closest(init_pos, mesh, max_dist=None):
    """Find mesh points closest to the given point."""

    init_pos = np.array(init_pos).reshape(1, -1)
    mesh = np.array(mesh)
    # num_pins_per_pt = max(1, mesh.shape[0] // 50)
    # num_to_pin = min(mesh.shape[0], num_pins_per_pt)
    num_to_pin = 1  # new pybullet behaves well with 1 vertex per anchor
    dists = np.linalg.norm(mesh - init_pos, axis=1)
    anchor_vertices = np.argpartition(dists, num_to_pin)[0:num_to_pin]
    if max_dist is not None:
        anchor_vertices = anchor_vertices[dists[anchor_vertices] <= max_dist]
    new_anc_pos = mesh[anchor_vertices].mean(axis=0)
    return new_anc_pos, anchor_vertices


def create_anchor_geom(sim, pos, mass=ANCHOR_MASS, radius=ANCHOR_RADIUS,
                       rgba=ANCHOR_RGBA_INACTIVE, use_collision=True):
    """Create a small visual object at the provided pos in world coordinates.
    If mass==0: the anchor will be fixed (not moving)
    If use_collision==False: this object does not collide with any other objects
    and would only serve to show grip location.
    input: sim (pybullet sim), pos (list of 3 coords for anchor in world frame)
    output: anchorId (long) --> unique bullet ID to refer to the anchor object
    """
    anchor_visual_shape = sim.createVisualShape(
        pybullet.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    if mass > 0 and use_collision:
        anchor_collision_shape = sim.createCollisionShape(
            pybullet.GEOM_SPHERE, radius=radius)
    else:
        anchor_collision_shape = -1
    anchor_id = sim.createMultiBody(
        baseMass=mass, basePosition=pos,
        baseCollisionShapeIndex=anchor_collision_shape,
        baseVisualShapeIndex=anchor_visual_shape,
        useMaximalCoordinates=True)
    return anchor_id


def create_anchor(sim, anchor_pos, anchor_idx, preset_vertices, mesh,
                  mass=0.1, radius=0.005, rgba=(1, 0, 1, 1.0),
                  use_preset=True, use_closest=True):
    """
    Create an anchor in Pybullet to grab or pin an object.
    :param sim: The simulator object
    :param anchor_pos: initial anchor position
    :param anchor_idx: index of the anchor (0:left, 1:right ...)
    :param preset_vertices: a preset list of vertices for the anchors
                            to grab on to (if use_preset is enabled)
    :param mesh: mesh of the deform object
    :param mass: mass of the anchor
    :param radius: visual radius of the anchor object
    :param rgba: color of the anchor
    :param use_preset: Use preset of anchor vertices
    :param use_closest: Use closest vertices to anchor as grabbing vertices
           (if no preset is used), ensuring anchors
    has something to grab on to
    :return: Anchor's ID, anchor's position, anchor's vertices
    """
    anchor_vertices = None
    mesh = np.array(mesh)
    if use_preset and preset_vertices is not None:
        anchor_vertices = preset_vertices[anchor_idx]
        anchor_pos = mesh[anchor_vertices].mean(axis=0)
    elif use_closest:
        anchor_pos, anchor_vertices = get_closest(anchor_pos, mesh)

    anchor_geom_id = create_anchor_geom(sim, anchor_pos, mass, radius, rgba)
    return anchor_geom_id, anchor_pos, anchor_vertices


def target_pos_to_velocity_controller(sim, anchor_bullet_id, tgt_pos, t):
    anc_pos, anc_or = sim.getBasePositionAndOrientation(anchor_bullet_id) # getting position
    anc_linvel, anc_angvel = sim.getBaseVelocity(anchor_bullet_id) # getting velocity
    print('t=',t )
    pos_diff = np.array(tgt_pos) - np.array(anc_pos)
    print('d_pos', pos_diff)
    tgt_vel = (pos_diff) / t
    return tgt_vel


def command_anchor_velocity(sim, anchor_bullet_id, tgt_vel):
    print('tgt_vel', tgt_vel)
    anc_linvel, _ = sim.getBaseVelocity(anchor_bullet_id)
    vel_diff = tgt_vel - np.array(anc_linvel)
    print('anc_linvel', anc_linvel)
    print('vel_diff', vel_diff)
    force = CTRL_PD_KD * vel_diff
    print('force', force)
    force = np.clip(force, -1.0 * CTRL_MAX_FORCE, CTRL_MAX_FORCE)
    sim.applyExternalForce(
        anchor_bullet_id, -1, force.tolist(), [0, 0, 0], pybullet.LINK_FRAME)
    # If we were using a robot (e.g. Yumi or other robot with precise
    # non-compliant velocity control interface) - then we could simply command
    # that velocity to the robot. For a free-floating anchor - one option would
    # be to use PD control and applyExternalForce(). However, it is likely that
    # PD gains would need to be tuned for various objects (different mass,
    # stiffness, etc). So to simplify things we use a reset here. This should
    # be ok for use cases when anchors are mostly free to move.
    # For cases where the anchors are very much constrained by the cloth
    # (e.g. deformable is attached to a fixed object on multiple sides) -
    # other control methods would be more appropriate.
    # sim.resetBaseVelocity(anchor_bullet_id, linearVelocity=tgt_vel.tolist(),
    #                       angularVelocity=[0, 0, 0])


def attach_anchor(sim, anchor_id, anchor_vertices, deform_id, change_color=False):
    if change_color:
        sim.changeVisualShape(
            anchor_id, -1, rgbaColor=ANCHOR_RGBA_ACTIVE)
    for v in anchor_vertices:
        sim.createSoftBodyAnchor(deform_id, v, anchor_id, -1)


def release_anchor(sim, anchor_id):
    sim.removeConstraint(anchor_id)
    sim.changeVisualShape(anchor_id, -1, rgbaColor=ANCHOR_RGBA_INACTIVE)


def pin_fixed(sim, deform_id, vert_ids):
    _, v_pos_list = get_mesh_data(sim, deform_id)
    for v_idx in vert_ids:
        v_pos = v_pos_list[v_idx]
        anc_id = create_anchor_geom(sim, v_pos, mass=0.0, radius=0.002,
                                    rgba=ANCHOR_RGBA_PEACH)
        sim.createSoftBodyAnchor(deform_id, v_idx, anc_id, -1)
