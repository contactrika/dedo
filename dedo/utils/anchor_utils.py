#
# Utilities for deform sim in PyBullet.
#
# @contactrika
#
import numpy as np
import pybullet

from .mesh_utils import get_mesh_data

ANCHOR_MIN_DIST = 0.01  # 1cm
ANCHOR_MASS = 0.100     # 100g
ANCHOR_RADIUS = 0.007   # 7mm
ANCHOR_RGBA_ACTIVE = (1, 0, 1, 1)  # magenta
ANCHOR_RGBA_INACTIVE = (0.5, 0.5, 0.5, 1)  # gray


def get_closest(point, vertices, max_dist=None):
    """Find mesh points closest to the given point."""
    point = np.array(point).reshape(1, -1)
    vertices = np.array(vertices)
    num_pins_per_pt = max(1, vertices.shape[0]//50)
    num_to_pin = min(vertices.shape[0], num_pins_per_pt)
    dists = np.linalg.norm(vertices - point, axis=1)
    closest_ids = np.argpartition(dists, num_to_pin)[0:num_to_pin]
    if max_dist is not None:
        closest_ids = closest_ids[dists[closest_ids] <= max_dist]
    return closest_ids


def create_anchor(sim, pos, mass=ANCHOR_MASS, radius=ANCHOR_RADIUS,
                  rgba=ANCHOR_RGBA_INACTIVE, use_collision=True):
    """Create a small visual object at the provided pos in world coordinates.
    If mass==0: this object does not collide with any other objects
    and only serves to show grip location.
    input: sim (pybullet sim), pos (list of 3 coords for anchor in world frame)
    output: anchorId (long) --> unique bullet ID to refer to the anchor object
    """
    anchorVisualShape = sim.createVisualShape(
        pybullet.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    if mass > 0 and use_collision:
        anchorCollisionShape = sim.createCollisionShape(
        pybullet.GEOM_SPHERE, radius=radius)
    else:
        anchorCollisionShape = -1
    anchorId = sim.createMultiBody(baseMass=mass, basePosition=pos,
        baseCollisionShapeIndex=anchorCollisionShape,
        baseVisualShapeIndex=anchorVisualShape,
        useMaximalCoordinates=True)
    return anchorId


def command_anchor_velocity(sim, anchor_bullet_id, vel):
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
    sim.resetBaseVelocity(anchor_bullet_id, linearVelocity=vel.tolist(),
                          angularVelocity=[0, 0, 0])


def attach_anchor(sim, anchor_id, deform_id):
    sim.changeVisualShape(
        anchor_id, -1, rgbaColor=ANCHOR_RGBA_ACTIVE)
    pos, ori = sim.getBasePositionAndOrientation(anchor_id)
    deform_anchored_vertex_ids = get_closest(
        [pos], get_mesh_data(sim, deform_id)[1], max_dist=ANCHOR_MIN_DIST)
    for v in deform_anchored_vertex_ids:
        sim.createSoftBodyAnchor(deform_id, v, anchor_id, -1)


def release_anchor(sim, anchor_id):
    sim.removeConstraint(anchor_id)
    sim.changeVisualShape(anchor_id, -1, rgbaColor=ANCHOR_RGBA_INACTIVE)


def get_anchor_poses(sim, anchor_bullet_ids):
    anchor_poses = []
    for anchor_id in anchor_bullet_ids:
        pos, _ = sim.getBasePositionAndOrientation(anchor_id)
        anchor_poses.append(pos)
    return anchor_poses
