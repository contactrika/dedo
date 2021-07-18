#
# Utilities for deform sim in PyBullet.
#
# @contactrika
#

import numpy as np
np.set_printoptions(precision=2, linewidth=150, threshold=10000, suppress=True)
import pybullet

from ..control.control_util import plan_min_jerk_trajectory


# Anchor properties.
ANCHOR_MASS = 1.0
ANCHOR_RADIUS = 0.015  # 0.03  # default = 0.007, invisible = 0.001


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


def create_anchor(sim, pos, mass=0.0, radius=0.005, rgba=(1,0,1,1.0),
                  use_collision=True, visual_shape_id=None):
    """Create a small visual object at the provided pos in world coordinates.
    If mass==0: this object does not collide with any other objects
    and only serves to show grip location.
    input: sim (pybullet sim), pos (list of 3 coords for anchor in world frame)
    output: anchorId (long) --> unique bullet ID to refer to the anchor object
    """
    if visual_shape_id is None:
        anchorVisualShape = sim.createVisualShape(
            pybullet.GEOM_SPHERE, radius=radius*1.5, rgbaColor=rgba)
    else:
        anchorVisualShape = visual_shape_id
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


def create_trajectory(waypoints, steps_per_waypoint, frequency):
    """Creates a smoothed trajectory through the given waypoints."""
    assert(len(waypoints) == len(steps_per_waypoint))
    num_wpts = len(waypoints)
    tot_steps = sum(steps_per_waypoint[:-1])
    dt = 1.0/frequency
    traj = np.zeros([tot_steps, 3+3])  # 3D pos , 3D vel
    prev_pos = waypoints[0]  # start at the 0th waypoint
    t = 0
    for wpt in range(1, num_wpts):
        tgt_pos = waypoints[wpt]
        dur = steps_per_waypoint[wpt-1]
        if dur == 0:
            continue
        Y, Yd, Ydd = plan_min_jerk_trajectory(prev_pos, tgt_pos, dur*dt, dt)
        traj[t:t+dur,0:3] = Y[:]
        traj[t:t+dur,3:6] = Yd[:]   # vel
        # traj[t:t+dur,6:9] = Ydd[:]  # acc
        t += dur
        prev_pos = tgt_pos
    if t<tot_steps: traj[t:,:] = traj[t-1,:]  # set rest to last entry
    # print('create_trajectory(): traj', traj)
    return traj


def create_anchor_trajs(sim, all_waypoints, steps_per_waypoint,
                        deform_bullet_id, sim_frequency,
                        attach_anchors=True, debug=False):
    """Create anchors that can pull deform objects along trajectories."""
    kwargs = {}
    if hasattr(pybullet, 'MESH_DATA_SIMULATION_MESH'):
        kwargs['flags'] = pybullet.MESH_DATA_SIMULATION_MESH
    _, mesh_vetex_xyzs = sim.getMeshData(deform_bullet_id, **kwargs)
    anchored_vertex_ids = []
    for waypoints in all_waypoints:
        anchored_vertex_ids.append(get_closest(waypoints[0], mesh_vetex_xyzs))
        if debug:
            print('Starting waypoint ', waypoints[0],
                  'anchored at deform vertex ids', anchored_vertex_ids)
    anchor_ids = []; anchor_trajs = []
    pybullet.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
    for waypt_i in range(len(anchored_vertex_ids)):
        anchor_pos = all_waypoints[waypt_i][0]
        print('Load anchor at', anchor_pos)
        anchor_id = create_anchor(sim, anchor_pos, mass=ANCHOR_MASS,
                                  radius=ANCHOR_RADIUS)
        # anchor_id = pybullet.loadURDF(
        #    'anchor.urdf', anchor_pos, useMaximalCoordinates=True)
        if attach_anchors:
            for v in anchored_vertex_ids[waypt_i]:
                print('createSoftBodyAnchor for vertex', v)
                sim.createSoftBodyAnchor(
                    deform_bullet_id, v, anchor_id, -1, anchor_pos)
                # Note: can use createSoftBodyAnchor(deform_bullet_id, v, -1, -1)
                # to keep v in place instead of pinning to another object.
        traj = create_trajectory(
            all_waypoints[waypt_i], steps_per_waypoint, frequency=sim_frequency)
        anchor_ids.append(anchor_id); anchor_trajs.append(traj)

    return anchor_ids, anchor_trajs


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
    sim.resetBaseVelocity(
        anchor_bullet_id, linearVelocity=vel, angularVelocity=[0, 0, 0])


def release_anchors(sim, anchor_bullet_ids, dbg_msg):
    print(dbg_msg)
    for i in range(len(anchor_bullet_ids)):
        sim.removeConstraint(anchor_bullet_ids[i])
        sim.changeVisualShape(
            anchor_bullet_ids[i], -1, rgbaColor=[0.5, 0.5, 0.5, 1.0])
        # For pybullet v==2.6.4
        # sim.changeDynamics(anchor_bullet_ids[i], -1, mass=0.00001)


def get_anchor_poses(sim, anchor_bullet_ids):
    anchor_poses = []
    for anchor_id in anchor_bullet_ids:
        pos, _ = sim.getBasePositionAndOrientation(anchor_id)
        anchor_poses.append(pos)
    return anchor_poses
