#
# Utilities for following trajectories in PyBullet.
#
# @contactrika, @pyshi
#
import numpy as np

from scipy.signal import savgol_filter


def create_traj_savgol(waypts, steps_per_waypt):
    '''

    :param waypts:
    :param steps_per_waypt:
    :return:
    '''
    """A scratch function to smooth trajectory."""
    waypts = waypts.reshape(-1, 3)
    n_waypts = waypts.shape[0]
    dists = []
    for i in range(n_waypts-1):
        dists.append(np.linalg.norm(waypts[i]-waypts[i+1]))
    tot_dist = sum(dists)
    t_max = n_waypts*steps_per_waypt
    dense_waypts = np.zeros((t_max, 3))
    t = 0
    for i in range(n_waypts-1):
        steps_per_waypt_weighted = int((dists[i]/tot_dist)*t_max)
        for k in range(steps_per_waypt_weighted):
            dense_waypts[t] = waypts[i]
            t += 1
    if t < t_max:
        dense_waypts[t:,:] = dense_waypts[t-1,:]  # set rest to last entry
    # dense_waypts = np.repeat(waypts, steps_per_waypt, axis=0)  # simple repeat
    dense_waypts = savgol_filter(dense_waypts,
                                 window_length=int(steps_per_waypt*2+1),
                                 polyorder=4, axis=0)
    print('dense_waypts', dense_waypts.shape)
    return dense_waypts

def create_traj(init_pos, waypoints, steps_per_waypoint, frequency):
    '''
    Create a smoothed velocity based trajectory through a given waypoints.
    :param init_pos: Initial position of the anchor
    :param waypoints: way points for each anchor [ n_waypoints,  3]
    :param steps_per_waypoint: Number of steps per way point [ n_waypoints ]
    :param frequency: Control frequency
    :return: A trajectory (of velocities) passing through all way points. []
    '''
    assert(len(waypoints)== len(steps_per_waypoint))
    init_pos = np.array(init_pos)
    waypoints = np.array(waypoints)
    assert init_pos.shape[-1] == waypoints.shape[-1], \
        'init_pos and waypoints must both be arrays of coordinates'
    if len(init_pos.shape) == 1:
        init_pos = init_pos[None,...]  # empty dim
    waypoints = np.concatenate([init_pos, waypoints], axis=0)
    num_wpts = len(waypoints)
    tot_steps = sum(steps_per_waypoint)
    dt = 1.0/frequency
    traj = np.zeros([tot_steps, 3+3])  # 3D pos , 3D vel
    prev_pos = waypoints[0]  # start at the 0th waypoint
    t = 0
    for wpt in range(1,num_wpts):
        tgt_pos = waypoints[wpt]
        dur = steps_per_waypoint[wpt-1]
        Y, Yd, Ydd = plan_min_jerk_trajectory(prev_pos, tgt_pos, dur*dt, dt)
        traj[t:t+dur,0:3] = Y[:]    # position
        traj[t:t+dur,3:6] = Yd[:]   # velocity
        # traj[t:t+dur,6:9] = Ydd[:]  # acceleration
        t += dur
        prev_pos = tgt_pos
    if t<tot_steps: traj[t:,:] = traj[t-1,:]  # set rest to last entry
    # print('create_trajectory(): traj', traj)
    return traj


def calculate_min_jerk_step(y_curr, yd_curr, ydd_curr, goal, rem_dur, dt):
    '''Math'''

    if rem_dur < 0:
        return

    if dt > rem_dur:
        dt = rem_dur

    t1 = dt
    t2 = t1 * dt
    t3 = t2 * dt
    t4 = t3 * dt
    t5 = t4 * dt

    dur1 = rem_dur
    dur2 = dur1 * rem_dur
    dur3 = dur2 * rem_dur
    dur4 = dur3 * rem_dur
    dur5 = dur4 * rem_dur

    dist = goal - y_curr
    a1t2 = 0.0  # goaldd
    a0t2 = ydd_curr * dur2
    v1t1 = 0.0  # goald
    v0t1 = yd_curr * dur1

    c1 = (6.0 * dist + (a1t2 - a0t2) / 2.0 - 3.0 * (v0t1 + v1t1)) / dur5
    c2 = (-15.0 * dist + (3.0 * a0t2 - 2.0 * a1t2) /
          2.0 + 8.0 * v0t1 + 7.0 * v1t1) / dur4
    c3 = (10.0 * dist + (a1t2 - 3.0 * a0t2) /
          2.0 - 6.0 * v0t1 - 4.0 * v1t1) / dur3
    c4 = ydd_curr / 2.0
    c5 = yd_curr
    c6 = y_curr

    y = c1 * t5 + c2 * t4 + c3 * t3 + c4 * t2 + c5 * t1 + c6
    yd = 5 * c1 * t4 + 4 * c2 * t3 + 3 * c3 * t2 + 2 * c4 * t1 + c5
    ydd = 20 * c1 * t3 + 12 * c2 * t2 + 6 * c3 * t1 + 2 * c4

    return y, yd, ydd


def plan_min_jerk_trajectory(y0, goal, dur, dt):
    N = round(dur / dt)
    nDim = np.shape(y0)[0]
    Y = np.zeros((N, nDim))
    Yd = np.zeros((N, nDim))
    Ydd = np.zeros((N, nDim))
    Y[0, :] = y0
    rem_dur = dur
    for n in range(1, N):
        y_curr = Y[n - 1, :]
        yd_curr = Yd[n - 1, :]
        ydd_curr = Ydd[n - 1, :]
        for d in range(nDim):
            Y[n, d], Yd[n, d], Ydd[n, d] = calculate_min_jerk_step(
                y_curr[d], yd_curr[d], ydd_curr[d], goal[d], rem_dur, dt)
        rem_dur = rem_dur - dt
    return Y, Yd, Ydd

