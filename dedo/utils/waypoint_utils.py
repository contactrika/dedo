#
# Utilities for following trajectories in PyBullet.
#
# @contactrika, @pyshi
#
import numpy as np

from scipy.signal import savgol_filter


def interpolate_waypts(waypts, steps_per_waypt):
    """A scratch function to smooth trajectory. Not tested."""
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
