"""
Waypoints for preset trajectories.

@yonkshi

"""

from .task_info import (
    MESH_MAJOR_VERSIONS, TOTE_MAJOR_VERSIONS, TOTE_VARS_PER_VERSION)

preset_traj = {
    'cloth/apron_0.obj': {  # HangGarment-v0, 600 steps
        'waypoints': {
            'a': [
                # [ x, y, z, seconds(time)]
                # [2, 2, 10.5, 1],  # waypoint 0
                [2, -0.5, 9.5, 1],
                [2, -1, 8, 1],
                [2, -1.2, 7.5, 1],
            ],
            'b': [
                # [ x, y, z, seconds(time)]
                # [-2, 2, 10.5, 1],  # waypoint 0
                # [-2, -0.5, 9.5, 1],
                # [-2, -1, 8, 1],
                [-1.6, -0.5, 9.5, 1],
                [-1.6, -1, 8, 1],
                [-1.6, -1.2, 7.5, 1],
            ],
        },
    },
    'cloth/shirt_0.obj': {  # HangGarment-v5, 1500 steps
        'waypoints': {
            'a': [
                # [ x, y, z, seconds(time)]
                # [0.173 , 2.4621 ,6.5115, 1],
                [2.5, -0.5, 9, 3],
                [1, -1, 10.5, 1],
                [1, -1, 8, 1],
                [1, -1, 8, 5],

            ],
            'b': [
                # [ x, y, z, seconds(time)]
                # [-0.1283 , 2.4621 , 6.5115, 1]
                [0, 0.5, 9, 3],
                [-2, 0, 10.5, 1],
                [-4, 0, 10.5, 3],
                [-3, -1, 10.5, 0.5],
                [-1, -1, 10.5, 1],
                [-1, -1, 9, 0.5],

            ]
        },
    },
    'cloth/button_cloth.obj': {  # ButtonSimple-v0, 800 steps
        'waypoints': {
            'a': [
                # [ x, y, z, timesteps]
                [2.9, 00, 3.6, 1],
                [2.9, 0, 3.6, 3],

            ],
            'b': [
                # [ x, y, z, timesteps]
                [2.9, 0, 1.6, 1],
                [2.9, 0, 1.6, 3],

            ]
        },
    },
    'bags/totes/bag0_0.obj': {  # HangBag-v0, 1500 steps
        'waypoints': {
            'a': [
                [0.2, 2, 10.4, 2.5],
                [0.2, 1, 10.4, 0.5],
            ],
            'b': [
                [-0.2, 2, 10.4, 2.5],
                [-0.2, 1, 10.4, 0.5],
            ]
        },
    },
    'cloth/vest_0.obj': {  # Dress-v5 #
        'waypoints': {
            'a': [
                # [-0.278 ,  1.7888,  6.245 ],
                [0.6, 1.7888, 6.245, 0.6],
                # [0.6 ,  1.1,  6.245, 100 ],
                # [0.6 ,  0.8,  6.245, 100 ],
                [0.6, 0.0, 6.245, 0.8],
                [0, 0, 6.445, 1.2],

            ],
            'b': [
                # [0.3004, 1.7888, 6.245 ]
                [2.3, 1.7888, 6.245, 0.6],
                # [2.3, 0.5, 6.245, 100 ],
                # [2.3, -0.3, 6.245, 100 ],
                [2.8, -1, 6.245, 0.8],
                [-0.5, -4, 6.245, 1.2],
                [-3, -1, 6.245, 0.6],
                [-2, 2, 6.245, 0.8],
                [-1, 2, 6.245, 0.2],
                # [-1, 2, 6.245, 0.6 ],
                [-1, 0, 6.245, 0.6],

            ]
        },
    },
    'bags/backpack_0.obj': {  # Dress-v0
        'waypoints': {
            'a': [
                [-0.8019, 0.9734, 4.0249, 1],
                [0.3, 1.7888, 6.5245, 2],
                [0.3, 0.0, 6.9245, 3],
                [0.8, -0.5, 6.945, 1.2],

            ],
            'b': [
                [0.1823, 0.9618, 4.4351, 1],
                [3.7, 1.7888, 6.945, 2],
                [3.7, -1, 6.945, 3],
                [0, -3, 7.245, 2],
                [-3, -1, 6.845, 0.6],
                [-1.5, 2, 6.845, 0.8],
                [-1.5, 1, 6.845, 0.6],

            ]
        },
    },
    'cloth/mask_0.obj': {  # Mask-v0
        'waypoints': {
            'a': [
                # [0.4332, 1.9885, 6.1941],
                [1, 0, 7.2, 1],
                [1.0, -0.5, 7.1, 0.5],
                [0.3, -1, 7.0, 0.5],
                # [0.3, -1, 7.1, 0.5],
                # [0.3, -1, 6.7, 0.5],
            ],
            'b': [
                # [-0.8332 , 1.9885 , 6.1941]
                [-1.5, 0, 7.2, 1],
                [-1.7, -0.5, 7.1, 0.5],
                [-0.6, -1.2, 7.0, 0.5],
                # [-0.6, -1.2, 6.0, 0.5],

            ]
        },
    },
    'ropes/lasso3d_0_fast.obj': {
        'waypoints': {
            'a': [
                # [ [-2.8518, -0.2436 , 5.9087]],
                [-2.8518, -0.2436, 5.9087, 2],
                [-0.3518, -0.0, 8, 1],
                [2, 0, 8, 1],
                [2, 0, 1, 1],

            ],
            'b': [
                # [-3.6025, -0.3533,  5.9768]
                [-3.6025, -0.3533, 5.9768, 2],
                [-1.1025, -0.0, 8, 1],
                [1.3, 0, 8, 1],
                [1.3, 0, 1, 1],

            ]
        },
    },
    'ropes/hoop3d_0.obj': {
        'waypoints': {
            'a': [
                # [1.5708, 1.9639, 1.1152],
                [1.5708, 1.9639, 6, 1.1],
                [-0.5, -0.3, 6, 1.1],
                [-0.5, -0.3, 1, 1.1],
            ],
            'b': [
                # [2.2903, 1.9168, 5.9768]
                [2.2903, 1.9168, 6, 1.1],
                [-0.2, -0.5, 6, 1.1],
                [-0.2, -0.5, 1, 1.1],

            ]
        },
    },
    'food': {
        'waypoints': {
            'a': [
                [2.0, 0.5, 0.9],
                [1.0, 2.1, 0.9],
                [2.0, 0.5, 0.9],
                [2.0, 4.3, 0.9],
                [-0.7, 0.9, 0.9],
            ],
            'a_theta': [
                [-3.115, 0.007, -3.069],
                [-3.115, 0.007, 2.3561],
                [-3.115, 0.007, 2.3561],
                [-3.115, 0.007, -3.069],
                [-3.115, 0.007, -2.4561],
            ],
        },
    },
}

# Populate presents for all mesh versions.
bag_waypts = preset_traj['bags/totes/bag0_0.obj']
for v in range(1, TOTE_MAJOR_VERSIONS):
    for vv in range(0, TOTE_VARS_PER_VERSION):
        preset_traj[f'bags/totes/bag{v:d}_{vv:d}.obj'] = bag_waypts
backpack_waypts = preset_traj['bags/backpack_0.obj']
apron_waypts = preset_traj['cloth/apron_0.obj']
shirt_waypts = preset_traj['cloth/shirt_0.obj']
mask_waypts = preset_traj['cloth/mask_0.obj']
vest_waypts = preset_traj['cloth/vest_0.obj']
for v in range(1, MESH_MAJOR_VERSIONS):
    preset_traj['bags/backpack_' + str(v) + '.obj'] = backpack_waypts
    preset_traj['cloth/apron_' + str(v) + '.obj'] = apron_waypts
    preset_traj['cloth/shirt_' + str(v) + '.obj'] = shirt_waypts
    preset_traj['cloth/vest_' + str(v) + '.obj'] = vest_waypts
    preset_traj['cloth/mask_' + str(v) + '.obj'] = mask_waypts
