# Names for the types of tasks.
#
# @ contactrika
#
import numpy as np


TASK_TYPES = ['HangBag', 'HangCloth', 'Button', 'Dress', 'Hoop', 'Lasso',] # 'Mask']


SCENE_INFO = {
    'basic': {
        # TODO reward function targets points
    },
    'hook': {
        'entities': {
            'cuboid.urdf': {
                'basePosition': [0.0, -0.15, 0.2],
            },
            'hook.urdf': {
                'basePosition': [0.00, (0.3 + 0.1) / 2 - 0.15, 0.30],
                'baseOrientation': [0, 0, np.pi / 2],
            },
        },
        'goal_pos': [0.00, 0.03, 0.31],
        'easy_target_pos': [0.00, 0.13, 0.33],
    },
    'dress': {
        'entities': {
            'figure.urdf': {
                'basePosition': [-0.22, 0, 0.28],
                'baseOrientation': [0, 0, np.pi / 2],
                'globalScaling': 0.95,
            },
            'zparts/pointy_ear_small.obj': {
                'basePosition': [-0.296, 0.00, 0.64],
                'baseOrientation': [0, 0, np.pi * 1.05],
                'globalScaling': 1,
                'rgbaColor': (0.9, 0.75, 0.65, 1),
            },
            'zparts/pointy_ear_small_flip.obj': {
                'basePosition': [-0.146, 0.00, 0.64],
                'baseOrientation': [0, 0, np.pi * 1.05],
                'globalScaling': 1,
                'rgbaColor': (0.9, 0.75, 0.65, 1),
            },
        },
        'goal_pos': [-0.077, 0.03, 0.315],
        'easy_target_pos': [-0.077, 0.12, 0.315],
    },

    'hang': {
        'entities': {
            'urdf/cuboid.urdf': {
                'basePosition': [0.0, -0.15, 0.20],
                'baseOrientation': [0, 0, 0],
                'globalScaling': 1.0,
            },
            'urdf/hook.urdf': {
                'basePosition': [0.0, (0.3+0.1)/2-0.15, 0.30],
                'baseOrientation': [0, 0, np.pi/2],
                'globalScaling': 1.0,
            },
        },
        'goal_pos': [0.00, 0.05, 0.26],
    },
    'hang_wide': {
        'entities': {
            'rod.urdf': {
                'basePosition': [0, 0.15, 0],
                'baseOrientation': [0, 0, 0, 1],
                'globalScaling': 0.95
            },
            'rigid/ts_hanger_curved_wide.obj': {
                'basePosition': [0, 0.15, 0.41],
                'baseOrientation': [0, 0, np.pi / 2],
            },
        },
        'goal_pos': [0, 0.15, 0.35],
        'easy_target_pos': [0, 0.15, 0.42],
    },

    # ======== Collect traj scenes stops here ===

    'button': {
        'entities': {
            'torso.urdf': {
                'basePosition': [0.0, 0.0, 0.15],
                'globalScaling': 1.28,
            },
            'button_fixed.urdf': {
                'basePosition': [-0.02, 0.13, 0.240],
                'baseOrientation': [0, 0, 0, 1],
                'globalScaling': 1.28,
            },
            'button_fixed2.urdf': {
                'basePosition': [0.00, 0.13, 0.13],
                'baseOrientation': [0, 0, 0, 1],
                'globalScaling': 1.28,
            },

        },
        'goal_pos': [-0.02, 0.13, 0.250],
        'easy_target_pos': [-0.02, 0.15, 0.250],
    },

    'hoop': {
        'entities': {
            'rod.urdf': {
                'basePosition': [0.00, 0.00, 0.00],
                'baseOrientation': [0, 0, 0, 1],
            },
        },
        'goal_pos': [0.00, 0.00, 0.05],
        'easy_target_pos': [0.00, 0.00, 0.35],
    },

    'lasso': {
        'entities': {
            'rod.urdf': {
                'basePosition': [0.00, 0.00, 0.00],
                'baseOrientation': [0, 0, 0, 1],
            },
        },
        'goal_pos': [0.00, 0.00, 0.05],
        'easy_target_pos': [0.00, 0.00, 0.35],
    },
}


"""
Scene and mesh presets.

@contactrika, @pyshi

"""

DEFORM_INFO = {
    'cloth/ts_apron_twoloops.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL
        'anchor_init_pos' : [-0.04, 0.35, 0.75],
        'other_anchor_init_pos' : [0.04, 0.35, 0.75],
        'deform_init_pos': [0, 0.40, 0.57],
        'deform_init_ori': [0, 0, np.pi/2],
        'deform_scale': 0.8,
        'deform_elastic_stiffness': 1.0,
        'deform_bending_stiffness': 1.0,
        'deform_anchored_vertex_ids': [
            [1158, 684, 1326, 1325, 1321, 1255, 1250, 683, 1015, 469, 470, 1235,
             1014, 1013, 479, 130, 1159, 1145, 1085, 478, 1087, 143, 131, 1160,
             1083, 542],
            [885, 116, 117, 118, 738, 495, 1210, 884, 1252, 883, 882, 881, 496,
             163, 164, 737, 165, 1290, 1166, 544, 739, 114, 115, 753, 886, 887]
        ],
        'deform_true_loop_vertices': [
            [81, 116, 117, 131, 145, 149, 150, 155, 160, 161, 164,
             168, 176, 299, 375, 377, 480, 483, 492, 497, 500, 501,
             502, 503, 504, 514, 521, 525, 539, 540, 542, 545, 548,
             735, 740, 743, 754, 761, 873, 992, 1019, 1084, 1149, 1159,
             1161, 1167, 1168, 1210, 1255, 1257],
            [51, 53, 57, 68, 157, 162, 170, 177, 179, 181, 182,
             185, 186, 195, 199, 201, 202, 205, 207, 229, 232, 240,
             295, 296, 297, 308, 309, 318, 328, 334, 361, 364, 365,
             367, 383, 392, 402, 409, 411, 414, 508, 510, 511, 527,
             530, 531, 532, 533, 536, 549, 551, 560, 577, 628, 633,
             647, 679, 680, 690, 691, 749, 752, 755, 771, 835, 854,
             856, 857, 859, 860, 867, 871, 872, 986, 988, 989, 990,
             991, 1001, 1008, 1021, 1023, 1152, 1153, 1163, 1164, 1169, 1170,
             1173, 1174, 1175, 1197, 1211, 1228, 1254, 1259, 1260, 1271, 1308,
             1319]
        ],
    },
    'bags/ts_purse_bag_resampled.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL
        'deform_init_pos': [0, 0.40, 0.57],
        'deform_init_ori': [0, 0, np.pi/2],
        'deform_scale' : 2.0,
        'deform_elastic_stiffness': 1.0,
        'deform_bending_stiffness': 1.0,
        'deform_anchored_vertex_ids': [
            [357, 355, 40, 356, 372, 379, 296, 403],
            [373, 342, 322, 384, 341, 383, 14, 317]],
        'deform_true_loop_vertices': [
            [0, 1, 6, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 22, 24, 27, 33, 34,
             36, 37, 38, 39, 40, 41, 42, 49, 50, 68, 104, 105, 106, 134, 136,
             137, 147, 172, 173, 174, 195, 198, 257, 264, 265, 266, 291, 303,
             304, 305, 306, 309, 310, 331, 333, 336, 337, 341, 343, 344, 345,
             346, 347, 350, 352, 353, 354, 355, 357, 358, 359, 361, 362, 363,
             364, 366, 367, 368, 372, 373, 378, 380, 382, 383, 386, 387, 389,
             390, 396, 397]
        ]
    },
    # An example of info for a custom item.
    'bags/bags_zehang/obj/normal/bag1-1.obj': {
        'deform_init_pos': [0, 0.47, 0.47],
        'deform_init_ori': [np.pi/2, 0, 0],
        'deform_scale': 0.1,
        'deform_elastic_stiffness': 1.0,
        'deform_bending_stiffness': 1.0,
        'deform_true_loop_vertices': [
            [0, 1, 2, 3]  # placeholder, since we don't know the true loops
        ]
    },
    'cloth/apron_z.obj': {
        'deform_init_pos': [0, 0.47, 0.47],
        'deform_init_ori': [np.pi / 2, 0, np.pi ],
        'deform_scale': 0.3,
        'deform_elastic_stiffness': 1.0,
        'deform_bending_stiffness': 1.0,
        'deform_true_loop_vertices': [
            [0, 1, 2, 3]  # placeholder, since we don't know the true loops
        ]
    },
    'cloth/apron_zehang.obj': {
        'deform_init_pos': [0, 0.47, 0.54],
        'deform_init_ori': [np.pi/2, 0, np.pi],

        'deform_scale': 0.3,
        'deform_elastic_stiffness': 1.0,
        'deform_bending_stiffness': 1.0,
        'deform_true_loop_vertices': [
            [0, 1, 2, 3]  # placeholder, since we don't know the true loops
        ]
    },
    'cloth/button_cloth.obj': {
        'deform_init_pos': [-0.13, 0.16, 0.21],
        'deform_init_ori': [0, 0, 0],
        'deform_scale': 0.5,
        'deform_elastic_stiffness': 10.0,
        'deform_bending_stiffness': 10.0,
        'deform_anchored_vertex_ids': [
            [1395, 1554, 1594, 1593, 1553, 1353, 1475, 1474, 1473, 1472, 1354, 1515, 1513, 1435, 1434, 1433, 1432, 1393,
             1394, 1514, 1555, 1315, 1355, 1392, 1592, 1552, 1595, 1313, 1512, 1352, 1314],
            [1560, 1400, 1439, 1440, 1479, 1480, 1481, 1519, 1520, 1521, 1559, 1399, 1478, 1359, 1401, 1522, 1442, 1441,
             1438, 1358, 1398, 1360, 1361, 1518, 1562, 1561, 1320, 1319, 1558, 1482, 1517]],
        'deform_fixed_anchor_vertex_ids':[[1, 120, 2, 3, 4, 5, 0, 82, 81, 80, 44, 43, 42, 41, 40, 121, 85, 83, 45, 200, 162, 161, 160, 122, 123, 6, 84, 124, 46, 201, 163],
            [31, 78, 119, 118, 117, 116, 115, 114, 79, 77, 76, 75, 74, 73, 72, 39, 38, 37, 36, 35, 34, 33, 32, 157, 158, 159, 113, 156, 155, 154, 71],
            [21, 104, 103, 102, 67, 66, 65, 64, 63, 62, 61, 60, 28, 27, 26, 25, 24, 23, 22, 19, 20, 105, 143, 18, 101, 100, 106, 68, 59, 29, 144],
            [51, 90, 45, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 47, 48, 46, 89, 88, 87, 86, 49, 50, 52, 44, 14, 128, 85, 91, 53, 129]],

        # 'cloth_friction_coeff': 0.1,
        'traj_preset': {
            'button': {
                'waypoints': [[[-0.13, 0.2028, 0.3056],
                               [-0.13, 0.2028, 0.3056],
                               [0.2, 0.115, 0.3056],
                               [0.2, 0.100, 0.3056],
                               [0.2, 0.16, 0.3056],
                               ],

                              [[-0.13, 0.2081, 0.1263],
                               [-0.13, 0.2081, 0.1263],
                               [0.2, 0.115, 0.1263],
                               [0.2, 0.100, 0.1263],
                               [0.2, 0.16, 0.1263],
                               ], ],
                'camera_pos': [0.5, 160, -30],
                'camera_target': (0.0, 0.0, 0.1),

            }
        }
    },
}
