# Scene and mesh info.
#
# @contactrika, @pyshi
#
# TODO: remove all meshes with prefix ts_ and replace with ours.
#
# TODO: replace slashes to accommodate Windows users (maybe later).
#
import numpy as np

# Task information dictionary/map from task names to mesh file names lists.
TASK_INFO = {
    'HangBag': ['bags/ts_purse_bag_resampled.obj',
                'bags/bags_zehang/obj/normal/bag1-1.obj'],
    'HangCloth': ['cloth/ts_apron_twoloops.obj',
                  'cloth/apron_zehang.obj',
                  'cloth/apron_z.obj'],
    'Button': ['button_cloth.obj'],
    'Dress': ['ts_backpack_resampled.obj'],
    'Hoop': ['ts_hoop.obj'],
    'Lasso': ['ts_lasso_sparser.obj'],
    'Mask': ['ts_mask.obj']  # TODO: add Mask task when new mesh is ready
}

# Information about rigid objects to load into the scene.
SCENE_INFO = {
    'dress': {
        'entities': {
            'urdf/figure.urdf': {
                'basePosition': [-0.22, 0, 0.28],
                'baseOrientation': [0, 0, np.pi / 2],
                'globalScaling': 0.95,
            },
            'parts/ts_pointy_ear_small.obj': {  # TODO: REMOVE - PROPRIETARY MESH
                'basePosition': [-0.296, 0.00, 0.64],
                'baseOrientation': [0, 0, np.pi * 1.05],
                'globalScaling': 1,
                'rgbaColor': (0.9, 0.75, 0.65, 1),
            },
            'parts/ts_pointy_ear_small_flip.obj': {  # TODO: REMOVE - PROPRIETARY MESH
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
                'basePosition': [0.0, -0.15, 0.25],
                'baseOrientation': [0, 0, 0],
                'globalScaling': 1.0,
            },
            'urdf/hook.urdf': {
                'basePosition': [0.0, (0.3+0.1)/2-0.15, 0.40],
                'baseOrientation': [0, 0, np.pi/2],
                'globalScaling': 1.0,
            },
        },
        'goal_pos': [0.00, 0.05, 0.46],
    },
    'button': {
        'entities': {
            'urdf/torso.urdf': {
                'basePosition': [0.0, 0.0, 0.15],
                'baseOrientation': [0, 0, 0],
                'globalScaling': 1.28,
            },
            'urdf/button_fixed.urdf': {
                'basePosition': [-0.02, 0.13, 0.240],
                'baseOrientation': [0, 0, np.pi/2],
                'globalScaling': 1.28,
            },
            'urdf/button_fixed2.urdf': {
                'basePosition': [0.00, 0.13, 0.13],
                'baseOrientation': [0, 0, np.pi/2],
                'globalScaling': 1.28,
            },
        },
        'goal_pos': [-0.02, 0.13, 0.250],
    },
    'hoop': {
        'entities': {
            'urdf/rod.urdf': {
                'basePosition': [0.00, 0.00, 0.00],
                'baseOrientation': [0, 0, 0],
                'globalScaling': 1.0,
            },
        },
        'goal_pos': [0.00, 0.00, 0.05],
        'easy_target_pos': [0.00, 0.00, 0.35],
    },
    'lasso': {
        'entities': {
            'urdf/rod.urdf': {
                'basePosition': [0.00, 0.00, 0.00],
                'baseOrientation': [0, 0, 0],
                'globalScaling': 1.0,
            },
        },
        'goal_pos': [0.00, 0.00, 0.05],
    },
}

# Information about deformable objects.
DEFORM_INFO = {
    'bags/ts_purse_bag_resampled.obj': {  # TODO: REMOVE - PROPRIETARY MESH
        'deform_init_pos': [0, 0.40, 0.57],
        'deform_init_ori': [0, 0, np.pi/2],
        'deform_scale': 2.0,
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
    'bags/bags_zehang/obj/normal/bag1-1.obj': {
        'anchor_init_pos' : [-0.04, 0.38, 0.75],
        'other_anchor_init_pos' : [0.04, 0.38, 0.75],
        'deform_init_pos': [0, 0.47, 0.47],
        'deform_init_ori': [np.pi/2, 0, 0],
        'deform_scale': 0.12,
    },
    'cloth/ts_apron_twoloops.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL
        'anchor_init_pos' : [-0.04, 0.35, 0.75],
        'other_anchor_init_pos' : [0.04, 0.35, 0.75],
        'deform_init_pos': [0, 0.40, 0.57],
        'deform_init_ori': [0, 0, np.pi/2],
        'deform_scale': 0.8,
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
    'cloth/apron_z.obj': {  # TODO: This used to be apron_zehang
        'deform_init_pos': [0, 0.47, 0.47],
        'deform_init_ori': [np.pi/2, 0, np.pi ],
        'deform_scale': 0.3,
    },
    'cloth/apron_zehang.obj': {  # TODO: Yonk - what is this object?
        'deform_init_pos': [0, 0.47, 0.54],
        'deform_init_ori': [np.pi/2, 0, np.pi],
        'deform_scale': 0.3,
    },
    'button_cloth.obj': {
        'anchor_init_pos': [-0.13, 0.8028, 0.3056],
        'other_anchor_init_pos': [-0.13, 0.8081, 0.1263],
        'deform_init_pos': [-0.13, 0.19, 0.17],  # [-0.13, 0.16, 0.21],
        'deform_init_ori': [0, 0, 0],
        'deform_scale': 0.8,  # 0.5
        'deform_fixed_anchor_vertex_ids':
            [1, 120, 2, 3, 31, 78, 119, 118, 21, 104, 103, 102, 51, 90, 45, 3],
    },
    'ts_lasso_sparser.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL
        'anchor_init_pos': [-0.1206, 0.4875, 0.4857],
        'other_anchor_init_pos': [-0.0867, 0.5413, 0.4843],
        'deform_init_pos': [-0.2, 0.42, 0.48],
        'deform_init_ori': [np.pi/2, 0, np.pi/2],
    },
    'ts_hoop.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL
        'anchor_init_pos': [-0.12, 0.48, 0.48],
        'other_anchor_init_pos': [-0.08, 0.54, 0.48],
        'deform_init_pos': [-0.14, 0.525, 0.48],
        'deform_init_ori': [0, np.pi/2, 0],
    },
    'ts_backpack_resampled.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL
        'deform_init_pos': [-0.08, 0.46, 0.52],
        'deform_init_ori': [0, 0, 0],
        'deform_scale': 0.75,
        'deform_true_loop_vertices': [
            [47, 51, 54, 75, 183, 184, 186, 188, 189, 190, 191,
             193, 656, 677, 681, 691, 785, 833, 863, 865, 871, 874,
             883, 885, 886, 887, 888, 890, 893, 897, 903, 909, 910,
             911, 913, 914, 916, 917, 920, 921, 925, 927, 928, 932,
             937, 944, 946, 948, 949, 950, 952, 953, 954, 956, 957,
             958, 965, 966, 967, 968, 1041, 1056, 1090, 1112, 1122, 1131,
             1132, 1172, 1173, 1176, 1177, 1183, 1196, 1345, 1361, 1362, 1364,
             1368, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1399, 1416, 1423,
             1445, 1464, 1472, 1477, 1480, 1485, 1491, 1493, 1494, 1495, 1496,
             1516],
            [21, 24, 244, 245, 249, 254, 255, 308, 696, 931, 941,
             943, 959, 962, 963, 969, 970, 971, 973, 975, 979, 982,
             986, 989, 991, 993, 994, 995, 996, 997, 998, 999, 1000,
             1003, 1004, 1007, 1010, 1012, 1015, 1017, 1021, 1022, 1023, 1024,
             1025, 1026, 1029, 1031, 1035, 1036, 1042, 1052, 1054, 1106, 1107,
             1108, 1134, 1169, 1179, 1182, 1189, 1220, 1230, 1252, 1334, 1335,
             1338, 1444, 1487, 1507, 1508, 1538, 1539, 1540, 1541, 1542, 1544,
             1545, 1559, 1563, 1567, 1572, 1578, 1599, 1603, 1604, 1605, 1606,
             1607, 1610, 1616, 1618, 1636, 1640, 1641, 1642, 1643, 1646, 1647,
             1653]
        ],
    },
    'ts_mask.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL
        'deform_init_pos': [0, 0.43, 0.65],
        'deform_init_ori': [0, 0, np.pi],
        'deform_scale': 0.50,
    },
}
