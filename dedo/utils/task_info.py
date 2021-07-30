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
                'bags/bags_zehang/obj/normal/bag1-1.obj',
                ],
    'HangCloth': ['cloth/apron_0.obj',
                  'cloth/apron_1.obj',
                  'cloth/apron_2_dense.obj',
                  'cloth/apron_3_dense.obj',
                  'cloth/apron_4.obj',
                  'cloth/tshirt_0.obj',
                  ],
    'ButtonSimple': ['button_cloth.obj'],
    'ButtonProcedural': ['button_cloth.obj'],  # TODO wire up procedural cloths
    'Dress': ['ts_backpack_resampled.obj',
              'bags/backpack_0.obj',
              'bags/backpack_0_dense.obj',
              'bags/backpack_0_dense_flipped.obj',
              'bags/backpack_0_thick.obj', # Failure case
              'bags/backpack_1.obj',
              'bags/backpack_2.obj',
              'bags/backpack_3.obj',
              ],
    'Hoop': ['ts_hoop.obj'],
    'Lasso': ['ts_lasso_sparser.obj',
              'ropes/lasso3d_0.obj',
              ],
    'Mask': ['cloth/mask_0.obj',
             'cloth/mask_1.obj',
             'cloth/mask_2.obj',
             'cloth/mask_3.obj',
             'cloth/mask_4.obj',
             ],
    'Debug': ['ts_backpack_resampled.obj',
              'bags/backpack_0.obj',
              'bags/backpack_0_dense.obj',
              'bags/backpack_0_thick.obj',
              'cloth/apron_2_dense.obj',
              'cloth/ts_apron_twoloops.obj',
              'cloth/apron_1.obj',

              'cloth/apron_2_dense.obj',
              'cloth/apron_3_dense.obj',
              ]
}

# Information about rigid objects to load into the scene.
SCENE_INFO = {
    'debug': {
        'entities': {
            # 'debug_pole_dense.obj': {
            #     'basePosition': [-0.22, 0.00, 0.5],  # 'basePosition': [-0.22, 0.00, 0.64],
            #     'baseOrientation': [np.pi/2, 0, 0],
            #     'globalScaling': 0.05,
            #     'rgbaColor': (0.9, 0.75, 0.65, 1),
            # },
            'urdf/hanger.urdf': {
                'basePosition': [0.0, 0, 5.5],
                'baseOrientation': [0, 0, 0],
                'globalScaling': 10.0,
            },
            'urdf/rod.urdf': {
                'basePosition': [0.00, 0.00, 2.00],
                'baseOrientation': [0, 0, 0],
                'globalScaling': 10.0,
            },
        },
        'goal_pos': [-0.077, 0.03, 0.315],
        'easy_target_pos': [-0.077, 0.12, 0.315],
    },
    'dress': {
        'entities': {
            'urdf/figure_headless.urdf': {
                'basePosition': [-0.22, 0, 3.00],
                'baseOrientation': [0, 0, np.pi / 2],
                'globalScaling': 10,
            },
            'head_with_ears.obj': {
                'basePosition': [-0.22, 0.00, 6.54],  # 'basePosition': [-0.22, 0.00, 0.64],
                'baseOrientation': [0, 0, np.pi * 1.05],
                'globalScaling': 0.5,
                'rgbaColor': (0.9, 0.75, 0.65, 1),
            },
        },
        'goal_pos': [-1, 0.03, 5.7], # TODO other arm's goal position
        'easy_target_pos': [-0.077, 2, 3.00],
    },
    'hang': {
        'entities': {
            'urdf/hanger.urdf': {
                'basePosition': [0.0, 0, 5.5],
                'baseOrientation': [0, 0, 0],
                'globalScaling': 10.0,
            },
            'urdf/rod.urdf': {
                'basePosition': [0.00, 0.00, 2.00],
                'baseOrientation': [0, 0, 0],
                'globalScaling': 10.0,
            },

        },
        'goal_pos': [0.00, 0.00, 5.7],
    },
    'button': {
        'entities': {
            'urdf/torso.urdf': {
                'basePosition': [0.0, 0.0, 2],
                'baseOrientation': [0, 0, 0],
                'globalScaling': 12.8,
            },
            'urdf/button_fixed.urdf': {
                'basePosition': [-0.02, 1.28, 2.9],
                'baseOrientation': [0, 0, np.pi / 2],
                'globalScaling': 12.8,
            },
            'urdf/button_fixed2.urdf': {
                'basePosition': [0.00, 1.28, 1.2],
                'baseOrientation': [0, 0, np.pi / 2],
                'globalScaling': 12.8,
            },
        },
        'goal_pos':[0.0, 1.1, 1.3], # TODO top button goal pos
    },
    'hoop': {
        'entities': {
            'urdf/rod.urdf': {
                'basePosition': [0.00, 0.00, 0.00],
                'baseOrientation': [0, 0, 0],
                'globalScaling': 10.0,
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
                'globalScaling': 10.0,
            },
        },
        'goal_pos': [0.00, 0.00, 0.05],
        'easy_target_pos': [0.00, 0.00, 0.35],
    },
}

# Information about deformable objects.
DEFORM_INFO = {
    'bags/ts_purse_bag_resampled.obj': {  # TODO: REMOVE - PROPRIETARY MESH
        'deform_init_pos': [0, 2, 7],
        'deform_init_ori': [0, 0, np.pi / 2],
        'deform_scale': 20.0,
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
        'deform_init_pos': [0, 2, 6],
        'deform_init_ori': [np.pi / 2, 0, 0],
        'deform_scale': 1,
        'anchor_init_pos': [-0.5, 2, 8],
        'other_anchor_init_pos': [0.5, 2, 8],
    },
    'cloth/ts_apron_twoloops.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL
        'anchor_init_pos': [-0.04, 0.35, 0.75],
        'other_anchor_init_pos': [0.04, 0.35, 0.75],
        'deform_init_pos': [0, 0.8, 6.5], # [0, 0.05, 0.47],
        'deform_init_ori': [-0, -np.pi / 2, np.pi / 2],
        'deform_scale': 8,
        'deform_anchor_vertices': [
            [131],
            [116],
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
    'ts_backpack_resampled.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL

        'deform_init_pos': [-0.2, 2, 4],  # [-0.2, 0.2, 0.52]
        # 'deform_init_ori': [0, 0, 0],
        'deform_init_ori': [0, 0, -np.pi / 2],
        'deform_scale': 5,
        'deform_anchor_vertices': [
            [999, 1000, 1138, 1140, 1152, 1293, 1298, 1301, 1302, 1304, 1305, 1341, 1342, 1343, 1344, 1370],
            [935, 1030, 1031, 1082, 1087, 1373, 1374, 1381, 1409],
        ],
        'deform_true_loop_vertices': [
            [47, 51, 54, ],
            [21, 24, 244, ]
        ],
    },

    'bags/backpack_0.obj': {
        'deform_init_pos': [-0.2, 2, 4],
        'deform_init_ori': [-np.pi / 2, -np.pi, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [ 270],
            [ 549],
        ],
        'deform_true_loop_vertices': [
            [140, 201, 220, 240, 241, 242, 243, 244, 247, 249, 252, 254, 256, 258, 259, 263, 264, 265, 266, 267, 268,
             269, 270, 271, 272, 278, 296, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 587],
            [311, 313, 326, 327, 391, 392, 406, 433, 435, 496, 516, 517, 518, 519, 520, 522, 524, 527, 530, 531, 533,
             535, 537, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 554, 572, 574, 576, 577, 578, 579, 580, 581,
             582, 583, 584],
        ],
    },
    'bags/backpack_0_dense.obj': {
        'deform_init_pos': [-0.2, 2, 4],
        'deform_init_ori': [-np.pi / 2, -np.pi, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [247, ],
            [522, ],
        ],
        'deform_true_loop_vertices': [
            [140, 201, 220, 240, 241, 242, 243, 244, 247, 249, 252, 254, 256, 258, 259, 263, 264, 265, 266, 267, 268,
             269, 270, 271, 272, 278, 296, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 587],
            [311, 313, 326, 327, 391, 392, 406, 433, 435, 496, 516, 517, 518, 519, 520, 522, 524, 527, 530, 531, 533,
             535, 537, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 554, 572, 574, 576, 577, 578, 579, 580, 581,
             582, 583, 584],
        ],
    },
    'bags/backpack_0_dense_flipped.obj': {
        'deform_init_pos': [-0.2, 2, 4],
        'deform_init_ori': [-np.pi / 2, -np.pi, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [244, 247, 259, 263, 1394, 1489, 1628, 1799, 1938, 1941, 2054, 2057],
            [518, 520, 522, 537, 782, 1012, 1013, 1034, 1036, 2034],
        ],
        'deform_true_loop_vertices': [
            [140, 201, 220, 240, 241, 242, 243, 244, 247, 249, 252, 254, 256, 258, 259, 263, 264, 265, 266, 267, 268,
             269, 270, 271, 272, 278, 296, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 587],
            [311, 313, 326, 327, 391, 392, 406, 433, 435, 496, 516, 517, 518, 519, 520, 522, 524, 527, 530, 531, 533,
             535, 537, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 554, 572, 574, 576, 577, 578, 579, 580, 581,
             582, 583, 584],
        ],
    },
    'bags/backpack_0_thick.obj': {
        'deform_init_pos': [-0.2, 2, 4],
        'deform_init_ori': [-np.pi / 2, -np.pi, np.pi],
        'deform_scale': 4,
        'deform_anchor_vertices': [
            [262, 263, 264, 265, 266, 267, 268, 269, 270],
            [522, 536, 537, 538, 545, 547, 1112, 1127, 1128, 1130, 1131, 1133],
        ],
        'deform_true_loop_vertices': [
            [140, 201, 220, 240, 241, 242, 243, 244, 247, 249, 252, 254, 256, 258, 259, 263, 264, 265, 266, 267, 268,
             269, 270, 271, 272, 278, 296, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 587],
            [311, 313, 326, 327, 391, 392, 406, 433, 435, 496, 516, 517, 518, 519, 520, 522, 524, 527, 530, 531, 533,
             535, 537, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 554, 572, 574, 576, 577, 578, 579, 580, 581,
             582, 583, 584],
        ],
    },
    'bags/backpack_1.obj': {
        'deform_init_pos': [-0.2, 2, 4],
        'deform_init_ori': [-np.pi / 2, -np.pi, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [260, ],
            [535, ],
        ],
        'deform_true_loop_vertices': [
            [116, 140, 202, 219, 239, 240, 242, 243, 244, 245, 246, 247, 248, 250, 251, 252, 253, 254, 257, 261, 262,
             263, 264, 265, 266, 267, 268, 269, 270, 271, 295, 297, 298, 299, 300, 301, 302, 303, 304, 305, 587],
            [185, 398, 407, 427, 479, 495, 516, 517, 519, 520, 521, 522, 523, 526, 527, 528, 529, 531, 536, 537, 539,
             540, 541, 542, 543, 544, 545, 546, 548, 571, 573, 575, 576, 577, 578, 579, 580, 581, 582, 583, 589],
        ],
    },
    'bags/backpack_2.obj': {
        'deform_init_pos': [-0.2, 0.27, 0.6],
        'deform_init_ori': [-np.pi / 2, -np.pi, np.pi],
        'deform_scale': 0.3,
        'deform_anchor_vertices': [
            [261,],
            [538, ],
        ],
        'deform_true_loop_vertices': [
            [23, 25, 96, 98, 115, 116, 146, 154, 243, 244, 246, 247, 249, 250, 252, 253, 254, 255, 256, 257, 258, 259,
             261, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 280, 298, 299, 300, 301, 302, 303, 304, 305,
             306, 307, 308],
            [183, 220, 239, 313, 328, 329, 393, 394, 397, 411, 434, 436, 496, 518, 519, 520, 521, 522, 524, 526, 529,
             530, 531, 532, 534, 536, 539, 540, 542, 543, 544, 545, 546, 547, 548, 549, 551, 574, 576, 578, 579, 580,
             581, 582, 583, 584, 585, 586, 589],
        ],
    },
    'bags/backpack_3.obj': {
        'deform_init_pos': [-0.2, 0.27, 0.6],
        'deform_init_ori': [-np.pi / 2, -np.pi, np.pi],
        'deform_scale': 0.3,
        'deform_anchor_vertices': [
            [265, ],
            [525,],
        ],
        'deform_true_loop_vertices': [
            [20, 26, 98, 100, 118, 246, 249, 250, 251, 253, 255, 256, 257, 259, 264, 265, 266, 267, 268, 269, 270, 271,
             272, 273, 298, 299, 300, 301, 302, 303, 304, 305, 306, 308],
            [95, 185, 313, 330, 398, 412, 439, 441, 445, 481, 496, 520, 522, 525, 526, 529, 531, 532, 533, 534, 536,
             539, 540, 542, 543, 544, 545, 546, 547, 548, 549, 551, 574, 576, 578, 579, 580, 581, 582, 583, 584, 586],
        ],
    },
    'bags/backpack_4.obj': {
        'deform_init_pos': [-0.2, 0.27, 0.6],
        'deform_init_ori': [-np.pi / 2, -np.pi, np.pi],
        'deform_scale': 0.3,
        'deform_anchor_vertices': [
            [0],
            [0],
        ],
        'deform_true_loop_vertices': [
            [0],
            [0],
        ],
    },

    'cloth/apron_0.obj': {  # This used to be apron_zehang
        'deform_init_pos': [0, 0.47, 0.47],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 0.3,
        'deform_anchor_vertices': [
            [15],  # 10, 12, 13, 14, 15],
            [170],  # 163, 165, 167, 168, 170],
        ],
        'deform_true_loop_vertices': [
            [0, 1, 3, 4, 7, 8, 11, 12, 14, 16, 22, 24, 26, 30, 31, 32, 154, 155, 157, 158, 160, 161, 162, 165, 167, 169,
             175, 177, 179, 183, 184, 284],
            [65, 98, 99, 100, 101, 124, 134, 135, 136, 137, 138, 139, 140, 141, 148, 149, 152, 153, 235, 236, 237, 238,
             258, 260, 266, 267, 268, 269, 270, 271, 272, 273, 276, 278, 280, 281, 282, 287, 288]
        ],
    },
    'cloth/apron_1.obj': {  # This used to be apron_zehang
        'deform_init_pos': [0, 0.3, 7.5],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [15],  # 10, 12, 13, 14, 15],
            [174],  # 167, 169, 171, 172, 174],
        ],
        'deform_true_loop_vertices': [
            [1, 2, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22, 24, 26, 29, 30, 31, 32, 33, 158, 159, 160, 161,
             162, 163, 164, 165, 166, 168, 169, 171, 172, 173, 174, 175, 179, 181, 183, 184, 187, 188, 215, 289],
            [67, 69, 102, 103, 104, 105, 127, 128, 137, 138, 139, 140, 141, 142, 143, 145, 147, 148, 151, 154, 155, 156,
             157, 209, 210, 239, 240, 241, 242, 263, 264, 270, 271, 272, 273, 274, 276, 277, 278, 279, 285, 286, 288,
             293],
        ],
    },
    'cloth/apron_2_dense.obj': {
        'deform_init_pos': [0, 0.45, 7.5],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [181,],  # , 72, 73, 76, 77, 78, 79, 227, 230, 602, 604, 721, 722],
            [ 16,],  # 159, 161, 162, 163, 285, 288, 520, 522, 680, 681, 763],
        ],
        'deform_true_loop_vertices': [
            [3, 4, 5, 9, 10, 11, 14, 62, 68, 69, 70, 71, 76, 78, 163, 164, 239, 281, 282, 283, 286, 287, 288, 354, 406,
             408, 410, 411, 412, 414, 415, 417, 419, 441, 442, 481, 517, 524, 526, 527, 528, 529, 530, 532, 559, 604,
             607, 608, 676, 677, 679, 680, 681, 685, 688, 689, 717, 719, 720, 721, 724, 733, 734, 761, 766, 785, 806,
             807, 808, 809, 810, 836, 837, 839, 841, 843, 869],
            [34, 35, 60, 61, 103, 127, 150, 153, 213, 215, 218, 219, 221,
             249, 270, 271, 274, 276, 278, 322, 328, 340, 363, 372, 378, 384,
             386, 387, 388, 391, 392, 394, 436, 447, 472, 492, 495, 496, 497,
             500, 502, 503, 504, 505, 506, 507, 552, 553, 570, 589, 590, 591,
             594, 597, 598, 599, 616, 617, 631, 654, 658, 661, 662, 663, 666,
             669, 670, 671, 672, 709, 710, 712, 713, 715, 716, 750, 752, 753,
             754, 756, 757, 759, 778, 780, 782, 793, 801, 803, 828, 829, 830,
             831, 832, 833, 863, 864, 865, 866, 867, 868],
        ],
    },
    'cloth/apron_3_dense.obj': {  # This used to be apron_zehang
        'deform_init_pos': [0, 2, 6.5],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [128],  # 10, 12, 13, 14, 15],
            [79],  # 167, 169, 171, 172, 174],
        ],
        'deform_true_loop_vertices': [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 20, 22, 24, 26, 29, 30, 31, 32, 33, 115, 158, 159,
             160, 161, 162, 163, 164, 165, 166, 168, 169, 171, 172, 173, 174, 177, 178, 179, 181, 183, 184, 187, 188,
             287],
            [67, 68, 69, 102, 103, 104, 105, 127, 128, 137, 138, 139, 140, 141, 142, 143, 145, 150, 151, 154, 155, 156,
             157, 209, 210, 239, 240, 241, 242, 263, 264, 270, 271, 272, 273, 274, 276, 277, 278, 279, 285, 286, 288,
             293],
        ],
    },
    'cloth/apron_4.obj': {  # This used to be apron_zehang
        'deform_init_pos': [0, 0.47, 0.47],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 0.3,
        'deform_anchor_vertices': [
            [15],  # 10, 12, 13, 14, 15],
            [173],  # 166, 168, 170, 171, 173],
        ],
        'deform_true_loop_vertices': [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 22, 24, 26, 29, 30, 31, 32, 33, 133, 157,
             158, 159, 160, 161, 162, 163, 164, 165, 167, 168, 170, 171, 172, 173, 174, 178, 180, 182, 183, 186, 187,
             214, 289],
            [67, 69, 102, 103, 104, 105, 127, 128, 137, 138, 139, 143, 145, 146, 147, 148, 153, 155, 156, 208, 238, 239,
             240, 241, 262, 263, 269, 270, 271, 272, 277, 278, 282, 283, 284, 285, 287, 288],
        ],
    },
    'cloth/apron_zehang.obj': {  # TODO: Yonk - what is this object?
        'deform_init_pos': [0, 0.47, 0.54],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 0.3,
        'deform_true_loop_vertices': [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 21, 22, 44, 76, 79, 80, 81, 82,
             83, 84, 85, 86, 87, 88, 89, 90, 91, 97, 98],
            [35, 37, 51, 52, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
             72, 73, 74, 75, 78, 109, 111, 112, 114, 119, 120, 121, 122, 123, 124, 125, 126, 127,
             128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138]
        ],
    },
    'cloth/tshirt_0.obj': {  # This used to be apron_zehang
        'deform_init_pos': [0, 0.47, 8],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [0],
            [0],
        ],

        'deform_true_loop_vertices': [
            [0],
            [0]
        ],
    },
    'cloth/mask_0.obj': {
        'deform_init_pos': [-0.2, 2, 6.2],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 1,
        'deform_anchor_vertices': [
            [354, 356, 357, 358, 359, 360, 361, 380, 382, 389, 398],
            [288, 289, 290, 292, 293, 294, 295, 315, 317, 405]
        ],
        'deform_true_loop_vertices': [
            [1, 56, 82, 232, 233, 258, 338, 339, 341, 343, 345, 347, 349, 357, 362, 365, 367, 370, 371, 372, 373, 374,
             375, 379, 387, 389, 393, 394, 395, 396, 397, 399, 406, 408],
            [117, 193, 194, 198, 212, 215, 216, 217, 268, 269, 271, 273, 275, 277, 278, 279, 280, 281, 290, 296, 297,
             299, 301, 303, 305, 306, 307, 308, 309, 314, 322, 402, 404]
        ],
    },
    'cloth/mask_1.obj': {
        'deform_init_pos': [-0.2, 2, 6.2],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 1,
        'deform_anchor_vertices': [
            [355, 357, 360, 363, 383, 384, 385, 407, 408],
            [288, 289, 294, 295, 315, 316, 317, 405, 406],
        ],
        'deform_true_loop_vertices': [
            [0, 232, 233, 340, 341, 342, 344, 346, 348, 350, 351, 353, 360, 361, 363, 365, 367, 369, 372, 374, 376, 378,
             380, 382, 387, 389, 390, 391, 396, 398, 400, 401, 407, 408],
            [117, 200, 215, 217, 268, 270, 272, 274, 276, 278, 280, 282, 284, 286, 293, 294, 295, 297, 298, 299, 301,
             303, 305, 307, 309, 311, 313, 322, 323, 324, 405, 406],
        ],
    },
    'cloth/mask_2.obj': {
        'deform_init_pos': [-0.2, 2, 6.2],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 1,
        'deform_anchor_vertices': [
            [354, 359, 362, 382, 383, 384, 407, 408],
            [288, 289, 294, 295, 315, 316, 317, 405, 406],
        ],
        'deform_true_loop_vertices': [
            [0, 231, 232, 339, 340, 347, 349, 352, 357, 360, 364, 371, 373, 377, 379, 381, 385, 388, 398, 401, 407,
             408],
            [117, 199, 215, 217, 266, 272, 274, 282, 284, 295, 296, 301, 303, 311],
        ],
    },
    'cloth/mask_3.obj': {
        'deform_init_pos': [-0.2, 2, 6.2],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 1,
        'deform_anchor_vertices': [
            [358, 362, 363, 364, 365, 385, 387],
            [289, 290, 291, 294, 295, 296, 297, 317, 318, 319, 405, 406],
        ],
        'deform_true_loop_vertices': [
            [0, 1, 3, 81, 83, 165, 166, 233, 234, 268, 333, 342, 350, 352, 353, 367, 374, 376, 378, 380, 399, 401],
            [99, 117, 119, 126, 190, 192, 198, 212, 213, 215, 270, 278, 280, 281, 282, 284, 299, 307, 309, 310, 311,
             313],
        ],
    },
    'cloth/mask_4.obj': {
        'deform_init_pos': [-0.2, 2, 6.2],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 1,
        'deform_anchor_vertices': [
            [359, 365, 366, 387, 407, 408],
            [294, 295, 296, 299, 300, 301, 302, 323, 406],
        ],
        'deform_true_loop_vertices': [
            [0, 54, 153, 234, 235, 236, 254, 269, 344, 345, 346, 348, 350, 352, 357, 369, 371, 373, 376, 378, 384, 386,
             399, 401, 404, 407, 408],
            [93, 96, 98, 99, 123, 188, 197, 200, 219, 221, 275, 276, 277, 279, 283, 289, 291, 293, 298, 300, 304, 305,
             306, 307, 309, 313, 315, 319, 321, 327, 329, 331, 405, 406],
        ],
    },

    'ropes/lasso3d_0.obj': {
        'deform_init_pos': [-0.2,3, 6],
        'deform_init_ori': [-np.pi / 2, -np.pi, np.pi],
        'deform_scale': 5,
        'deform_anchor_vertices': [
            [324, ],
            [131,],
        ],
        'deform_true_loop_vertices': [
            [338, 351, 356, 357, 358, 366, 378, 397, 399, 400, 402, 405, 406,
             409, 412, 423, 425, 428, 431, 440, 442, 455, 459, 460, 464, 469,
             472, 473, 475, 479, 485, 487, 488, 489, 490, 491, 505, 511, 519,
             520, 521, 527, 528, 533, 537, 540, 549, 551, 560, 705, 712, 721,
             723, 724, 740, 742, 748, 749, 750, 755, 764, 765, 768, 771, 784,
             785, 787, 788, 791, 796, 797, 799, 803, 807, 813, 816, 817, 823,
             824, 832, 833, 834, 849, 851, 852, 855, 856, 860, 864, 865, 866,
             876, 878, 881, 882, 883, 893, 897, 898, 903],
        ],
    },
    'button_cloth.obj': {
        'deform_init_pos': [-0.6, 1.95, 2.3],  # [-0.13, 0.16, 0.21],
        'anchor_init_pos': [-0.6, 3, 4],
        'other_anchor_init_pos': [-0.6, 3, 0.4],
        'deform_init_ori': [0, 0, 0],
        'deform_scale': 8,  # 0.5
        'deform_fixed_anchor_vertex_ids':
            [1, 120, 2, 3, 31, 78, 119, 118, 21, 104, 103, 102, 51, 90, 45, 3],
    },
    'ts_lasso_sparser.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL
        'anchor_init_pos': [-0.1206, 0.4875, 0.4857],
        'other_anchor_init_pos': [-0.0867, 0.5413, 0.4843],
        'deform_init_pos': [-0.2, 0.42, 0.48],
        'deform_init_ori': [np.pi / 2, 0, np.pi / 2],
    },
    'ts_hoop.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL
        'anchor_init_pos': [-0.12, 0.48, 0.48],
        'other_anchor_init_pos': [-0.08, 0.54, 0.48],
        'deform_init_pos': [-0.14, 0.525, 0.48],
        'deform_init_ori': [0, np.pi / 2, 0],
    },

    'ts_mask.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL
        'deform_init_pos': [0, 0.43, 0.65],
        'deform_init_ori': [0, 0, np.pi],
        'deform_scale': 0.50,
    },
}

# Info for camera rendering without debug visualizer.
# Outputs 2,3 from pybullet.getDebugVisualizerCamera()
CAM_INFO = {
    # Camera info for {cameraDistance: 11.0, cameraYaw: 140, 
    # cameraPitch: -40, cameraTargetPosition: array([0., 0., 0.])}
    'viewMatrix': (0.7660443782806396, 0.4131759703159332, -0.49240395426750183, 0.0,
                   -0.0, 0.7660444974899292, 0.6427876353263855, 0.0,
                   0.6427876949310303, -0.49240386486053467, 0.5868240594863892, 0.0,
                   -0.0, -0.0, -10.999998092651367, 1.0),
    'projectionMatrix': (1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, -1.0000200271606445, -1.0,
                         0.0, 0.0, -0.02000020071864128, 0.0)
}

"""
    # Camera info for {cameraDistance: 1.2, cameraYaw 140,
    # cameraPitch: -40, cameraTargetPosition: array([0., 0., 0.])}
    'viewMatrix': (-0.7660443782806396, -0.41317591071128845, 0.49240392446517944, 0.0,
                   0.6427876353263855, -0.4924038052558899, 0.5868241190910339, 0.0,
                   0.0, 0.7660444974899292, 0.6427875757217407, 0.0,
                   -0.0, -5.960464477539063e-08, -1.2000000476837158, 1.0),
    'projectionMatrix': (0.7499999403953552, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, -1.0000200271606445, -1.0,
                         0.0, 0.0, -0.02000020071864128, 0.0)
"""
