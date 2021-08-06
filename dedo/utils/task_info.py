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
    'HangBag': ['bags/bags_zehang/bag1_0.obj',
                'bags/bags_zehang/bag2_0.obj',
                'bags/bags_zehang/bag3_0.obj',
                ],
    'HangCloth': ['cloth/apron_0.obj',
                  'cloth/apron_1.obj',
                  'cloth/apron_2.obj',
                  'cloth/apron_3.obj',
                  'cloth/apron_4.obj',
                  'cloth/tshirt_0.obj',
                  'cloth/tshirt_1.obj',
                  'cloth/tshirt_2.obj',
                  'cloth/tshirt_3.obj',
                  'cloth/tshirt_4.obj',
                  ],
    'ButtonSimple': ['cloth/button_cloth.obj'],
    'ButtonProcedural': ['cloth/button_cloth.obj'],  # TODO wire up procedural cloths
    'Dress': [
        'bags/backpack_0.obj',
        'bags/backpack_1.obj',
        'bags/backpack_2.obj',
        'bags/backpack_3.obj',
        'bags/backpack_4.obj',
        'cloth/cardigan_0.obj',
        'cloth/cardigan_1.obj',
        'cloth/cardigan_2.obj',
        'cloth/cardigan_3.obj',
        'cloth/cardigan_4.obj',
        'bags/backpack_0_dense.obj',
        'bags/backpack_0_thick.obj',  # Failure case
    ], # TODO Backpack preset traj
    'Hoop': ['ropes/hoop3d_0.obj'], # TODO Hoop preset traj
    'Lasso': [
        'ropes/lasso3d_0.obj',
    ],
    'Mask': ['cloth/mask_0.obj',
             'cloth/mask_1.obj',
             'cloth/mask_2.obj',
             'cloth/mask_3.obj',
             'cloth/mask_4.obj',
'cloth/mask_999.obj',
             ],     # TODO Model issue & preset traj
    'Debug': ['ts/ts_backpack_resampled.obj',
              'bags/backpack_0.obj',
              'bags/backpack_0_dense.obj',
              'bags/backpack_0_thick.obj',
              'cloth/apron_2_dense.obj',
              'ts/ts_apron_twoloops.obj',
              'cloth/apron_1.obj',

              'cloth/apron_2_dense.obj',
              'cloth/apron_3_dense.obj',
              ]
}

# Information about rigid objects to load into the scene.
SCENE_INFO = {
    'debug': {
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
        'goal_pos': [[-1, 0.03, 5.3],
                     [0.6, 0.03, 5.3], ],
        'easy_target_pos': [-0.077, 2, 3.00],
    },
    'hangbag': {
        'entities': {
            'pole.obj': {
                'basePosition': [0.0, 0.0, 5],
                'baseOrientation': [np.pi / 2, 0, 0],
                'globalScaling': 0.5,
                'rgbaColor': (0.9, 0.75, 0.65, 1),
            },
            'urdf/hook_small.urdf': {
                'basePosition': [0, 1.28, 9],
                'baseOrientation': [0, 0, np.pi / 2],
                'globalScaling': 10,
            },

        },
        'goal_pos': [[0.00, 1.28, 9]],
    },
    'hangcloth': {
        'entities': {
            'urdf/hanger.urdf': {
                'basePosition': [0.0, 0, 5.5],
                'baseOrientation': [0, 0, 0],
                'globalScaling': 10.0,
            },
            'urdf/tallrod.urdf': {
                'basePosition': [0.00, 0.00, 0],
                'baseOrientation': [0, 0, 0],
                'globalScaling': 10.0,
            },

        },
        'goal_pos': [[0, 0.00, 5.7]],
    },
    'button': {
        'entities': {
            'urdf/torso.urdf': {
                'basePosition': [0.0, 0.0, 2],
                'baseOrientation': [0, 0, 0],
                'globalScaling': 12.8,
            },
            'urdf/button_fixed.urdf': {
                'basePosition': [0.85, 1.25, 3.0],
                'baseOrientation': [0, 0, np.pi / 2],
                'globalScaling': 7,
            },
            'urdf/button_fixed2.urdf': {
                'basePosition': [0.85, 1.25, 1.35],
                'baseOrientation': [0, 0, np.pi / 2],
                'globalScaling': 7,
            },
        },
        'goal_pos': [[0.85, 1.2, 1.4], [0.85, 1.2, 3.05]],

    },
    'hoop': {
        'entities': {
            'urdf/rod.urdf': {
                'basePosition': [0.00, 0.00, 0.00],
                'baseOrientation': [0, 0, 0],
                'globalScaling': 10.0,
            },
        },
        'goal_pos': [[0, 0.00, 0.5]],
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
        'goal_pos': [[0, 0.00, 0.5]],
        'easy_target_pos': [0.00, 0.00, 0.35],
    },
}

# Information about deformable objects.
DEFORM_INFO = {

    'bags/bags_zehang/bag1_0.obj': {
        'deform_init_pos': [0, 8, 1],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 0.5,
        'deform_elastic_stiffness': 50,
        'deform_bending_stiffness': 10,
        'deform_anchor_vertices': [
            [808],
            [1238],

        ],
        'deform_true_loop_vertices': [
            [267, 269, 271, 273, 275, 820, 822, 824, 832, 835, 838, 1274, 1275, 1276],
            [687, 689, 691, 693, 695, 732, 734, 736, 744, 747, 750, 1271, 1272, 1273],
        ],
        'cam_viewmat': [12, -8.0, 211, 1.2, 0.74, 2.5],
    },
    'bags/bags_zehang/bag2_0.obj': {
        'deform_init_pos': [0, 8, 1],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 0.5,
        'deform_anchor_vertices': [
            [306],
            [626],
        ],
        'deform_true_loop_vertices': [
            [267, 269, 335, 337, 339, 341, 343, 363, 365, 608, 610, 868, 870, 872, 876, 879, 882, 1276, 1278, 1314,
             1315, 1316],
            [442, 444, 446, 448, 450, 470, 472, 568, 570, 572, 574, 576, 598, 600, 748, 750, 785, 787, 789, 793, 796,
             799],
        ],
        'cam_viewmat': [12, -8.0, 211, 1.2, 0.74, 2.5],
    },
    'bags/bags_zehang/bag3_0.obj': {
        'deform_init_pos': [0, 8, 1],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 0.5,
        'anchor_init_pos': [-0.5, 2, 8],
        'other_anchor_init_pos': [0.5, 2, 8],
        'deform_anchor_vertices': [
            [610],
            [1112],

        ],
        'deform_true_loop_vertices': [
            [261, 263, 307, 309, 311, 313, 315, 317, 319, 333, 335, 348, 350, 557, 559, 572, 574, 763, 765, 767, 771,
             774, 777, 1102, 1104, 1140, 1141, 1142, 1143, 1144],
            [399, 401, 403, 405, 407, 409, 411, 425, 427, 440, 442, 502, 504, 506, 508, 510, 512, 514, 530, 532, 547,
             549, 670, 672, 701, 703, 705, 709, 712, 715],
        ],
        'cam_viewmat': [12, -8.0, 211, 1.2, 0.74, 2.5],
    },
    'bags/backpack_0.obj': {
        'deform_init_pos': [-0.2, 2, 4],
        'deform_init_ori': [-np.pi / 2, -np.pi, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [270],
            [549],
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
    'bags/backpack_0_thick.obj': {
        'deform_init_pos': [-0.2, 2, 4],
        'deform_init_ori': [-np.pi / 2, -np.pi, np.pi],
        'deform_scale': 4,
        'deform_anchor_vertices': [
            [262, ],
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
        'deform_init_pos': [-0.2, 2, 4],
        'deform_init_ori': [-np.pi / 2, -np.pi, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [261, ],
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
        'deform_init_pos': [-0.2, 2, 4],
        'deform_init_ori': [-np.pi / 2, -np.pi, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [265, ],
            [525, ],
        ],
        'deform_true_loop_vertices': [
            [20, 26, 98, 100, 118, 246, 249, 250, 251, 253, 255, 256, 257, 259, 264, 265, 266, 267, 268, 269, 270, 271,
             272, 273, 298, 299, 300, 301, 302, 303, 304, 305, 306, 308],
            [95, 185, 313, 330, 398, 412, 439, 441, 445, 481, 496, 520, 522, 525, 526, 529, 531, 532, 533, 534, 536,
             539, 540, 542, 543, 544, 545, 546, 547, 548, 549, 551, 574, 576, 578, 579, 580, 581, 582, 583, 584, 586],
        ],
    },
    'bags/backpack_4.obj': {
        'deform_init_pos': [-0.2, 2, 4],
        'deform_init_ori': [-np.pi / 2, -np.pi, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [522, ],
            [262, ],
        ],
        'deform_true_loop_vertices': [
            [19, 116, 151, 243, 246, 247, 253, 255, 257, 258, 259, 261, 262, 264, 265, 266, 267, 268, 269, 270, 271,
             272, 273, 299, 300, 301, 302, 303, 304, 305, 307, 308],
            [410, 518, 520, 521, 522, 528, 530, 532, 533, 535, 536, 537, 538, 539, 541, 542, 543, 544, 545, 546, 547,
             548, 550, 575, 577, 578, 579, 580, 581, 582, 584, 585],
        ],
    },
    'cloth/apron_0.obj': {  # This used to be apron_zehang
        'deform_init_pos': [0, 5, 5.5],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 3,
        'deform_elastic_stiffness': 50,
        'deform_bending_stiffness': 50,
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
        'deform_init_pos': [0, 5, 5.5],
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
    'cloth/apron_2.obj': {
        'deform_init_pos': [0, 5, 5.5],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [172, ],  # , 72, 73, 76, 77, 78, 79, 227, 230, 602, 604, 721, 722],
            [16, ],  # 159, 161, 162, 163, 285, 288, 520, 522, 680, 681, 763],
        ],
        'deform_true_loop_vertices': [
            [0, 2, 4, 6, 8, 9, 10, 11, 13, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 159, 162, 164,
             165, 166, 167, 168, 170, 174, 175, 177, 179, 180, 181, 182, 183, 184, 185, 188, 189, 289],
            [68, 69, 71, 104, 105, 106, 107, 129, 130, 137, 138, 139, 140, 141, 142, 143, 144, 146, 147, 148, 155, 156,
             157, 158, 240, 241, 243, 244, 264, 265, 271, 272, 273, 274, 275, 277, 278, 279, 280, 282, 283, 286, 287,
             293],
        ],
    },
    'cloth/apron_3.obj': {  # This used to be apron_zehang
        'deform_init_pos': [0, 5, 5.5],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [128],  # 10, 12, 13, 14, 15],
            [79],  # 167, 169, 171, 172, 174],
        ],
        'deform_true_loop_vertices': [
            [0, 1, 2, 31, 32, 33, 34, 35, 75, 76, 77, 79, 81, 82, 83, 123, 124, 125, 126, 128, 129, 163, 165, 166, 168,
             171, 173, 272, 273, 274, 277, 307, 308, 347, 348, 349, 350, 351, 352, 354, 355, 403, 404, 406, 467, 468,
             469, 471, 472, 473, 474, 475, 476, 478, 480, 481, 482, 483, 501, 502, 503, 504, 505, 506, 509, 516, 517,
             544, 567, 568, 584, 591, 592, 593, 594, 595, 597, 598, 604],
            [20, 25, 26, 28, 48, 49, 56, 57, 61, 62, 63, 66, 67, 70, 71, 72, 108, 111, 113, 114, 115, 117, 120, 121,
             154, 155, 157, 160, 162, 193, 206, 207, 256, 258, 262, 265, 266, 267, 268, 269, 270, 271, 336, 338, 339,
             340, 341, 342, 343, 344, 345, 346, 377, 395, 396, 397, 398, 400, 401, 402, 425, 438, 458, 459, 460, 461,
             462, 464, 465, 466, 491, 498, 499, 534, 538, 554, 564, 566, 582, 583, 589, 590, 602, 603],
        ],
    },
    'cloth/apron_4.obj': {  # This used to be apron_zehang
        'deform_init_pos': [0, 5, 5.5],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 3,
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
    'cloth/tshirt_0.obj': {
        'deform_init_pos': [0, -0.2, 8],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 3,
        'deform_elastic_stiffness': 80,
        'cam_viewmat': [11.3, -28, 193.4, -0.08, 0.29, 1.80],
        'deform_anchor_vertices': [
            [110],
            [383],

        ],
        'deform_true_loop_vertices': [
            [0, 3, 4, 5, 6, 7, 9, 13, 14, 17, 19, 21, 249, 305, 306, 307, 308, 309, 310, 312, 314, 317, 319, 321, 323,
             325, 326, 533],
        ],
    },
    'cloth/tshirt_1.obj': {
        'deform_init_pos': [0, -0.2, 8],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [111],
            [382],
        ],

        'deform_true_loop_vertices': [
            [0, 3, 5, 7, 9, 13, 14, 16, 17, 19, 21, 22, 250, 305, 307, 309, 310, 312, 314, 317, 319, 321, 323, 325, 326,
             534],
        ],
    },
    'cloth/tshirt_2.obj': {
        'deform_init_pos': [0, -0.2, 8],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [111],
            [381],
        ],

        'deform_true_loop_vertices': [
            [0, 3, 5, 7, 9, 13, 14, 17, 19, 21, 250, 305, 307, 309, 310, 312, 314, 317, 319, 321, 323, 325, 326, 533],
        ],
    },
    'cloth/tshirt_3.obj': {
        'deform_init_pos': [0, -0.2, 8],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [111],
            [381],
        ],

        'deform_true_loop_vertices': [
            [0, 3, 5, 7, 9, 13, 14, 17, 19, 21, 250, 304, 306, 308, 309, 311, 313, 316, 318, 320, 322, 324, 325, 533],
        ],
    },
    'cloth/tshirt_4.obj': {
        'deform_init_pos': [0, -0.2, 8],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 3,
        'deform_anchor_vertices': [
            [111],
            [381],
        ],

        'deform_true_loop_vertices': [
            [0, 3, 5, 7, 9, 13, 14, 17, 19, 21, 250, 305, 307, 309, 310, 312, 314, 317, 319, 321, 323, 325, 326, 534],
        ],
    },
    'cloth/cardigan_0.obj': {
        'deform_init_pos': [0, 2, 5],
        'deform_init_ori': [np.pi / 2, 0, 0],
        'deform_scale': 3,
        'deform_elastic_stiffness': 1,
        'deform_bending_stiffness': 5,
        'deform_anchor_vertices': [
            [0],
            [192],
        ],
        'deform_true_loop_vertices': [
            [200, 201, 203, 247, 248, 285, 288, 293, 296, 299, 358, 359, 360, 361, 366, 368, 369],
            [6, 8, 9, 10, 57, 59, 93, 95, 96, 99, 102, 104, 172, 173, 174, 179, 180, 182, 183],

        ],
        'cam_viewmat': [8.4, -48, 153, -0.08, 0.29, 1.80],
    },
    'cloth/cardigan_1.obj': {
        'deform_init_pos': [0, 2, 5],
        'deform_init_ori': [np.pi / 2, 0, 0],
        'deform_scale': 3,
        'deform_elastic_stiffness': 1,
        'deform_bending_stiffness': 5,
        'deform_anchor_vertices': [
            [0],
            [195],
        ],

        'deform_true_loop_vertices': [
            [200, 202, 203, 205, 249, 250, 289, 292, 297, 299, 300, 302, 362, 363, 364, 369, 371, 372],
            [6, 8, 9, 10, 56, 58, 95, 97, 98, 101, 103, 104, 106, 175, 176, 181, 182, 184, 185],

        ],
        'cam_viewmat': [8.4, -48, 153, -0.08, 0.29, 1.80],
    },
    'cloth/cardigan_2.obj': {
        'deform_init_pos': [0, 2, 5],
        'deform_init_ori': [np.pi / 2, 0, 0],
        'deform_scale': 3,
        'deform_elastic_stiffness': 1,
        'deform_bending_stiffness': 5,
        'deform_anchor_vertices': [
            [0],
            [214],
        ],

        'deform_true_loop_vertices': [
            [221, 222, 224, 267, 268, 308, 316, 320, 324, 325, 326, 386, 387, 388, 393, 395, 396],
            [53, 54, 55, 56, 60, 62, 109, 114, 115, 119, 121, 194, 195, 201],

        ],
        'cam_viewmat': [8.4, -48, 153, -0.08, 0.29, 1.80],
    },
    'cloth/cardigan_3.obj': {
        'deform_init_pos': [0, 2, 5],
        'deform_init_ori': [np.pi / 2, 0, 0],
        'deform_scale': 3,
        'deform_elastic_stiffness': 1,
        'deform_bending_stiffness': 5,
        'deform_anchor_vertices': [
            [0],
            [198],
        ],

        'deform_true_loop_vertices': [
            [206, 207, 209, 254, 255, 305, 310, 314, 315, 379, 380, 385, 387, 388],
            [8, 9, 10, 57, 59, 97, 104, 108, 109, 111, 112, 165, 167, 180, 188, 190, 191, 192, 193],
        ],
        'cam_viewmat': [8.4, -48, 153, -0.08, 0.29, 1.80],
    },
    'cloth/cardigan_4.obj': {
        'deform_init_pos': [0, 2, 5],
        'deform_init_ori': [np.pi / 2, 0, 0],
        'deform_scale': 3,
        'deform_elastic_stiffness': 1,
        'deform_bending_stiffness': 5,
        'deform_anchor_vertices': [
            [0],
            [200],
        ],

        'deform_true_loop_vertices': [
            [206, 208, 209, 211, 255, 256, 296, 304, 305, 307, 308, 310, 370, 371, 372, 377, 379, 380],
            [8, 9, 10, 56, 58, 96, 103, 107, 111, 112, 113, 182, 183, 184, 189, 191, 192],

        ],
        'cam_viewmat': [8.4, -48, 153, -0.08, 0.29, 1.80],
    },
    'cloth/mask_0.obj': {
        'deform_init_pos': [-0.2, 2, 6.2],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 1,

        'deform_anchor_vertices': [
            [354, ],
            [288, ]
        ],
        'deform_true_loop_vertices': [
            [1, 56, 82, 232, 233, 258, 338, 339, 341, 343, 345, 347, 349, 357, 362, 365, 367, 370, 371, 372, 373, 374,
             375, 379, 387, 389, 393, 394, 395, 396, 397, 399, 406, 408],
            [117, 193, 194, 198, 212, 215, 216, 217, 268, 269, 271, 273, 275, 277, 278, 279, 280, 281, 290, 296, 297,
             299, 301, 303, 305, 306, 307, 308, 309, 314, 322, 402, 404]
        ],
        'cam_viewmat': [3.80, -23.6, 177.4, -0.42, 0.06, 5.00],
    },
    'cloth/mask_1.obj': {
        'deform_init_pos': [-0.2, 2, 6.2],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 1,
        'deform_elastic_stiffness': 10,
        'deform_bending_stiffness': 10,
        'deform_anchor_vertices': [
            [355, ],
            [288, ],
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
            [354, ],
            [288, ],
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
            [358, ],
            [289, ],
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
            [367, ],
            [294, ],
        ],
        'deform_true_loop_vertices': [
            [0, 54, 153, 234, 235, 236, 254, 269, 344, 345, 346, 348, 350, 352, 357, 369, 371, 373, 376, 378, 384, 386,
             399, 401, 404, 407, 408],
            [93, 96, 98, 99, 123, 188, 197, 200, 219, 221, 275, 276, 277, 279, 283, 289, 291, 293, 298, 300, 304, 305,
             306, 307, 309, 313, 315, 319, 321, 327, 329, 331, 405, 406],
        ],
    },
    'cloth/mask_999.obj': {
        'deform_init_pos': [-0.2, 2, 6.2],
        'deform_init_ori': [np.pi / 2, 0, np.pi],
        'deform_scale': 1,
        'deform_anchor_vertices': [
            [384, ],
            [321, ],
        ],

    },
    'ropes/lasso3d_0.obj': {
        'deform_init_pos': [-0.2, 3, 6],
        'deform_init_ori': [-np.pi / 2, -np.pi, np.pi],
        'deform_scale': 5,
        'deform_anchor_vertices': [
            [378, ],
            [131, ],
        ],
        'deform_elastic_stiffness': 100,
        'deform_bending_stiffness': 10,
        'deform_true_loop_vertices': [
            [356, 358, 359, 381, 400, 401, 402, 403, 405, 406, 407, 409, 413,
             419, 424, 427, 428, 431, 436, 440, 453, 455, 458, 460, 468, 473,
             474, 475, 476, 480, 484, 487, 489, 495, 500, 506, 508, 511, 516,
             517, 520, 522, 524, 528, 533, 535, 536, 538, 549, 551, 560, 720,
             721, 724, 733, 736, 737, 738, 740, 749, 751, 753, 755, 756, 764,
             766, 771, 772, 774, 785, 787, 791, 799, 800, 802, 807, 813, 817,
             818, 820, 823, 824, 833, 835, 844, 852, 856, 861, 863, 864, 867,
             868, 871, 877, 878, 880, 881, 886, 892, 897],
        ],
        'cam_viewmat': [7, -45, 267, -0.08, 0.32, 2.7],
    },
    'ropes/hoop3d_0.obj': {
        'deform_init_pos': [-0.2, 3, 6],
        'deform_init_ori': [-np.pi / 2, -np.pi, np.pi],
        'deform_scale': 5,
        'deform_anchor_vertices': [
            [378, ],
            [131, ],
        ],
        'deform_elastic_stiffness': 1,
        'deform_bending_stiffness': 1,
        'deform_true_loop_vertices': [
            [356, 358, 359, 381, 400, 401, 402, 403, 405, 406, 407, 409, 413,
             419, 424, 427, 428, 431, 436, 440, 453, 455, 458, 460, 468, 473,
             474, 475, 476, 480, 484, 487, 489, 495, 500, 506, 508, 511, 516,
             517, 520, 522, 524, 528, 533, 535, 536, 538, 549, 551, 560, 720,
             721, 724, 733, 736, 737, 738, 740, 749, 751, 753, 755, 756, 764,
             766, 771, 772, 774, 785, 787, 791, 799, 800, 802, 807, 813, 817,
             818, 820, 823, 824, 833, 835, 844, 852, 856, 861, 863, 864, 867,
             868, 871, 877, 878, 880, 881, 886, 892, 897],
        ],
        'cam_viewmat': [7, -45, 267, -0.08, 0.32, 2.7],
    },
    'cloth/button_cloth.obj': {
        'deform_init_pos': [-0.0, 1.95, 2.3],  # [-0.13, 0.16, 0.21],
        'anchor_init_pos': [-0.6, 3, 4],
        'other_anchor_init_pos': [-0.6, 3, 0.4],
        'deform_init_ori': [0, 0, 0],
        'deform_scale': 8,  # 0.5
        'deform_elastic_stiffness': 10,
        'deform_bending_stiffness': 10,
        'deform_fixed_anchor_vertex_ids':
            [0, 1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54,
             56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78],

        'deform_true_loop_vertices': [
            [647, 648, 649, 650, 687, 688, 723, 724, 760, 762],
            [667, 668, 669, 670, 705, 706, 741, 742, 782, 784],
        ],
        'cam_viewmat': [6, -16, 176.2, -0.08, 0.29, 1.80],
    },
    'ts/ts_lasso_sparser.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL
        'anchor_init_pos': [-0.1206, 0.4875, 0.4857],
        'other_anchor_init_pos': [-0.0867, 0.5413, 0.4843],
        'deform_init_pos': [-0.2, 0.42, 0.48],
        'deform_init_ori': [np.pi / 2, 0, np.pi / 2],
    },
    'ts/ts_hoop.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL
        'anchor_init_pos': [-0.12, 0.48, 0.48],
        'other_anchor_init_pos': [-0.08, 0.54, 0.48],
        'deform_init_pos': [-0.14, 0.525, 0.48],
        'deform_init_ori': [0, np.pi / 2, 0],
    },
    'ts/ts_apron_twoloops.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL
        'anchor_init_pos': [-0.04, 0.35, 0.75],
        'other_anchor_init_pos': [0.04, 0.35, 0.75],
        'deform_init_pos': [0, 0.8, 6.5],  # [0, 0.05, 0.47],
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
    'ts/ts_purse_bag_resampled.obj': {  # TODO: REMOVE - PROPRIETARY MESH
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
    'ts/ts_backpack_resampled.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL

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
    'ts/ts_mask.obj': {  # TODO: REMOVE - PROPRIETARY MESH MODEL
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
    # 'viewMatrix': (0.22495108842849731, -0.371303528547287, 0.9008499979972839, 0.0, 0.0, 0.9245460629463196, 0.3810703456401825, 0.0, -0.9743700623512268, -0.0857221856713295, 0.20797763764858246, 0.0, 1.771862268447876, 0.3927137553691864, -7.591782093048096, 1.0),
    'projectionMatrix': (
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0)
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
