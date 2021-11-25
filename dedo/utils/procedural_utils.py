"""
Utilities for generating procedural cloth.

Procedural generation works as follows:
1. Generate a mesh, randomly carves out a square hole.
2. If hole > 1, also checks for overlapping of two holes. If overlap, randomly choose a new hole position. Repeat until no overlap found.
3. Saves hollowed mesh into .obj file in the /tmp/ directory.



Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@yonkshi, @jackson

"""

import os
import numpy as np
from matplotlib import pyplot as plt


def gen_procedural_hang_cloth(args, preset_obj_name, deform_info_dict):
    '''
    Hang cloth procedrual generator. Generates a cloth of random size and places random holes inside. Checks for overlap
    :param args:
    :param preset_obj_name:
    :param deform_info_dict:
    :return:
    '''
    num_holes = args.num_holes
    node_density = args.node_density

    # cloth dimensions
    width_range = [0.5, 2.0]
    height_range = [0.5, 2.0]
    w = np.random.uniform(*width_range) / 2
    h = np.random.uniform(*height_range) / 2

    # hole generation
    constraints = {}
    constraints['x_range'] = (2, args.node_density - 2)
    constraints['y_range'] = (2, args.node_density - 2)
    constraints['width_range'] = (1, int(round(node_density * 0.3)))
    constraints['height_range'] = (1, int(round(node_density * 0.3)))
    holes = try_gen_holes(args.node_density, num_holes, constraints)

    # cloth save path
    rand_id = np.random.uniform(1e7)
    args.deform_obj = f'/tmp/procedural_hang{rand_id}.obj'
    data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
    savepath = os.path.join(data_path, args.deform_obj)

    cloth_obj_path, cloth_anchor_indices, gt_loop_vertices = create_cloth_obj(
        # min_point=[0.00, -0.3, -0.3], max_point=[0.00, 0.3, 0.3],
        min_point=[0.00, -w, -h], max_point=[0.00, w, h],
        # min_point=[0,0.42,0.48], max_point=[0.2,0.45,0.52],
        node_density=args.node_density,
        holes=holes,
        data_path=savepath,
    )
    if args.deform_obj not in deform_info_dict.keys():
        deform_info_dict[args.deform_obj] = deform_info_dict[preset_obj_name].copy()

    deform_info_dict[args.deform_obj]['deform_anchor_vertices'] = list(cloth_anchor_indices)
    deform_info_dict[args.deform_obj]['deform_true_loop_vertices'] = gt_loop_vertices

    return args.deform_obj


def gen_procedural_button_cloth(args, preset_obj_name, deform_info_dict):
    '''
    Button cloth procedural generator, generates one or two holes for the button cloth.
    :param args: args object
    :param preset_obj_name:
    :param deform_info_dict:
    :return:
    '''
    num_holes = args.num_holes

    # These are fine tuned ranges.
    width_range = [2, 3]
    height_range = [2, 3.5]
    w = np.random.uniform(*width_range)
    h = np.random.uniform(*height_range)

    # Dynamic node density based on fabric size.
    node_density = int(round((w + h) / 2 * 25 / 3))

    # Hole generation.
    constraints = {}
    constraints['x_range'] = (2, 7)  # (2, args.node_density - 2)
    constraints['y_range'] = (2, node_density - 2)  # (2, args.node_density - 2)
    constraints['width_range'] = (1, 2)  # (1, int(round(node_density*0.3)))
    constraints['height_range'] = (1, 2)  # (1, int(round(node_density*0.3)))
    holes = try_gen_holes(node_density, num_holes, constraints)

    # Make temporary obj file path.
    rand_id = np.random.uniform(1e7)
    args.deform_obj = f'/tmp/procedural_hang{rand_id}.obj'
    data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
    savepath = os.path.join(data_path, args.deform_obj)

    node_coords = []
    cloth_obj_path, cloth_anchor_indices, gt_loop_vertices, fixed_anchors = create_cloth_obj(
        min_point=[0.00, -w, -h / 2], max_point=[0.00, 0, h / 2],
        node_density=node_density,
        holes=holes,
        data_path=savepath,
        gen_fixed_anchors=True,
        node_coords=node_coords,
    )
    if args.deform_obj not in deform_info_dict.keys():
        deform_info_dict[args.deform_obj] = deform_info_dict[preset_obj_name].copy()

    # Find center of holes (coords).
    node_coords = np.array(node_coords)
    hole_centers = [np.mean(node_coords[gt_loop], axis=0) for gt_loop in gt_loop_vertices]

    deform_info_dict[args.deform_obj]['deform_anchor_vertices'] = list(cloth_anchor_indices)
    deform_info_dict[args.deform_obj]['deform_fixed_anchor_vertex_ids'] = fixed_anchors
    deform_info_dict[args.deform_obj]['deform_true_loop_vertices'] = gt_loop_vertices
    # print('deform_true_loop_vertices', gt_loop_vertices)
    # print('deform_anchor_vertices', cloth_anchor_indices)
    # print('deform_fixed_anchor_vertex_ids', fixed_anchors)

    return args.deform_obj, hole_centers


def overlap_constraint(A, B):
    """ Make sure two holes are not overlapping and have enough vertices between
    to create faces in between."""
    mb = 3  # minimum boundary
    lr = A['x0'] < B['x1'] + mb
    rl = A['x1'] > B['x0'] - mb
    tb = A['y0'] < B['y1'] + mb
    bt = A['y1'] > B['y0'] - mb
    return not (lr and rl and tb and bt)


def boundary_constraint(node_density, hole):
    """ Each hole should be at least 2 vertices away from the edge,
    so the edge could form a face."""

    for key, val in hole.items():
        # Setting the minimum boundary between edge and hole.
        # Min two vertices away so edge could form face.
        upper_bound = node_density - 3  # 0-index node_density-1
        lower_bound = 2
        if hole[key] >= upper_bound or hole[key] <= lower_bound:
            return False

    return True


def gen_random_hole(node_density, dim_constraints):
    """Generates a hole, minding existing hole so they don't overlap."""
    hole = {}

    x_range = dim_constraints['x_range']
    y_range = dim_constraints['y_range']
    width_range = dim_constraints['width_range']
    height_range = dim_constraints['height_range']

    # Infer actual coordinates from constraints and dimensions
    hole['x0'] = np.random.randint(*x_range)
    hole['x1'] = hole['x0'] + np.random.randint(*width_range)

    hole['y0'] = np.random.randint(*y_range)
    hole['y1'] = hole['y0'] + np.random.randint(*height_range)

    return hole


def try_gen_holes(node_density, num_holes, constraints):
    '''
    Monte Carlo method for placing holes in a cloth, checks for overlap so they don't overlap
    :param node_density:
    :param num_holes:
    :param constraints:
    :return:
    '''
    for i in range(1000):  # 1000 MC
        if num_holes == 2:
            holeA = gen_random_hole(node_density, constraints)
            holeB = gen_random_hole(node_density, constraints)
            if boundary_constraint(node_density, holeA) and boundary_constraint(
                    node_density, holeB) and overlap_constraint(holeA, holeB):
                return [holeA, holeB]  # satisfies boundary constraints
        elif num_holes == 1:
            hole = gen_random_hole(node_density, constraints)
            if boundary_constraint(node_density, hole):
                return [hole]
        else:
            raise NotImplemented('num_holes > 2 is not implemented yet')
    print('Failed to generate hole according to constraint')


def create_cloth_obj(min_point, max_point, node_density,
                     holes, data_path,
                     gen_fixed_anchors=False,
                     node_coords=[]):
    '''
    Core procedural generator code
    :param min_point: bottom,left corner
    :param max_point: top, right corner
    :param node_density: density of the nodes (per cm)
    :param holes: list of holes to be generated
    :param data_path: temporary path for storing the cloth obj file
    :param gen_fixed_anchors: annotate fixed anchors if true (For buttoning
    :param node_coords:
    :return:
    '''

    def validate_and_integerize(hole):
        # Convert ratio to aboslute.
        for key, val in hole.items():
            if isinstance(val, float):
                hole[key] = int(round(val * node_density))
                # Setting the minimum boundary between edge and hole.
                # Min two vertices away so edge could form face
                if hole[key] >= node_density - 2:
                    hole[key] = node_density - 3
                elif hole[key] <= 1:
                    hole[key] = 2
            assert isinstance(hole[key], int), \
                f'{hole} {key} must be either an int or a float'
        assert len(min_point) == len(max_point) == 3, \
            'min_point and max_point must both have length 3'

    holes_range = []
    holes_fp = []

    for hole in holes:
        holes_fp.append(hole.copy())
        validate_and_integerize(hole)
        # Create a 2d range of hole coords.
        x_range = np.arange(hole['x0'], hole['x1'] + 1)
        y_range = np.arange(hole['y0'], hole['y1'] + 1, )
        xx, yy = np.meshgrid(x_range, y_range)
        # (x1, x1 ..., x2, x2 ...)
        xx = xx.flatten()
        # (y1, y2 ..., y1, y2 ...)
        yy = yy.flatten()
        # ((x1, y1), (x1, y2), ... (y1, y2), (x2, y2)
        r = list(zip(xx, yy))
        holes_range.append(r)
    if data_path.endswith('.obj'):  # If data_path is already a file path
        obj_path = data_path
    else:
        # Check if file already exists
        if not os.path.exists(os.path.join(data_path, "generated_cloth")):
            os.makedirs(os.path.join(data_path, "generated_cloth"))
        fnm = "cloth_" + str(node_density) + "_" + str(hole[0]['x']) + "_" + \
              str(hole[0]['y']) + "_" + str(hole[0]['x']) + "_" + \
              str(min_point[0]) + "_" + str(min_point[1]) + "_" + \
              str(min_point[2]) + "_" + str(max_point[0]) + "_" + \
              str(max_point[1]) + "_" + str(max_point[2]) + ".obj"
        obj_path = os.path.join(
            data_path, "generated_cloth", fnm)

        if os.path.isfile(obj_path):
            print("Cloth obj file already exists, skipping mesh creation.")
            anchor_indices = []
            with open(obj_path, 'r') as f:
                words = f.readline().split()
                anchor_indices = (int(words[1]), int(words[2]))
                f.close()
            return obj_path, anchor_indices

    def node_in_hole(x, y):
        # Check all holes
        for hole in holes_range:
            if (x, y) in hole: return True
        return False

    def which_hole(x, y):
        # Check all holes
        for i, hole in enumerate(holes_range):
            if (x, y) in hole: return i

        return None

    # Construct the list of nodes [(x1, y1), (x2, y2), ... , (xn, yn)]
    nodes = []
    gt_loop_vertices = [[] for _ in range(len(holes))]
    for x in range(node_density):
        for y in range(node_density):
            if not node_in_hole(x, y):
                nodes.append((x, y))

                # Get a hole's boundary
                # get boundary nodes, used for topo_latents
                for neighbour_pos in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
                    if node_in_hole(*neighbour_pos):
                        hole_id = which_hole(*neighbour_pos)
                        vertex_id = nodes.index((x, y))
                        gt_loop_vertices[hole_id].append(vertex_id)
                        break

    # Construct the list of faces (clockwise around triangles) [(i, i2, i3), ...]
    faces = []
    for x in range(node_density - 1):
        for y in range(node_density - 1):
            # Skip quads where not all nodes are kept
            if (node_in_hole(x, y) or node_in_hole(x + 1, y) or
                    node_in_hole(x, y + 1) or node_in_hole(x + 1, y + 1)):
                continue
            # Warning: each of the nodes.index() calls scales linearly with the
            # length of nodes consider switching to a map instead?
            faces.append((
                nodes.index((x, y)) + 1,
                nodes.index((x, y + 1)) + 1,
                nodes.index((x + 1, y)) + 1))
            faces.append((
                nodes.index((x, y + 1)) + 1,
                nodes.index((x + 1, y + 1)) + 1,
                nodes.index((x + 1, y)) + 1))

    # Helper to linearly interpolate between two 3d points using cloth coords
    def lerp(pt1, pt2, percents):
        return (pt1[0] + (pt2[0] - pt1[0]) * percents[0],
                pt1[1] + (pt2[1] - pt1[1]) * percents[0],
                pt1[2] + (pt2[2] - pt1[2]) * percents[1])

    def get_neighbour_indices(center, size):
        left = max(0, center[0] - size)
        right = min(node_density, center[0] + size)
        bottom = max(0, center[1] - size)
        top = min(node_density, center[1] + size)

        indices = []

        for i in range(left, right):
            for j in range(bottom, top):
                if (i, j) in nodes:  # could be that hole is too big
                    indices.append(nodes.index((i, j)))
        return indices

    idx_left = (0, 0)
    idx_right = (0, node_density - 1)
    anchor_index = nodes.index(idx_right)
    anchor_index2 = nodes.index(idx_left)

    with open(obj_path, 'w') as f:
        # f.write("# %d %d anchor index\n" % (anchor_index, anchor_index2))
        for n in nodes:
            coord = lerp(min_point, max_point, (n[0] / (node_density - 1),
                                                n[1] / (node_density - 1)))
            node_coords.append(coord)
            f.write("v %.4f %.4f %.4f\n" % coord)
        for tri in faces:
            f.write("f %d %d %d\n" % tri)
        f.close()

    if gen_fixed_anchors:
        # pinned nodes are from (1, 1) to (node_density-1, 1)
        node_y = np.arange(0, node_density)
        fixed_anchors = [nodes.index((node_density - 1, i)) for i in node_y]
        return obj_path, ([anchor_index], [anchor_index2]), \
               gt_loop_vertices, fixed_anchors

    return obj_path, ([anchor_index], [anchor_index2]), gt_loop_vertices


def plotter(hole1, hole2, type):
    plt.figure()

    def plot_one(h):
        pts = np.array([[h['x0'], h['y0']], [h['x0'], h['y1']],
                        [h['x1'], h['y0']], [h['x1'], h['y1']]])
        plt.scatter(pts[:, 0], pts[:, 1])

    plot_one(hole1)
    plot_one(hole2)
    plt.savefig(f'/tmp/debug_procedural_cloth_{type}.png')
