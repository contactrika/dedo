#
# Utilities for the buttoning task.
#
# @jackson, @pyshi
#
import os
import numpy as np
from matplotlib import pyplot as plt


'''
    Creates a .obj file containing a button loop mesh with the
    given parameters if it doesn't already exist. If the .obj file already
    exists, this method returns the path to the existing file. Generated meshes 
    are constructed such that the first (density) nodes correspond to the 
    leftmost edge which will be pinned to the "torso".
arguments:
    min_point (list or tuple of 3 numbers) --> 
        the (min X, min Y, min Z) point of the cloth
    max_point (list or tuple of 3 numbers) --> 
        the (max X, max Y, max Z) point of the cloth
    node_density (int) --> the number of nodes along the edge of the cloth
    
    holes (int) --> the 
    
    
    hole_width (int or float) --> the width of the hole as either an integer 
        (number of edges cut out) or a float (percentage of cloth width cut out)
    hole_height (int or float) --> the height of the hole as either an integer
        (number of edges cut out) or a float (percentage of cloth height cut out)
    hole_offset (int or float) --> thickness of the loop to be wrapped around 
        the button (distance of the hole from the right-hand side of the cloth)
returns:
    a tuple containing...
    obj_path --> path to the obj file for use when loading the cloth softbody
    anchor_index --> index of the node to be anchored (currently top right)
'''
def create_cloth_obj(min_point, max_point, node_density,
                     holes, data_path,
                     use_hanging_anchors=False,
                     __ptr__hole_boundary_nodes_idx=None,
                     corners_ptr=[],
                     __ptr__hole_corners_idx=[],
                     ):
    assert(len(holes) > 0), 'Specify one or more holes'

    def validate_and_integerize(hole):
        for key, val in hole.items():
            if isinstance(val, float):
                hole[key] = int(round(val * node_density))
            assert isinstance(hole[key], int), \
                f"{hole} {key} must be either an int or a float"

        assert len(min_point) == len(max_point) == 3, \
            "min_point and max_point must both have length 3"

    holes_range = []
    holes_fp = []

    for hole in holes:
        holes_fp.append(hole.copy())
        validate_and_integerize(hole)

        # creates a 2d range of hole coords
        x_range = np.arange(hole['x0']+1, hole['x1'])
        y_range = np.arange(hole['y0']+1, hole['y1'])
        xx, yy = np.meshgrid(x_range, y_range)
        # (x1, x1 ..., x2, x2 ...)
        xx = xx.flatten()
        # (y1, y2 ..., y1, y2 ...)
        yy = yy.flatten()
        # ((x1, y1), (x1, y2), ... (y1, y2), (x2, y2)
        r = list(zip(xx,yy))
        holes_range.append(r)
    if data_path.endswith('.obj'):  # if data_path is already a file path
        obj_path = data_path
    else:
        # Check if file already exists
        if not os.path.exists(os.path.join(data_path, "generated_cloth")):
            os.makedirs(os.path.join(data_path, "generated_cloth"))

        obj_path = os.path.join(
            data_path, "generated_cloth", "cloth_" + str(node_density) + "_" \
            + str(hole[0]['x']) + "_" + str(hole[0]['y']) + "_" \
            + str(hole[0]['x']) + "_" + str(min_point[0]) + "_" \
            + str(min_point[1]) + "_" + str(min_point[2]) + "_" \
            + str(max_point[0]) + "_" + str(max_point[1]) + "_" \
            + str(max_point[2]) + ".obj")

        if os.path.isfile(obj_path):
            print("Cloth obj file already exists, skipping mesh creation.")
            anchor_indices = []
            with open(obj_path, 'r') as f:
                words = f.readline().split()
                anchor_indices = (int(words[1]), int(words[2]))
                f.close()
            return obj_path, anchor_indices

    def node_in_hole(x,y):
        # Check all holes
        for hole in holes_range:
            if (x,y) in hole: return True
        return False

    def which_hole(x,y):
        # Check all holes
        for i , hole in enumerate(holes_range):
            if (x, y) in hole: return i

        return None

    # Construct the list of nodes [(x1, y1), (x2, y2), ... , (xn, yn)]
    nodes = []
    for x in range(node_density):
        for y in range(node_density):
            if not node_in_hole(x, y):
                nodes.append((x, y))

                # Get a hole's boundary
                if __ptr__hole_boundary_nodes_idx is not None:
                    # get boundary nodes, used for topo_latents
                    for neighbour_pos in ((x+1, y), (x, y+1),
                                          (x-1, y), (x, y-1)):
                        if node_in_hole(*neighbour_pos):
                            hole_id = which_hole(*neighbour_pos)
                            __ptr__hole_boundary_nodes_idx[hole_id].append(
                                nodes.index((x, y)))

    # Construct the list of faces
    # (clockwise around triangles) [(i, i2, i3), ...]
    faces = []
    for x in range(node_density - 1):
        for y in range(node_density - 1):
            # Skip quads where not all nodes are kept
            if (node_in_hole(x, y) or node_in_hole(x+1, y) or
                node_in_hole(x, y+1) or node_in_hole(x+1, y+1)): continue
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
        left = max(0, center[0]-size)
        right = min(node_density, center[0]+size)
        bottom = max(0, center[1]-size)
        top = min(node_density, center[1] + size)

        indices = []

        for i in range(left, right):
            for j in range(bottom, top):
                if (i,j) in nodes: # could be that hole is too big
                    indices.append(nodes.index((i, j)))
        return indices

    idx_right = (node_density-1, node_density-1)
    idx_left = (node_density-1, 1)
    anchor_index = nodes.index(idx_right)
    anchor_index2 = nodes.index(idx_left)
    if use_hanging_anchors:
        idx_left = (1, node_density - 1)
        idx_right = (node_density - 1, node_density - 1)
        anchor_index = get_neighbour_indices(idx_right, 3)
        anchor_index2 = get_neighbour_indices(idx_left, 3)

    # Build hole corners label
    if type(__ptr__hole_corners_idx) is list:
        for hole in holes:
            try:
                c0 = nodes.index((hole['x0'],hole['y0']))
                c1 = nodes.index((hole['x0'],hole['y1']))
                c2 = nodes.index((hole['x1'], hole['y1']))
                c3 = nodes.index((hole['x1'], hole['y0']))
            except ValueError as e:
                plotter(holes_fp[0], holes_fp[1], 'fp')
                plotter(holes[0], holes[1], 'int')
                print(e)
            __ptr__hole_corners_idx.append([c0, c1, c2, c3])

    with open(obj_path, 'w') as f:
        # f.write("# %d %d anchor index\n" % (anchor_index, anchor_index2))
        for n in nodes:
            f.write("v %.4f %.4f %.4f\n" % lerp(
                min_point, max_point,
                (n[0]/(node_density-1), n[1]/(node_density-1))))
        for tri in faces:
            f.write("f %d %d %d\n" % tri)
        f.close()

    return obj_path, (anchor_index, anchor_index2)


def plotter(hole1,hole2, type):
    plt.figure()

    def plot_one(h):
        pts = np.array([[h['x0'], h['y0']],[h['x0'], h['y1']],[h['x1'],
                        h['y0']],[h['x1'], h['y1']]])
        plt.scatter(pts[:,0], pts[:,1])

    plot_one(hole1)
    plot_one(hole2)
    plt.savefig(f'/tmp/debug_procedural_cloth_{type}.png')
