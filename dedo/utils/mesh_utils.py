#
# Mesh utilities for deform sim in PyBullet.
#
# @contactrika
#
import numpy as np
import pybullet


def get_mesh_data(sim, deform_id):
    """Returns num mesh vertices and vertex positions."""
    kwargs = {}
    if hasattr(pybullet, 'MESH_DATA_SIMULATION_MESH'):
        kwargs['flags'] = pybullet.MESH_DATA_SIMULATION_MESH
    num_verts, mesh_vert_positions = sim.getMeshData(deform_id, **kwargs)
    return num_verts, mesh_vert_positions


def print_mesh_data(sim, deform_id, deform_anchored_vertex_ids, step):
    """Prints mesh vertex IDs."""
    num_verts, mesh_vert_positions = get_mesh_data(sim, deform_id)
    print('Step', step, 'anchored mesh locations:')
    num_anchored = len(deform_anchored_vertex_ids)
    for i in range(num_anchored):
        for v in deform_anchored_vertex_ids[i]:
            print(np.array(mesh_vert_positions[v]))
