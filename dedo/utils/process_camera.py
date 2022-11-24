# Process depth and RGB camera input from PyBullet.
#
# @contactrika
# 
# Updated pcd generation is vectorized & camera config now loads from json
#
#  @edwin-pan

import os
import sys
import math
import time
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=4, linewidth=150, threshold=np.inf, suppress=True)

import pybullet
from typing import Optional

from .camera_utils import cameraConfig

def assert_close(ars, ars0):
    for ar, ar0 in zip(ars, ars0):
        assert(np.linalg.norm(np.array(ar)-np.array(ar0))<1e-6)


class ProcessCamera():
    # In non-GUI mode we will render without X11 context but *with* GPU accel.
    # examples/pybullet/examples/testrender_egl.py
    # Note: use alpha=1 (in rgba), otherwise depth readings are not good
    # Using defaults from PyBullet.
    # See examples/pybullet/examples/pointCloudFromCameraImage.py
    PYBULLET_FAR_PLANE = 10000
    PYBULLET_NEAR_VAL = 0.01
    PYBULLET_FAR_VAL = 1000.0
    # Don't change default CAM_DIST and CAM_TGT without updating CAM_VALS.
    CAM_DIST = 0.85
    CAM_TGT = np.array([0.35, 0, 0])
    # Yaws and pitches are set to get the front and side view of the scene.
    # ATTENTION: CAM_VALS *HAVE* TO BE UPDATED IF CAM_YAWS, CAM_PITCHES,
    # CAM_DIST, or CAM_TGT_POS is changed.
    CAM_YAWS = list(range(-30,211,40)) # [-30, 0, ..., 170, 210]
    CAM_PITCHES = [-70, -10, -65, -40, -10, -25, -60]   

    def make_pcd(obs:np.ndarray, config:cameraConfig,
                 segment_mask:np.ndarray=None, object_ids:list=None) -> np.ndarray:
        """ Convert screen_space coordinates (u,v,depth) to world_space coordinates (x,y,z) using basic pinhole projection.

            Args:
                obs (np.ndarray): 2D array of raw depth observations (W,H) 
                config (cameraConfig): Camera configuration information
                segment_mask (np.ndarray, optional): Segmentation mask for returning 
                    the point cloud of only specific objects
                object_ids (list, optional): Pybullet object ids for the objects that 
                    will have their pointcloud returned.

            Returns:
                    {
                        'pcd': point cloud data,
                        'ids': object_id mask (note, id=0 reserved for softbodies)
                    }

            Raises:
                ValueError: If `object_ids` specified, then `segment_mask` should be too.

        """

        # construct homogeneous screen space coordinates
        indices = np.indices(obs.shape).reshape(2, -1)
        uv_depth = obs[indices[1], indices[0]]
        input_homogeneous = np.vstack([indices, uv_depth, np.ones(len(uv_depth))])

        # Verify inputs
        if object_ids is not None:
            if segment_mask is None:
                raise ValueError("Segmentation mask (segment_mask) should be specified if object ids are provided")
            else:
                # remove points that aren't members of the objects in object_ids
                object_id_mask = segment_mask[indices[1], indices[0]]

                mask = np.zeros_like(object_id_mask)
                for object_id in object_ids:
                    mask = np.logical_or(mask, object_id_mask==object_id)

                input_homogeneous = input_homogeneous.T[mask].T
                ids = object_id_mask[mask]
        else:
            ids = np.zeros((pcd.shape[0], 1))


        # Get frustum and camera constants
        far_plane = ProcessCamera.PYBULLET_FAR_PLANE
        near_val = ProcessCamera.PYBULLET_NEAR_VAL
        far_val = ProcessCamera.PYBULLET_FAR_VAL
        cam_pos = config.cam_tgt - config.cam_dist*np.array(config.cam_forward)
        ray_forward = (config.cam_tgt - cam_pos)
        ray_forward *= far_plane/np.linalg.norm(ray_forward)
        horizontal = np.array(config.cam_horiz); vertical = np.array(config.cam_vert)

        height, width = obs.shape

        # convert to world space coordinates 
        # -- ortho
        constant = vertical/2-horizontal/2
        A_ortho = np.vstack([horizontal/width, -vertical/height, np.zeros(3), constant]).T
        ortho_v = A_ortho@input_homogeneous
        # -- vec
        vec_v = (ortho_v.T + ray_forward).T
        # -- alpha
        alpha_v = np.arctan(np.linalg.norm(ortho_v, axis=0)/far_plane)
        # -- depth 
        depth_v = far_val*near_val/(far_val-(far_val-near_val)*input_homogeneous[2])
        # -- res
        res_v = (depth_v/np.cos(alpha_v))*(vec_v/np.linalg.norm(vec_v, axis=0))
        # -- pcd
        pcd = cam_pos + res_v.T
        
        return pcd, ids


    @staticmethod
    def draw_point_clouds_from_file(fname_pfx, max_num_pts=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.axes3d import Axes3D
        draw_every = 1; max_steps = 10000
        cmap = plt.cm.jet
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        for step in range(0,max_steps+1,draw_every):
            fname = fname_pfx+'_step'+str(step)+'.npz'
            if step==0:
                print('Loading', fname)
                assert(os.path.exists(fname))  # assert npz for step=0 exists
            if not os.path.exists(fname): continue
            data = np.load(fname, allow_pickle=True)
            assert('ptcloud' in data.keys())
            assert('tracking_ids' in data.keys())
            pts  = data['ptcloud']
            ids  = data['tracking_ids']
            if step==0: print('ptcloud', pts.shape, 'tracking_ids', ids.shape)
            if pts.shape[0]==0: continue  # no pts to plot
            print('Draw step {:d} num_pts {:d} {:s}'.format(
                step, pts.shape[0],
                '' if max_num_pts is None else 'keeping '+str(max_num_pts)))
            if max_num_pts is not None:
                perm = np.random.permutation(pts.shape[0])[0:max_num_pts]
                pts = pts[perm]; ids = ids[perm]
            # Plot points using matplotlib Axes3D
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], marker='.', s=2.0,
                       c=ids[:,0], cmap=cmap)
            ax.set_xlim(-0.4,1.0); ax.set_ylim(-0.6,0.6); ax.set_zlim(0,0.8)
            ax.view_init(10, -25)
            plt.tight_layout()
            plt.show()


    def render(sim, config:cameraConfig, width:int=100, height:int=100,
                object_ids:Optional[list]=None, return_rgb:bool=False, 
                retain_unknowns:bool=False, debug:bool=False) -> np.ndarray:
        """ Produce pointcloud data from depth image. 
        
            Args:
                sim: Simulation object
                config (cameraConfig): Camera parameters for this viewpoint
                width (int): Width of image to render from this camera 
                    viewpoint. Default=100 
                object_ids (list): Specify specific object ids to 
                    retain. Default=None
                return_rgb (bool): Choose to return the RGB image from this 
                    camera viewpoint. Default=False
                retain_unknowns: Sometimes, pybullet seems to return 
                    mysteriously large object ids in the segment mask. Choose 
                    whether to retain these. Default=False

            Returns:
                    {
                        'pcd': point cloud data,
                        'ids': object_id mask (note, id=0 reserved for softbodies)
                        'rgba_px (optional): RGB image from this camera viewpoint.
                    } 
        """
        start = time.time()
        w, h, rgba_px, depth_raw_cam_dists, segment_mask = sim.getCameraImage(
            width=width, height=height,
            viewMatrix=config.view_matrix, projectionMatrix=config.proj_matrix,
            shadow=False,
            lightDirection=[1,1,1],
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)

        #print(f'get cam image time: {time.time() - start}')
        tic = time.time()

        if retain_unknowns:
            next_id = object_ids[-1]+1
            unique_ids = np.unique(segment_mask)
            object_ids.extend(list(unique_ids[unique_ids>1E6]))

        pcd, ids = ProcessCamera.make_pcd(depth_raw_cam_dists, config=config, object_ids=object_ids, segment_mask=segment_mask)

        if debug: print('make_pcd took {:4f}'.format(time.time()-tic))
        
        if return_rgb:
            return pcd, ids, rgba_px
        else:
            return pcd, ids


    def draw_point_clouds(pts:np.ndarray, ids:np.ndarray, cmap=plt.cm.jet):
        _, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], marker='.', s=2.0,
                c=ids[:,0], cmap=cmap)
        ax.set_xlim(-0.4,1.0); ax.set_ylim(-0.6,0.6); ax.set_zlim(0,0.8)
        ax.view_init(10, -25)
        plt.tight_layout()
        plt.show()
        return


if __name__== "__main__":
    pass