import numpy as np
import matplotlib.pyplot as plt
import subprocess
import shutil


def render_video(img_dir_path: str, output_fname: str = 'vtx_test.mp4', 
                    framerate: int = 30, remove_frames:bool=True) -> None:
    """ Calls ffmpeg to render a directory of images"""
    
    fmpeg_render_cmd = ['ffmpeg', '-r', str(framerate), '-i', 
                        f'{img_dir_path}/%06d.png', '-vcodec', 'libx264', 
                        '-pix_fmt', 'yuv420p' ,f'{output_fname}']
    subprocess.call(fmpeg_render_cmd)

    if remove_frames:
        shutil.rmtree(img_dir_path)
    return


def visualize_pcd(pcd:np.ndarray, ids=None, intersection:bool=False, save_path=None):
    """ Useful, but not tested yet
    """
    
    color = 'green' if intersection else 'red'

    # Choose colormap
    cmap = plt.cm.viridis

    # Get the colormap colors
    custom_cmap = cmap(np.arange(cmap.N))

    # Set alpha
    custom_cmap[:, -1] = 0.3 # Set everything to be partially see-through    
    custom_cmap[-1] = np.array([1, 0, 0, 1]) # Set feature to be red
    custom_cmap[-2] = np.array([0, 0, 1, 1]) # Set feature to be blue

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1, projection='3d')
    if ids is None:
        ax1.scatter(pcd[:,0], pcd[:,1], pcd[:,2], 
                        marker='.', s=2.0, cmap=plt.cm.PRGn)
    else:
        ax1.scatter(pcd[:,0], pcd[:,1], pcd[:,2], 
                        marker='.', s=2.0, c=custom_cmap[ids.squeeze()])
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([-0.5, 0.5])
    ax1.set_zlim([-0.05, 1])
    ax1.set_title(f'Point Cloud (intersection={intersection})', 
                        color=color, fontsize=8)
    ax1.view_init(elev=12, azim=110)

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    fig.clear()
    plt.close(fig)


def visualize_data(img:np.ndarray, pcd:np.ndarray, ids:np.ndarray, 
                    azimuth:float=160, elevation:float=12,
                    status:bool=True, fig=None, save_path=None) -> None:
    """ Two pannel plot, with 3D pcd view on left, 2D RGB view on the right
    
    """
    if fig is None:
        fig = plt.figure(figsize=(10,5))

    color = 'green' if status else 'red'

    # the unknown ids are HUGE... remap to something reasonable
    id_remap = {id:i for i, id in enumerate(np.unique(ids))}
    id_remapped = np.array([id_remap[id] for id in ids])

    # The combined pcd
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(pcd[:,0], pcd[:,1], pcd[:,2], 
                    marker='.', s=2.0, c=id_remapped, cmap=plt.cm.PRGn)
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([-3, 10])
    ax1.set_zlim([-0.5, 10])
    ax1.set_title(f'Combined pointcloud -- Intersection={status}', color=color)
    ax1.view_init(elev=elevation, azim=azimuth)

    # The original image
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('RGB View')

    if save_path is not None:
        plt.savefig(save_path)
        fig.clear()

    return