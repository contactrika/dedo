# Utility class for managing camera configuration parameters
#
# @edwin-pan
#

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """ Makes numpy arrays serializable"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_camera_config(sim):
    """ Grabs camera config from sim & constructs a cameraConfig object"""
    _, _, view_matrix, proj_matrix, _, cam_forward, cam_horiz, cam_vert, \
    _, _, cam_dist, cam_tgt = sim.getDebugVisualizerCamera()
    camera_config = cameraConfig(view_matrix=view_matrix,
                                 proj_matrix=proj_matrix,
                                 cam_forward=cam_forward,
                                 cam_horiz=cam_horiz,
                                 cam_vert=cam_vert,
                                 cam_dist=cam_dist,
                                 cam_tgt=cam_tgt)
    return camera_config


class cameraConfig():
    def __init__(self, view_matrix:np.ndarray, 
                       proj_matrix:np.ndarray, 
                       cam_forward:np.ndarray, 
                       cam_horiz:np.ndarray, 
                       cam_vert:np.ndarray, 
                       cam_dist:float, 
                       cam_tgt:np.ndarray, 
                       file_location:str='') -> None:
        """ Initialize a camera config object w/ values"""

        self.view_matrix = view_matrix
        self.proj_matrix = proj_matrix
        self.cam_forward = cam_forward
        self.cam_horiz = cam_horiz
        self.cam_vert = cam_vert
        self.cam_dist = cam_dist
        self.cam_tgt = cam_tgt

        self.file_location=file_location

    @classmethod
    def from_file(cls, file):

        with open(file, "r") as f:
            config = json.load(f)
            
        view_matrix = np.asarray(config['view_matrix'])
        proj_matrix = np.asarray(config['proj_matrix'])
        cam_forward = np.asarray(config['cam_forward'])
        cam_horiz = np.asarray(config['cam_horiz'])
        cam_vert = np.asarray(config['cam_vert'])
        cam_dist = config['cam_dist']
        cam_tgt = np.asarray(config['cam_tgt'])
        
        # Construct class object
        cfg = cls(view_matrix, proj_matrix, cam_forward, cam_horiz, cam_vert, 
                        cam_dist, cam_tgt, file_location=file)

        return cfg

    def __str__(self):
        return f"Custom camera config. location:({self.file_location})"


    def __repr__(self):
        return f'[view_matrix, proj_matrix, cam_forward, cam_horiz, cam_vert, \
                  cam_dist, cam_tgt]' + '\n' + \
               f'[ {self.view_matrix} , {self.proj_matrix} , {self.cam_forward}\
                , {self.cam_horiz} , {self.cam_vert} , {self.cam_dist}\
                , {self.cam_tgt} ]'


    def get_as_dict(self):
        cam_config = {'view_matrix': self.view_matrix,
                      'proj_matrix': self.proj_matrix,
                      'cam_forward': self.cam_forward,
                      'cam_horiz': self.cam_horiz,
                      'cam_vert': self.cam_vert,
                      'cam_dist': self.cam_dist,
                      'cam_tgt': self.cam_tgt
                      }
        return cam_config

    def dump_json(self, destination_path):
        with open(destination_path, "w") as f:
            json.dump(self.get_as_dict(), f, indent = 4, cls=NumpyEncoder)
        f.close()
        self.file_location = destination_path
        return

    def load_json(self, source_path):
        config = json.loads(source_path)
        self.view_matrix = np.asarray(config['view_matrix'])
        self.proj_matrix = np.asarray(config['proj_matrix'])
        self.cam_forward = np.asarray(config['cam_forward'])
        self.cam_horiz = np.asarray(config['cam_horiz'])
        self.cam_vert = np.asarray(config['cam_vert'])
        self.cam_dist = config['cam_dist']
        self.cam_tgt = np.asarray(config['cam_tgt'])
        return

    @staticmethod
    def _load_json(source_path):
        config = json.loads(source_path)
        view_matrix = np.asarray(config['view_matrix'])
        proj_matrix = np.asarray(config['proj_matrix'])
        cam_forward = np.asarray(config['cam_forward'])
        cam_horiz = np.asarray(config['cam_horiz'])
        cam_vert = np.asarray(config['cam_vert'])
        cam_dist = config['cam_dist']
        cam_tgt = np.asarray(config['cam_tgt'])
        return view_matrix, proj_matrix, cam_forward, cam_horiz, cam_vert, cam_dist, cam_tgt
