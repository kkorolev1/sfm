from typing import Dict, Tuple
import os
import numpy as np

from common.dataset import Dataset
from common.intrinsics import Intrinsics
from sfm.geom import quaternion_to_rotation_matrix


def pose_list_to_mat(pose_list):
    mat = np.zeros((4, 4), dtype=np.float64)
    R = np.linalg.inv(quaternion_to_rotation_matrix(pose_list[3:]))
    t = np.array(pose_list[:3], dtype=np.float64)
    mat[:3,:3] = R
    mat[:3,-1] = -R @ t
    mat[3,3] = 1
    return mat

def get_frame_to_image_path(data_dir: str) -> Dict[int, str]:
    frame_to_image_path = Dataset.read_dict_of_lists(Dataset.get_rgb_list_file(data_dir))
    return {frame: os.path.join(data_dir, image_path) 
            for frame, image_path in frame_to_image_path.items()}


def get_frame_to_extrinsic_matrix(data_dir: str) -> Dict[int, np.array]:
    frame_to_to_extrinsic_mat = Dataset.read_dict_of_lists(Dataset.get_known_poses_file(data_dir))
    return {frame: pose_list_to_mat(pose_list)
            for frame, pose_list in frame_to_to_extrinsic_mat.items()}


def get_intrinsics_matrix(data_dir: str) -> np.array:
    intrinsics = Intrinsics.read(Dataset.get_intrinsics_file(data_dir))
    return np.array([
        [intrinsics.fx, 0, intrinsics.cx],
        [0, intrinsics.fy, intrinsics.cy],
        [0, 0, 1],
    ], dtype=np.float64)


def get_frame_to_proj_matrix(intrinsics_mat: np.array,
                             frame_to_extrinsic_mat: Dict[int, np.array]) -> Dict[int, np.array]:
    frame_to_proj_matrix = {}
    for frame, extrinsic_mat in frame_to_extrinsic_mat.items():
        frame_to_proj_matrix[frame] = intrinsics_mat @ extrinsic_mat[:3]
    return frame_to_proj_matrix