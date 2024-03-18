from typing import Dict

import copyreg
import cv2 as cv
import numpy as np
import os
import pickle

def patch_keypoint_pickling():
    """
    Patches pickle to use with OpenCV KeyPoint
    See: https://stackoverflow.com/questions/10045363/pickling-cv2-keypoint-causes-picklingerror
    """
    def _pickle_keypoint(keypoint): #  : cv2.KeyPoint
        return cv.KeyPoint, (
            keypoint.pt[0],
            keypoint.pt[1],
            keypoint.size,
            keypoint.angle,
            keypoint.response,
            keypoint.octave,
            keypoint.class_id,
        )
    # C++ Constructor, notice order of arguments : 
    # KeyPoint (float x, float y, float _size, float _angle=-1, float _response=0, int _octave=0, int _class_id=-1)

    # Apply the bundling to pickle
    copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoint)

class FrameKeypoints:
    """
    Stores OpenCV keypoints and descriptors for a frame
    """
    def __init__(self, kp, des, frame_id, method, points3d=None):
        self.kp = kp
        self.des = des
        self.frame_id = frame_id
        self.method = method
        if points3d is not None:
            self.points3d = points3d

    @staticmethod
    def load_from_disk(filepath):
        with open(filepath, "rb") as fp:
            frame_keypoints = pickle.load(fp)
        return frame_keypoints

    @staticmethod
    def create(frame_id: int, image_path: str, data_dir: str,
               save_dir: str = "keypoints", method: str = "orb",
               n_keypoints: int = 500, overwrite_cache: bool = False):
        """
        Computes keypoints for a frame or loads them from disk
        """
        # using cache only for debug
        # because in testing system the filesystem is read-only
        if save_dir is not None:
            keypoints_dir = os.path.join(save_dir, data_dir)
            filename = os.path.splitext(os.path.basename(image_path))[0]
            if not os.path.exists(keypoints_dir):
                os.makedirs(keypoints_dir, exist_ok=True)
            keypoints_filename = os.path.join(keypoints_dir, f"{filename}_{method}.pickle")
            if os.path.exists(keypoints_filename) and not overwrite_cache:
                return FrameKeypoints.load_from_disk(keypoints_filename)
        img = cv.imread(image_path)
        if method == "orb":
            kp_detector = cv.ORB_create(nfeatures=n_keypoints)
        elif method == "sift":
            kp_detector = cv.SIFT_create(nfeatures=n_keypoints)
        kp, des = kp_detector.detectAndCompute(img, None)
        keypoints = FrameKeypoints(kp, des, frame_id, method)
        if save_dir is not None:
            keypoints.save_to_disk(keypoints_filename)
        return keypoints

    def save_to_disk(self, filepath):
        with open(filepath, "wb") as fp:
            pickle.dump(self, fp)
        return self
    
    def __getitem__(self, i):
        return (self.kp[i], self.des[i])

    def __len__(self):
        return len(self.kp)

    def get_point2d(self, kp_idx: int) -> np.array:
        return np.array(self.kp[kp_idx].pt, dtype=np.float64)
    
    def get_point3d(self, kp_idx: int) -> np.array:
        return self.points3d[kp_idx]
    

def get_frame_to_keypoints(
        frame_to_image_path: Dict[int, str], **kwargs) -> Dict[int, FrameKeypoints]:
    """
    - Calculates keypoints for all frames and stores them into dictionary
    frame_id -> keypoints
    """
    frame_to_keypoints = {}
    for frame_id, image_path in frame_to_image_path.items():
        frame_to_keypoints[frame_id] = FrameKeypoints.create(
            frame_id, image_path, **kwargs
        )
    return frame_to_keypoints