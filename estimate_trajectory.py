from typing import Dict, Tuple, List
import os
import copyreg
import pickle
import logging
from collections import defaultdict

import numpy as np
import cv2 as cv
from scipy.optimize import least_squares

from common.dataset import Dataset
from common.intrinsics import Intrinsics
from common.trajectory import Trajectory

def patch_keypoint_pickiling():
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


class Track:
    def __init__(self, frames=None, items=None):
        self.frames = frames if frames is not None else []
        self.items = items if items is not None else []
        self.frame_to_item = {frame_id: item for frame_id, item in zip(self.frames, self.items)}

    def append(self, frame_id, item):
        self.frames.append(frame_id)
        self.items.append(item)
        self.frame_to_item[frame_id] = item
    
    def has_frame(self, frame_id):
        return frame_id in self.frame_to_item

    def get_item(self, i):
        return self.items[i]

    def get_frame(self, i):
        return self.frames[i]
    
    def __getitem__(self, i):
        return self.get_item(i)

    def __len__(self):
        return len(self.items)
    
    def __str__(self):
        return "[" + ",".join([f"({f},{i})" for f,i in zip(self.frames, self.items)]) + "]"


def get_frame_to_image_path(data_dir: str) -> Dict[int, str]:
    frame_to_image_path = Dataset.read_dict_of_lists(Dataset.get_rgb_list_file(data_dir))
    return {frame: os.path.join(data_dir, image_path) 
            for frame, image_path in frame_to_image_path.items()}

def quaternion_to_rotation_matrix(quaternion): 
    """
        Generate rotation matrix 3x3  from the unit quaternion.
        Input:
        qQuaternion -- tuple consisting of (qx,qy,qz,qw) where
             (qx,qy,qz,qw) is the unit quaternion.
        Output:
        matrix -- 3x3 rotation matrix
    """
    q = np.array(quaternion, dtype=np.float64, copy=True) 
    nq = np.dot(q, q)
    eps = np.finfo(float).eps * 4.0
    assert nq > eps
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
                (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3]),
                (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3]),
                (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1])
            ), dtype=np.float64)

def pose_list_to_mat(pose_list):
    mat = np.zeros((4, 4), dtype=np.float64)
    R = quaternion_to_rotation_matrix(pose_list[3:])
    t = np.array(pose_list[:3], dtype=np.float64)
    mat[:3,:3] = R.T
    mat[:3,-1] = -R.T @ t
    mat[3,3] = 1
    return mat

def get_frame_to_extrinsic_matrix(data_dir: str) -> Dict[int, np.array]:
    frame_to_pose = Dataset.read_dict_of_lists(Dataset.get_known_poses_file(data_dir))
    return {frame: pose_list_to_mat(pose_list)
            for frame, pose_list in frame_to_pose.items()}


def get_frame_to_keypoints(
        frame_to_image_path: Dict[int, str], *args, **kwargs) -> Dict[int, FrameKeypoints]:
    """
    - Calculates keypoints for all frames and stores them into dictionary
    frame_id -> keypoints
    """
    frame_to_keypoints = {}
    for frame_id, image_path in frame_to_image_path.items():
        frame_to_keypoints[frame_id] = FrameKeypoints.create(
            frame_id, image_path, *args, **kwargs
        )
    return frame_to_keypoints

def match_keypoints_pair(
        keypoints1: FrameKeypoints, keypoints2: FrameKeypoints,
        intrinsics_mat: np.array = None,
        method="bf",
        ratio_threshold=0.8) -> List[cv.DMatch]:
    norm_type = cv.NORM_L2 if keypoints1.method == "sift" else cv.NORM_HAMMING
    if method == "bf":
        matcher = cv.BFMatcher(norm_type)
    elif method == "flann":
        matcher = cv.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        ) 
    matches = matcher.knnMatch(keypoints1.des, keypoints2.des, k=2)
    filtered_matches = []
    points1 = []
    points2 = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            filtered_matches.append(m)
            points1.append(keypoints1.kp[m.queryIdx].pt)
            points2.append(keypoints2.kp[m.trainIdx].pt)
    if len(filtered_matches) == 0:
        return []
    points1 = np.array(points1, dtype=np.float64)
    points2 = np.array(points2, dtype=np.float64)
    if intrinsics_mat is None:
        _, mask = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC)
    else:
        _, mask = cv.findEssentialMat(points1, points2, intrinsics_mat, cv.FM_RANSAC)
    if mask is None:
        return []
    mask = mask.ravel().astype(bool)
    inliers_matches = [match for match, m in zip(filtered_matches, mask) if m]
    #logging.info(f"""Frames [{keypoints1.frame_id}/{keypoints2.frame_id}] Matches:{len(matches)} Filtered:{len(filtered_matches)} Inliers:{len(inliers_matches)}""")
    return inliers_matches

def match_anchor_keypoints(
        anchor_frames, 
        frame_to_keypoints: Dict[int, FrameKeypoints],
        *args, **kwargs) -> Dict[Tuple[int, int], List[cv.DMatch]]:
    """
    Matches keypoints for anchor frames
    """
    anchor_frames = sorted(anchor_frames)
    matches_for_frames = {}
    for i in range(len(anchor_frames) - 1):
        for j in range(i + 1, len(anchor_frames)):
            frame_id1, frame_id2 = anchor_frames[i], anchor_frames[j]
            inliers_matches = match_keypoints_pair(
                frame_to_keypoints[frame_id1],
                frame_to_keypoints[frame_id2],
                *args, **kwargs
            )
            matches_for_frames[(frame_id1, frame_id2)] = inliers_matches
    return matches_for_frames

def match_test_with_anchor(
        test_frame_to_keypoints: Dict[int, FrameKeypoints],
        anchor_frame_to_keypoints: Dict[int, FrameKeypoints],
        *args, **kwargs) -> Dict[Tuple[int, int], List[cv.DMatch]]:
    frames_to_matches = {}
    for test_frame, test_keypoints in test_frame_to_keypoints.items():
        for anchor_frame, anchor_keypoints in anchor_frame_to_keypoints.items():
            matches = match_keypoints_pair(
                test_keypoints,
                anchor_keypoints,
                *args, **kwargs
            )
            frames_to_matches[(test_frame, anchor_frame)] = matches
    return frames_to_matches

def dfs(graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
        v: Tuple[int, int], used: Dict[Tuple[int, int], bool],
        tracks: List[Track]):
    used[v] = True
    tracks[-1].append(frame_id=v[0], item=v[1])

    for u in graph[v]:
        # check if we didn't use this frame and we didn't visit this vertex
        if not tracks[-1].has_frame(u[0]) and not used[u]:
            dfs(graph, u, used, tracks)

def get_tracks(
        frames_to_matches: Dict[Tuple[int, int], List[cv.DMatch]],
        track_min_length=5) -> List[Track]:
    """
    Takes dictionary (frame_id1, frame_id2) -> list of matches
    and returns list of tracks
    """    
    graph = defaultdict(list)
    
    # Build graph
    for (frame_id1, frame_id2), matches in frames_to_matches.items():
        for match in matches:
            u = (frame_id1, match.queryIdx)
            v = (frame_id2, match.trainIdx)
            graph[u].append(v)
            graph[v].append(u)
    
    used = {v: False for v in graph}
    tracks = []
    for v in graph:
        if not used[v]:
            tracks.append(Track())
            dfs(graph, v, used, tracks)
    tracks = list(filter(lambda track: len(track) >= track_min_length, tracks))
    return tracks

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

def project_points3d(points3d: np.array,
                     intrinsics_mat: np.array,
                     extrinsic_mat: np.array) -> np.array:
    R = extrinsic_mat[:3,:3]
    t = extrinsic_mat[:3,-1]
    proj_points, _ = cv.projectPoints(points3d, R, t, intrinsics_mat, None)
    return proj_points

def reprojection_error(
        point2d: np.array,
        point3d: np.array,
        intrinsics_mat: np.array,
        extrinsic_mat: Dict[int, np.array]) -> bool:
    reproj_point = project_points3d(point3d, intrinsics_mat, extrinsic_mat).ravel()
    return np.linalg.norm(reproj_point - point2d)

def convert_from_homo(point: np.array) -> np.array:
    """
    point in homogenous coordinates: (D+1,N) or (D+1,) -> (D,N) or (D,)
    """
    d = point.shape[0]
    return point[:d-1] / point[d-1]

def triangulate_track(track: Track,
           frame_to_keypoints: Dict[int, FrameKeypoints],
           intrinsics_mat: np.array,
           frame_to_extrinsic_mat: Dict[int, np.array],
           frame_to_proj_matrix: Dict[int, np.array],
           n_samples: int = 20,
           reproj_error_threshold: int = 10):
    points = []
    for _ in range(n_samples):
        i, j = np.random.choice(len(track), size=2, replace=False)
        
        frame_id1 = track.get_frame(i)
        frame_id2 = track.get_frame(j)
        point2d_frame1 = frame_to_keypoints[frame_id1].get_point2d(track[i])
        point2d_frame2 = frame_to_keypoints[frame_id2].get_point2d(track[j])

        proj_matr1 = frame_to_proj_matrix[frame_id1]
        proj_matr2 = frame_to_proj_matrix[frame_id2]

        point4d = cv.triangulatePoints(proj_matr1, proj_matr2, point2d_frame1, point2d_frame2).ravel()
        points.append(point4d)
    
    # points_errors = []
    # for point4d in points:
    #     track_error = 0
    #     for i in range(len(track)):
    #         frame_id = track.get_frame(i)
    #         point2d = frame_to_keypoints[frame_id].get_point2d(track[i])
    #         point3d = convert_from_homo(point4d)
    #         error = reprojection_error(point2d, point3d, intrinsics_mat, frame_to_extrinsic_mat[frame_id])
    #         track_error += error
    #     points_errors.append(track_error)
    
    point4d = np.mean(points, axis=0)

    def res_fn(x):
        res = []
        for i in range(len(track)):
            frame_id = track.get_frame(i)
            proj_matr = frame_to_proj_matrix[frame_id]
            proj_point = convert_from_homo(proj_matr @ x)
            point2d = frame_to_keypoints[frame_id].get_point2d(track[i])
            res.append(proj_point - point2d)
        return np.array(res, dtype=np.float64).ravel()

    ls_result = least_squares(res_fn, x0=point4d, method="lm")
    point3d = convert_from_homo(ls_result.x)

    for i in range(len(track)):
        frame_id = track.get_frame(i)
        point2d = frame_to_keypoints[frame_id].get_point2d(track[i])
        point3d = convert_from_homo(point4d)
        error = reprojection_error(point2d, point3d, intrinsics_mat, frame_to_extrinsic_mat[frame_id])
        if error > reproj_error_threshold:
            return None

    return point3d


def triangulate_tracks(
        tracks: List[Track],
        frame_to_keypoints: Dict[int, FrameKeypoints],
        intrinsics_mat: np.array,
        frame_to_extrinsic_mat: Dict[int, np.array],
        reproj_error_threshold=10) -> Tuple[List[Track], List[np.array]]:
    
    frame_to_proj_matrix = get_frame_to_proj_matrix(intrinsics_mat, frame_to_extrinsic_mat)
    good_tracks = []
    tracks_point3d = []
    for track in tracks:
        point3d = triangulate_track(
            track, frame_to_keypoints, intrinsics_mat, 
            frame_to_extrinsic_mat, frame_to_proj_matrix,
            reproj_error_threshold=reproj_error_threshold
        )
        if point3d is not None:
            good_tracks.append(track)
            tracks_point3d.append(point3d)

    return good_tracks, tracks_point3d

def get_inliers_keypoints(
        tracks: List[Track],
        tracks_points: List[np.array],
        frame_to_keypoints: Dict[int, FrameKeypoints]) -> Dict[int, FrameKeypoints]:
    frame_to_kp = defaultdict(list)
    frame_to_des = defaultdict(list)
    frame_to_points3d = defaultdict(list)
    for track, point3d in zip(tracks, tracks_points):
        for i in range(len(track)):
            frame_id = track.get_frame(i)
            point_idx = track.get_item(i)
            kp, des = frame_to_keypoints[frame_id][point_idx]
            frame_to_kp[frame_id].append(kp)
            frame_to_des[frame_id].append(des)
            frame_to_points3d[frame_id].append(point3d)
    anchor_frame_to_keypoints = {}
    for frame_id in frame_to_kp:
        keypoints = FrameKeypoints(
            frame_to_kp[frame_id],
            np.array(frame_to_des[frame_id]),
            frame_id, 
            frame_to_keypoints[frame_id].method,
            np.array(frame_to_points3d[frame_id], dtype=np.float64),
        )
        anchor_frame_to_keypoints[frame_id] = keypoints
    return anchor_frame_to_keypoints

def create_pose(rvec, tvec):
    R, _ = cv.Rodrigues(rvec)
    pose_mat = np.zeros((4, 4))
    pose_mat[:3, :3] = R.T
    pose_mat[:3, 3] = -R.T @ tvec.ravel()
    pose_mat[3, 3] = 1
    return pose_mat

def estimate_test_poses(
        test_frame_to_points2d: Dict[int, List[np.array]],
        test_frame_to_points3d: Dict[int, List[np.array]],
        intrinsics_mat):
    frame_to_pose = {}
    for frame_id in test_frame_to_points2d:
        points2d = np.array(test_frame_to_points2d[frame_id])
        points3d = np.array(test_frame_to_points3d[frame_id])
        _, rvec, tvec = cv.solvePnP(points3d, points2d, intrinsics_mat, None)
        frame_to_pose[frame_id] = create_pose(rvec, tvec)
    return frame_to_pose

class SFMConfig:
    keypoints_save_dir: str = None
    keypoints_method: str = "sift" # [orb, sift]
    n_keypoints: int = 300
    keypoints_overwrite_cache: bool = False
    matching_method: str = "flann" # [bf, flann]
    matching_ratio_threshold: float = 0.7
    track_min_length: int = 3
    reproj_error_threshold: int = 9

def estimate_trajectory(data_dir: str, out_dir: str):
    patch_keypoint_pickiling()

    config = SFMConfig()
    # DEBUG ONLY
    #config.keypoints_save_dir = "keypoints"

    frame_to_image_path = get_frame_to_image_path(data_dir)
    logging.info("Get keypoints for frames...")
    frame_to_keypoints = get_frame_to_keypoints(
        frame_to_image_path, data_dir, 
        save_dir=config.keypoints_save_dir, 
        method=config.keypoints_method,
        n_keypoints=config.n_keypoints,
        overwrite_cache=config.keypoints_overwrite_cache
    )
    frame_to_extrinsic_mat = get_frame_to_extrinsic_matrix(data_dir)
    intrinsics_mat = get_intrinsics_matrix(data_dir)

    anchor_frames = list(frame_to_extrinsic_mat.keys())
    logging.info("Get matches within keypoints...")
    frames_to_matches = match_anchor_keypoints(
        anchor_frames, frame_to_keypoints, 
        intrinsics_mat=intrinsics_mat, method=config.matching_method,
        ratio_threshold=config.matching_ratio_threshold
    )
    logging.info("Get keypoints tracks...")
    tracks = get_tracks(frames_to_matches, track_min_length=config.track_min_length)
    logging.info(f"Found {len(tracks)} tracks!")
    logging.info(f"Mean track length {np.mean([len(t) for t in tracks]):.3f}")
    
    logging.info("Triangulate points along tracks...")
    good_tracks, tracks_points3d = triangulate_tracks(
        tracks, frame_to_keypoints, intrinsics_mat, frame_to_extrinsic_mat,
        reproj_error_threshold=config.reproj_error_threshold
    )
    logging.info(f"Found {len(good_tracks)} good tracks!")
    logging.info(f"Mean good track length {np.mean([len(t) for t in good_tracks]):.3f}")
    anchor_frame_to_keypoints = get_inliers_keypoints(good_tracks, tracks_points3d, frame_to_keypoints)

    for frame_id in anchor_frame_to_keypoints:
        logging.info(f"Frame [{frame_id}] #Points [{len(anchor_frame_to_keypoints[frame_id])}]")

    test_frames = frame_to_keypoints.keys() - set(anchor_frames)
    test_frame_to_keypoints = {f:k for f,k in frame_to_keypoints.items() if f in test_frames}

    frames_to_matches = match_test_with_anchor(
        test_frame_to_keypoints,
        anchor_frame_to_keypoints,
        intrinsics_mat=intrinsics_mat, method=config.matching_method,
        ratio_threshold=config.matching_ratio_threshold
    )

    test_frame_to_points2d = defaultdict(list)
    test_frame_to_points3d = defaultdict(list)
    for (test_frame, anchor_frame), matches in frames_to_matches.items():
        for match in matches:
            point2d = anchor_frame_to_keypoints[anchor_frame].get_point2d(match.trainIdx)
            point3d = anchor_frame_to_keypoints[anchor_frame].get_point3d(match.trainIdx)

            test_frame_to_points2d[test_frame].append(point2d)
            test_frame_to_points3d[test_frame].append(point3d)
    
    for frame_id in test_frame_to_points2d:
        logging.info(f"Test Frame [{frame_id}] #Points [{len(test_frame_to_points2d[frame_id])}]")

    trajectory = estimate_test_poses(
        test_frame_to_points2d, test_frame_to_points3d, intrinsics_mat
    )
    Trajectory.write(Dataset.get_result_poses_file(out_dir), trajectory)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filemode="w", filename="log.txt", format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())
    out_dir = "out_dir"
    os.makedirs(out_dir, exist_ok=True)
    estimate_trajectory("public_tests/00_test_slam_input", out_dir)