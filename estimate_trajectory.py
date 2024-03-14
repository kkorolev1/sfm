from typing import Dict, Tuple, List
import os
import copyreg
import pickle
import logging
from collections import defaultdict

import numpy as np
import cv2 as cv

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
    def __init__(self, kp, des, frame_id, method):
        self.kp = kp
        self.des = des
        self.frame_id = frame_id
        self.method = method

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


class Track:
    def __init__(self, items=None, frames=None):
        self.items = items if items is not None else []
        self.frames = frames if frames is not None else []

    def append(self, item, frame_id):
        self.items.append(item)
        self.frames.append(frame_id)
    
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
        method="bf",
        ratio_threshold=0.8,
        ransacReprojThreshold=1) -> List[cv.DMatch]:
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
    points1 = np.array(points1, dtype=np.int32)
    points2 = np.array(points2, dtype=np.int32)
    _, mask = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC,
                                    ransacReprojThreshold=ransacReprojThreshold)
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
    frames = sorted(anchor_frames)
    matches_for_frames = {}
    for i in range(len(frames) - 1):
        inliers_matches = match_keypoints_pair(
            frame_to_keypoints[frames[i]],
            frame_to_keypoints[frames[i + 1]],
            *args, **kwargs
        )
        matches_for_frames[(frames[i], frames[i + 1])] = inliers_matches
    return matches_for_frames

def match_unknowns_with_inliers(
        unk_frame_to_keypoints: Dict[int, FrameKeypoints],
        anchor_frame_to_inliers: Dict[int, FrameKeypoints],
        *args, **kwargs) -> Dict[Tuple[int, int], List[cv.DMatch]]:
    frames_to_matches = {}
    for unk_frame, unk_keypoints in unk_frame_to_keypoints.items():
        for anchor_frame, inliers in anchor_frame_to_inliers.items():
            inliers_matches = match_keypoints_pair(
                unk_keypoints,
                inliers,
                *args, **kwargs
            )
            frames_to_matches[(unk_frame, anchor_frame)] = inliers_matches
    return frames_to_matches

def get_tracks(
        frames_to_matches: Dict[Tuple[int, int], List[cv.DMatch]],
        track_min_length=5) -> List[Track]:
    """
    Takes dictionary (frame_id1, frame_id2) -> list of matches
    and returns list of tracks
    """
    tracks = []
    frames_ids = list(sorted(frames_to_matches.keys()))
    valid_tracks = set()

    # Iterating over frames
    for frame_ids in frames_ids:
        # Indices of matches that we can use on this pair (frame,next_frame)
        matches_to_use = set(range(len(frames_to_matches[frame_ids])))
        # Copy tracks indices that are not finished
        new_valid_tracks = valid_tracks.copy()
        # Iterating over tracks that are not finished
        for track_idx in valid_tracks:
            # Iterating over matches in a frame pair
            for match_idx in matches_to_use:
                match = frames_to_matches[frame_ids][match_idx]
                if match.queryIdx == tracks[track_idx][-1]:
                    tracks[track_idx].append(match.trainIdx, frame_ids[1])
                    matches_to_use.remove(match_idx)
                    break
            else:
                new_valid_tracks.remove(track_idx)
        # Start new tracks for remaining matches
        for match_idx in matches_to_use:
            match = frames_to_matches[frame_ids][match_idx]
            new_valid_tracks.add(len(tracks))
            tracks.append(Track([match.queryIdx, match.trainIdx], [frame_ids[0], frame_ids[1]]))
        valid_tracks = new_valid_tracks
    tracks = [track for track in tracks if len(track) >= track_min_length]
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

def point4d_to_3d(point4d: np.array) -> np.array:
    """
    point4d: (4,N) or (4,)
    """
    return point4d[:3] / point4d[3]

def ransac_triangulate_track(track: Track,
           frame_to_keypoints: Dict[int, FrameKeypoints],
           intrinsics_mat: np.array,
           frame_to_extrinsic_mat: Dict[int, np.array],
           frame_to_proj_matrix: Dict[int, np.array],
           max_trials: int = 1000,
           subsample_size: int = 4,
           reproj_error_threshold: int = 10):
    points3d = []
    for i in range(len(track) - 1):
        for j in range(i + 1, len(track)):
            frame_id1 = track.get_frame(i)
            frame_id2 = track.get_frame(j)
            point2d_frame1 = frame_to_keypoints[frame_id1].get_point2d(track[i])
            point2d_frame2 = frame_to_keypoints[frame_id2].get_point2d(track[i+1])

            proj_matr1 = frame_to_proj_matrix[frame_id1]
            proj_matr2 = frame_to_proj_matrix[frame_id2]

            point4d = cv.triangulatePoints(proj_matr1, proj_matr2, point2d_frame1, point2d_frame2)
            point3d = point4d_to_3d(point4d)

            points3d.append(point3d)

    points3d = np.array(points3d, dtype=np.float64)

    for iter in range(max_trials):
        indices = np.random.choice(len(points3d), 4)

        error = reprojection_error(
            point2d_frame1, point3d, intrinsics_mat, frame_to_extrinsic_mat[frame_id1]
        )


def triangulate_tracks(
        anchor_frames: List[int],
        tracks: List[Track],
        frame_to_keypoints: Dict[int, FrameKeypoints],
        intrinsics_mat: np.array,
        frame_to_extrinsic_mat: Dict[int, np.array],
        max_trials=1000,
        reproj_error_threshold=10) -> Tuple[Dict[int, List[int]], Dict[int, List[np.array]]]:
    
    anchor_frames = sorted(anchor_frames)
    frame_to_proj_matrix = get_frame_to_proj_matrix(intrinsics_mat, frame_to_extrinsic_mat)

    for track_index, track in enumerate(tracks):
        ransac_triangulate_track(
            track, frame_to_keypoints, intrinsics_mat, frame_to_extrinsic_mat, frame_to_proj_matrix,
            max_trials=max_trials, reproj_error_threshold=reproj_error_threshold
        )

    # for track_idx, points3d in track_idx_to_points3d.items():
    #     track = tracks[track_idx]
    #     for frame_id, point_idx, point3d in zip(track.frames[:-1], track.items[:-1], points3d):
    #         frame_to_point_indices[frame_id].append(point_idx)
    #         frame_to_points3d[frame_id].append(point3d)

    # return frame_to_point_indices, frame_to_points3d

def get_inliers_keypoints(
        frame_to_inliers_indices: Dict[int, List[int]],
        frame_to_keypoints: Dict[int, FrameKeypoints]) -> Dict[int, FrameKeypoints]:
    anchor_frame_to_inliers = {}
    for frame_id in frame_to_inliers_indices:
        kp_list, des_list = [], []
        for point_idx in frame_to_inliers_indices[frame_id]:
            kp, des = frame_to_keypoints[frame_id][point_idx]
            kp_list.append(kp)
            des_list.append(des)
        if len(kp_list) > 1:
            inliers_keypoints = FrameKeypoints(
                kp_list, np.array(des_list), frame_id, frame_to_keypoints[frame_id].method
            )
            anchor_frame_to_inliers[frame_id] = inliers_keypoints
    return anchor_frame_to_inliers

def create_pose(rvec, tvec):
    R, _ = cv.Rodrigues(rvec)
    pose_mat = np.zeros((4, 4))
    pose_mat[:3, :3] = R.T
    pose_mat[:3, 3] = -R.T @ tvec.ravel()
    pose_mat[3, 3] = 1
    return pose_mat

def solve_PnP(
        unk_frame_to_points2d: Dict[int, np.array],
        unk_frame_to_points3d: Dict[int, np.array],
        intrinsics_mat):
    frame_to_extrinsic = {}
    for frame_id in unk_frame_to_points2d:
        points2d = np.array(unk_frame_to_points2d[frame_id])
        points3d = np.array(unk_frame_to_points3d[frame_id])
        _, rvec, tvec = cv.solvePnP(points3d, points2d, intrinsics_mat, None)
        frame_to_extrinsic[frame_id] = create_pose(rvec, tvec)
    return frame_to_extrinsic

class SFMConfig:
    keypoints_save_dir: str = None
    keypoints_method: str = "sift" # [orb, sift]
    n_keypoints: int = 500
    keypoints_overwrite_cache: bool = False
    matching_method: str = "bf" # [bf, flann]
    matching_ratio_threshold: float = 0.7
    matching_ransac_threshold: int = 1
    track_min_length: int = 10
    reproj_error_threshold: int = 3

def estimate_trajectory(data_dir: str, out_dir: str):
    patch_keypoint_pickiling()

    config = SFMConfig()
    # DEBUG ONLY
    config.keypoints_save_dir = "keypoints"

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
    
    anchor_frames = list(frame_to_extrinsic_mat.keys())
    logging.info("Get matches within keypoints...")
    frames_to_matches = match_anchor_keypoints(
        anchor_frames, frame_to_keypoints,
        method=config.matching_method,
        ratio_threshold=config.matching_ratio_threshold,
        ransacReprojThreshold=config.matching_ransac_threshold
    )
    logging.info("Get keypoints tracks...")
    tracks = get_tracks(frames_to_matches, track_min_length=config.track_min_length)
    logging.info(f"Found {len(tracks)} tracks!")
    logging.info(f"Mean track length {np.mean([len(t) for t in tracks]):.3f}")

    intrinsics_mat = get_intrinsics_matrix(data_dir)
    
    logging.info("Triangulate points along tracks...")
    #frame_to_inliers_indices, frame_to_inliers_3d = 
    triangulate_points(
        anchor_frames, tracks, frame_to_keypoints, intrinsics_mat, frame_to_extrinsic_mat,
        reproj_error_threshold=config.reproj_error_threshold
    )

    # logging.info(f"Mean number of 3d points among frames {np.mean([len(points) for points in frame_to_inliers_3d.values()]):.3f}")
    # anchor_frame_to_inliers = get_inliers_keypoints(frame_to_inliers_indices, frame_to_keypoints)

    # unk_frames = frame_to_keypoints.keys() - set(anchor_frames)
    # unk_frame_to_keypoints = {f:k for f,k in frame_to_keypoints.items() if f in unk_frames}

    # frames_to_matches = match_unknowns_with_inliers(
    #     unk_frame_to_keypoints,
    #     anchor_frame_to_inliers,
    #     method=config.matching_method,
    #     ratio_threshold=config.matching_ratio_threshold,
    #     ransacReprojThreshold=config.matching_ransac_threshold
    # )

    # unk_frame_to_points2d = defaultdict(list)
    # unk_frame_to_points3d = defaultdict(list)
    # for (unk_frame, anchor_frame), matches in frames_to_matches.items():
    #     for match in matches:
    #         #unk_point2d = unk_frame_to_keypoints[unk_frame].get_point2d(match.queryIdx)
    #         anchor_point2d = anchor_frame_to_inliers[anchor_frame].get_point2d(match.trainIdx)
    #         anchor_point3d = frame_to_inliers_3d[anchor_frame][match.trainIdx]

    #         unk_frame_to_points2d[unk_frame].append(anchor_point2d)
    #         unk_frame_to_points3d[unk_frame].append(anchor_point3d)
        
    # trajectory = solve_PnP(unk_frame_to_points2d, unk_frame_to_points3d, intrinsics_mat)
    # Trajectory.write(Dataset.get_result_poses_file(out_dir), trajectory)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filemode="w", filename="log.txt", format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())
    out_dir = "out_dir"
    os.makedirs(out_dir, exist_ok=True)
    estimate_trajectory("public_tests/00_test_slam_input", out_dir)