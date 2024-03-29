from typing import Dict, Tuple, List
import os
import pickle
import logging
from collections import defaultdict

import numpy as np
import cv2 as cv

from common.dataset import Dataset
from common.trajectory import Trajectory

from sfm.track import Track, get_tracks
from sfm.keypoints import FrameKeypoints, get_frame_to_keypoints
from sfm.matching import match_anchor_keypoints, match_test_with_anchor
from sfm.utils import get_frame_to_image_path, get_frame_to_extrinsic_matrix, get_intrinsics_matrix
from sfm.triangulation import triangulate_tracks

def get_inliers_keypoints(
        tracks: List[Track],
        tracks_points: List[np.array],
        frame_to_keypoints: Dict[int, FrameKeypoints]) -> Dict[int, FrameKeypoints]:
    frame_to_kp = defaultdict(list)
    frame_to_des = defaultdict(list)
    frame_to_points3d = defaultdict(list)
    for track, point3d in zip(tracks, tracks_points):
        for frame_id, point_idx in zip(track.frames, track.items):
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
    pose_mat = np.zeros((4, 4), dtype=np.float64)
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
        _, rvec, tvec, _ = cv.solvePnPRansac(points3d, points2d, intrinsics_mat, None)
        frame_to_pose[frame_id] = create_pose(rvec, tvec)
    return frame_to_pose

def estimate_trajectory(data_dir: str, out_dir: str,
                        keypoints_save_dir: str = None,
                        keypoints_method: str = "orb", # [orb, sift]
                        n_keypoints: int = 500,
                        keypoints_overwrite_cache: bool = False,
                        matching_method: str = "bf", # [bf, flann]
                        matching_ratio_threshold: float = 0.6,
                        matching_use_intrinsics: bool = False,
                        track_min_length: int = 2,
                        reproj_error_threshold: int = 10):
    frame_to_image_path = get_frame_to_image_path(data_dir)
    logging.info("Get keypoints for frames...")
    
    frame_to_keypoints = get_frame_to_keypoints(
        frame_to_image_path, 
        data_dir=data_dir, 
        save_dir=keypoints_save_dir, 
        method=keypoints_method,
        n_keypoints=n_keypoints,
        overwrite_cache=keypoints_overwrite_cache
    )

    frame_to_extrinsic_mat = get_frame_to_extrinsic_matrix(data_dir)
    intrinsics_mat = get_intrinsics_matrix(data_dir)

    anchor_frames = list(frame_to_extrinsic_mat.keys())
    logging.info("Get matches within keypoints...")
    frames_to_matches = match_anchor_keypoints(
        anchor_frames, frame_to_keypoints, 
        intrinsics_mat=(intrinsics_mat if matching_use_intrinsics else None),
        method=matching_method,
        ratio_threshold=matching_ratio_threshold
    )
    logging.info("Get keypoints tracks...")
    tracks = get_tracks(frames_to_matches, track_min_length=track_min_length)
    logging.info(f"Found {len(tracks)} tracks!")
    logging.info(f"Mean track length {np.mean([len(t) for t in tracks]):.3f}")
    
    logging.info("Triangulate points along tracks...")

    good_tracks, tracks_points3d = triangulate_tracks(
        tracks, frame_to_keypoints, intrinsics_mat, frame_to_extrinsic_mat,
        reproj_error_threshold=reproj_error_threshold
    )
    logging.info(f"Found {len(good_tracks)} good tracks!")
    logging.info(f"Mean good track length {np.mean([len(t) for t in good_tracks]):.3f}")
    anchor_frame_to_keypoints = get_inliers_keypoints(good_tracks, tracks_points3d, frame_to_keypoints)

    test_frames = frame_to_keypoints.keys() - set(anchor_frames)
    test_frame_to_keypoints = {f:k for f,k in frame_to_keypoints.items() if f in test_frames}

    frames_to_matches = match_test_with_anchor(
        test_frame_to_keypoints,
        anchor_frame_to_keypoints,
        intrinsics_mat=(intrinsics_mat if matching_use_intrinsics else None),
        method=matching_method,
        ratio_threshold=matching_ratio_threshold
    )

    test_frame_to_points2d = defaultdict(list)
    test_frame_to_points3d = defaultdict(list)
    for (test_frame, anchor_frame), matches in frames_to_matches.items():
        for match in matches:
            point2d = test_frame_to_keypoints[test_frame].get_point2d(match[0])
            point3d = anchor_frame_to_keypoints[anchor_frame].get_point3d(match[1])

            test_frame_to_points2d[test_frame].append(point2d)
            test_frame_to_points3d[test_frame].append(point3d)

    trajectory = estimate_test_poses(
        test_frame_to_points2d, test_frame_to_points3d, intrinsics_mat
    )
    Trajectory.write(Dataset.get_result_poses_file(out_dir), trajectory)
