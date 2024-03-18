from typing import Dict, Tuple
import numpy as np
import cv2 as cv
from sfm.keypoints import FrameKeypoints

def match_keypoints_pair(
        keypoints1: FrameKeypoints, keypoints2: FrameKeypoints,
        intrinsics_mat: np.array = None,
        method="bf",
        ratio_threshold=0.8) -> np.array:
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
            filtered_matches.append([m.queryIdx, m.trainIdx])
            points1.append(keypoints1.kp[m.queryIdx].pt)
            points2.append(keypoints2.kp[m.trainIdx].pt)
    filtered_matches = np.array(filtered_matches)
    if len(filtered_matches) == 0:
        return filtered_matches
    points1 = np.array(points1, dtype=np.float64)
    points2 = np.array(points2, dtype=np.float64)
    if intrinsics_mat is None:
        _, mask = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC)
    else:
        _, mask = cv.findEssentialMat(points1, points2, intrinsics_mat, cv.FM_RANSAC)
    if mask is None:
        return np.array([])
    mask = mask.ravel().astype(bool)
    inliers_matches = filtered_matches[mask]
    #logging.info(f"""Frames [{keypoints1.frame_id}/{keypoints2.frame_id}] Matches:{len(matches)} Filtered:{len(filtered_matches)} Inliers:{len(inliers_matches)}""")
    return inliers_matches

def match_anchor_keypoints(
        anchor_frames, 
        frame_to_keypoints: Dict[int, FrameKeypoints],
        **kwargs) -> Dict[Tuple[int, int], np.array]:
    """
    Matches keypoints for anchor frames
    """
    matches_for_frames = {}
    for i in range(len(anchor_frames) - 1):
        for j in range(i + 1, len(anchor_frames)):
            frame_id1, frame_id2 = anchor_frames[i], anchor_frames[j]
            matches = match_keypoints_pair(
                frame_to_keypoints[frame_id1],
                frame_to_keypoints[frame_id2],
                **kwargs
            )
            matches_for_frames[(frame_id1, frame_id2)] = matches
    return matches_for_frames

def match_test_with_anchor(
        test_frame_to_keypoints: Dict[int, FrameKeypoints],
        anchor_frame_to_keypoints: Dict[int, FrameKeypoints],
        **kwargs) -> Dict[Tuple[int, int], np.array]:
    frames_to_matches = {}
    for test_frame, test_keypoints in test_frame_to_keypoints.items():
        for anchor_frame, anchor_keypoints in anchor_frame_to_keypoints.items():
            matches = match_keypoints_pair(
                test_keypoints,
                anchor_keypoints,
                **kwargs
            )
            frames_to_matches[(test_frame, anchor_frame)] = matches
    return frames_to_matches