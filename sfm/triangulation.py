from typing import Dict, Tuple, List
import numpy as np
import cv2 as cv
from scipy.optimize import least_squares

from sfm.keypoints import FrameKeypoints
from sfm.track import Track
from sfm.geom import convert_from_homo, convert_to_homo, reprojection_error
from sfm.utils import get_frame_to_proj_matrix

def triangulate_pairwise_estimate(
        proj_matrices: np.array,
        points2d: np.array,
        reproj_error_threshold: int) -> np.array:
    
    num_views = proj_matrices.shape[0]
    points4d = []
    for i in range(num_views - 1):
        for j in range(i + 1, num_views):
            point4d = cv.triangulatePoints(proj_matrices[i], proj_matrices[j], points2d[i], points2d[j]).ravel()
            points4d.append(point4d)
    points4d = np.array(points4d)

    # (num_views, num_views_pairs, 2)
    proj_points = convert_from_homo(np.einsum("nij,mj->nmi", proj_matrices, points4d))
    reproj_errors = np.linalg.norm(proj_points - points2d[:, None, :], axis=-1)
    num_frames_for_points = (reproj_errors < reproj_error_threshold).sum(axis=0)
    return points4d[np.argmax(num_frames_for_points)]


def triangulate_track(track: Track,
           frame_to_keypoints: Dict[int, FrameKeypoints],
           frame_to_proj_matrix: Dict[int, np.array],
           reproj_error_threshold: int = 10):
    proj_matrices = []
    points2d = []
    for frame_id, point_idx in zip(track.frames, track.items):
        proj_matrices.append(frame_to_proj_matrix[frame_id])
        points2d.append(frame_to_keypoints[frame_id].get_point2d(point_idx))
    proj_matrices = np.array(proj_matrices)
    points2d = np.array(points2d)

    point4d = triangulate_pairwise_estimate(
        proj_matrices, points2d,
        reproj_error_threshold=reproj_error_threshold
    )

    def res_fn(x):
        return (points2d - convert_from_homo(proj_matrices @ x)).ravel()

    ls_result = least_squares(res_fn, x0=point4d)
    point4d = ls_result.x

    reproj_errors = np.linalg.norm(points2d - convert_from_homo(proj_matrices @ point4d), axis=-1)
    if np.any(reproj_errors > reproj_error_threshold):
        return None

    return convert_from_homo(point4d)


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
            track, frame_to_keypoints, frame_to_proj_matrix,
            reproj_error_threshold=reproj_error_threshold
        )
        if point3d is not None:
            good_tracks.append(track)
            tracks_point3d.append(point3d)

    return good_tracks, tracks_point3d