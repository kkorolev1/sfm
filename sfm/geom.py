import numpy as np

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

def project_points(points: np.array,
                   proj_mat: np.array) -> np.array:
    """
    points: (N, 3)
    proj_mat: (3, 4)
    """
    if points.shape[-1] == 3:
        points = convert_to_homo(points)
    return convert_from_homo((proj_mat @ points.T).T)

def reprojection_error(point2d: np.array, point3d: np.array, proj_mat: np.array) -> float:
    reproj_point = project_points(point3d, proj_mat)
    return np.linalg.norm(reproj_point - point2d)

def convert_from_homo(point: np.array) -> np.array:
    """
    point in homogenous coordinates: (N,D+1) or (D+1,) -> (N,D) or (D,)
    """
    d = point.shape[-1]
    return point[..., :d-1] / point[..., d-1][..., None]

def convert_to_homo(point: np.array) -> np.array:
    """
    point in euclidean coordinates: (N,D) or (D,) -> (N,D+1) or (D+1,)
    """
    if len(point.shape) == 1:
        return np.concatenate((point, [1]))
    return np.concatenate((point, np.ones((point.shape[0], 1))), axis=1)