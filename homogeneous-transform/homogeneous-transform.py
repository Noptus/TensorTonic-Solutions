import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)

    is_single_point = points.ndim == 1
    points_2d = np.atleast_2d(points)

    homogeneous_points = np.concatenate(
        [points_2d, np.ones((points_2d.shape[0], 1))],
        axis=1,
    )

    transformed = (T @ homogeneous_points.T).T[:, :3]

    return transformed[0] if is_single_point else transformed