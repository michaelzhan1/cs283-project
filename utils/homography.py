import numpy as np
from scipy.interpolate import RegularGridInterpolator

def get_normalizing_transform(X):
    """Get a normalizing transformation matrix for a set of identified points"""
    N = X.shape[0]

    # get centroid
    c_x = X[:, 0].mean()
    c_y = X[:, 1].mean()

    # get T
    denom = (np.sqrt((X[:, 0] - c_x) ** 2 + (X[:, 1] - c_y) ** 2 )).sum()
    s = N * 2 ** 0.5 / denom
    t_x = -s * c_x
    t_y = -s * c_y
    T = np.eye(3) * s
    T[:, 2] = np.array([t_x, t_y, 1])
    return T

def get_homography(X1, X2, normalize=False):
    """Get the homography matrix that maps X1 to X2"""
    # get normalizing transformation matrices
    T1 = get_normalizing_transform(X1)
    T2 = get_normalizing_transform(X2)

    n = X1.shape[0]

    # normalize each matrix
    X1n = np.hstack([X1, np.ones((n, 1))]) @ T1.T
    X2n = np.hstack([X2, np.ones((n, 1))]) @ T2.T

    # build A
    A = np.array([]).reshape(0, 9)
    for i in range(n):
        xip, yip, wip = X2n[i, :]
        xi = X1n[i, :].reshape(1, 3)
        row1 = np.hstack([np.zeros((1, 3)), -wip* xi, yip * xi])
        row2 = np.hstack([wip * xi, np.zeros((1, 3)), -xip * xi])
        A = np.vstack([A, row1, row2])

    # solve for homography in normalized coordinates
    _, _, V = np.linalg.svd(A)
    Hn = V[8, :].reshape(3, 3)

    # undo normalize transformation
    H = np.linalg.inv(T2) @ Hn @ T1

    # normalize to det(H) = 1
    if normalize:
        det = np.linalg.det(H)
        factor = np.sign(det) * np.abs(det) ** (1 / 3)
        H /= factor
    return H

def apply_homography(Iin, H, bounds=None):
    """Apply a homography to a full image"""
    n, m = Iin.shape[:2]

    # find bounds of image, then build mesh in rectified space
    I_corners = np.array([[0, 0], [0, m - 1], [n - 1, 0], [n - 1, m - 1]])
    Iph_corners = np.hstack([I_corners, np.ones((4, 1))]) @ H.T
    Ip_corners = Iph_corners[:, :2] / Iph_corners[:, 2].reshape(-1, 1)

    if bounds is None:
        imin, jmin = Ip_corners.min(axis=0)
        imax, jmax = Ip_corners.max(axis=0)
    else:
        imin, jmin, imax, jmax = bounds

    ii, jj = np.meshgrid(np.linspace(imin, imax, n), np.linspace(jmin, jmax, m), indexing='ij')
    Xp = np.vstack([ii.ravel(), jj.ravel(), np.ones(n * m)]).T

    # find corresponding points in the original space by solving the equation
    # X^T = H^-1 @ X'^T  ->  H @ X^T = X'^T
    Xh = np.linalg.lstsq(H, Xp.T, rcond=None)[0].T
    X = Xh[:, :2] / Xh[:, 2].reshape(-1, 1)

    # For a point in Ip, its corresponding point in I is found at a coordinate in X
    res = np.zeros((n, m, 3))
    for i in range(3):
        I = Iin[:, :, i]
        interpolator = RegularGridInterpolator(
            (np.arange(I.shape[0]), np.arange(I.shape[1])),
            I,
            bounds_error=False,
            fill_value=0
        )
        Ip = interpolator(X).reshape(m, n)
        res[:, :, i] = Ip
    return np.clip(res, 0, 1)