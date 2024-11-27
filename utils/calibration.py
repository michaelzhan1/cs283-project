import numpy as np
from .homography import get_homography

def reconstruct_k(w_coeffs):
    """Reconstruct a camera matrix from the IAC conic coefficients"""
    w = np.array([
        [w_coeffs[0], w_coeffs[1], w_coeffs[2]],
        [w_coeffs[1], w_coeffs[3], w_coeffs[4]],
        [w_coeffs[2], w_coeffs[4], w_coeffs[5]]
    ])
    
    w /= w[2, 2]

    U = np.linalg.cholesky(w, upper=False)
    K = np.linalg.inv(U).T
    K /= K[2, 2]
    return K

def calibrate(*points_2d):
    """Perform self-calibration from a non-moving camera with constant internals.

    Takes a list of matching points across images and estimates the camera's
    intrinsic parameters. Assumes the first element of the list as the 
    reference image.
    
    @param points_2d, a list of Nx2 points used to determine the calibration.

    @return K, an estimation of the 3x3 intrinsic parameter matrix for the camera.
    """

    ref = points_2d[0]

    A = np.zeros((0, 6))

    for i in range(1, len(points_2d)):
        H = get_homography(ref, points_2d[i], normalize=True)
        H_inv = np.linalg.inv(H)

        h1, h2, h3, h4, h5, h6, h7, h8, h9 = H_inv.ravel()

        row1 = np.array([h1**2 - 1, 2*h1*h4, 2*h1*h7, h4**2, 2*h4*h7, h7**2])
        row2 = np.array([h1*h2, h1*h5+h2*h4 - 1, h1*h8+h2*h7, h4*h5, h4*h8+h5*h7, h7*h8])
        row3 = np.array([h1*h3, h1*h6+h3*h4, h1*h9+h3*h7 - 1, h4*h6, h4*h9+h6*h7, h7*h9])
        row4 = np.array([h2**2, 2*h2*h5, 2*h2*h8, h5**2 - 1, 2*h5*h8, h8**2])
        row5 = np.array([h2*h3, h2*h6+h3*h5, h2*h9+h3*h8, h5*h6, h5*h9+h6*h8 - 1, h8*h9])
        row6 = np.array([h3**2, 2*h3*h6, 2*h3*h9, h6**2, 2*h6*h9, h9**2 - 1])

        A = np.vstack([A, row1, row2, row3, row4, row5, row6])
    
    _, _, V = np.linalg.svd(A)
    w_coeffs = V[5, :]
    K = reconstruct_k(w_coeffs)
    return K