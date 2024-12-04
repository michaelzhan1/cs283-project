import numpy as np
import cv2

def build_camera_matrix(fx, fy, u, v, s, C):
    K = np.diag([fx, fy, 1])
    K[0, 2] = u
    K[1, 2] = v

    P = K @ (np.hstack([np.eye(3), -(C[:3] / C[3]).reshape(-1, 1)]))
    return P, K, C

def get_rotated_camera(K, C, x=0, y=0, z=0):
    """Apply a rotation to a camera matrix"""
    R = np.eye(3)
    R = R @ cv2.Rodrigues(np.array([x, y, z]))[0]
    P = K @ np.hstack([R, -(C[:3] / C[3]).reshape(-1, 1)])
    return P

def get_relative_error(K1, K2):
    """Get the relative error between two camera matrices. Trim small values to avoid division by zero"""
    diff = np.abs(K1 - K2)
    diff[diff < 1e-10] = 0

    dividend = K1
    dividend[np.abs(dividend) < 1e-10] = 0
    dividend[dividend == 0] = 1

    return diff / dividend

def display_relative_error(err):
    fx_err = err[0, 0]
    fy_err = err[1, 1]
    skew_err = err[0, 1]
    u_err = err[0, 2]
    v_err = err[1, 2]

    print(f"fx error: {fx_err}")
    print(f"fy error: {fy_err}")
    print(f"skew error: {skew_err}")
    print(f"u error: {u_err}")
    print(f"v error: {v_err}")
