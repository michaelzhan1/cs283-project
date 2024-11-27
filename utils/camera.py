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
