import numpy as np
import sys

from utils.calibration import calibrate
from utils.camera import build_camera_matrix, get_rotated_camera
from utils.plotting import get_title_from_orientation
from utils.points import generate_points_random, jitter_image_points
from utils.results import get_relative_error, write_results, write_errors

def run_z_angle_exp(z_orientation):
    # original camera
    P, K, C = build_camera_matrix(1000, 1000, 0, 0, 0, np.array([0, 0, 0, 1]))

    # orientations (degree rotations in (x, y, z))
    orientations = [
        (0, 0, 0),
        (0, -10, 0),
        (0, 10, 0),
        (10, 0, 0),
        (-10, 0, 0),
    ]

    orientations.append((0, 0, z_orientation))
    orientations.append((0, 0, -z_orientation))
    name = f'z{z_orientation}'

    cams = [get_rotated_camera(K, C, *o, deg=True) for o in orientations]
    titles = [get_title_from_orientation(o) for o in orientations]

    results = []
    errors = []
    for _ in range(50):
        # generate points
        points_3dh = generate_points_random(100)

        points_2dh = [points_3dh @ cam.T for cam in cams]
        points_2d = [pts[:, :2] / pts[:, 2].reshape(-1, 1) for pts in points_2dh]
        points_2d_jittered = [jitter_image_points(pts, sigma=5) for pts in points_2d]

        
        recovered_k = calibrate(*points_2d_jittered)
        error = get_relative_error(K, recovered_k)

        results.append(recovered_k)
        errors.append(error)

    write_results(results, f'results/results_{name}.csv')
    write_errors(errors, f'results/errors_{name}.csv')

def run_range_exp(count, max_x, max_y, max_z):
    if count % 2 == 1:
        print('count must be even')
        sys.exit(1)

    # original camera
    P, K, C = build_camera_matrix(1000, 1000, 0, 0, 0, np.array([0, 0, 0, 1]))

    # orientations (degree rotations in (x, y, z))
    orientations = [
        (0, 0, 0),
        (0, -10, 0),
        (0, 10, 0),
        (10, 0, 0),
        (-10, 0, 0),
    ]

    for i in range(1, count // 2 + 1):
        orientations.append((0, 0, i * max_z / (count // 2)))
        orientations.append((0, 0, -i * max_z / (count // 2)))
        # orientations.append((0, i * max_y / (count // 2), 0))
        # orientations.append((0, -i * max_y / (count // 2), 0))
        # orientations.append((i * max_x / (count // 2), 0, 0))
        # orientations.append((-i * max_x / (count // 2), 0, 0))
    
    cams = [get_rotated_camera(K, C, *o, deg=True) for o in orientations]

    name = f'range{count}'

    results = []
    errors = []
    for _ in range(50):
        # generate points
        points_3dh = generate_points_random(100)

        points_2dh = [points_3dh @ cam.T for cam in cams]
        points_2d = [pts[:, :2] / pts[:, 2].reshape(-1, 1) for pts in points_2dh]
        points_2d_jittered = [jitter_image_points(pts, sigma=4) for pts in points_2d]

        
        recovered_k = calibrate(*points_2d_jittered)
        error = get_relative_error(K, recovered_k)

        results.append(recovered_k)
        errors.append(error)
    
    write_results(results, f'results/range/results_{name}.csv')
    write_errors(errors, f'results/range/errors_{name}.csv')

def run_extra_combo_exp(num_extra):
    # original camera
    P, K, C = build_camera_matrix(1000, 1000, 0, 0, 0, np.array([0, 0, 0, 1]))

    # orientations (degree rotations in (x, y, z))
    orientations = [
        (0, 0, 0),
        (0, 0, -10),
        (0, 0, 10),
        (0, -10, 0),
        (0, 10, 0),
        (10, 0, 0),
        (-10, 0, 0),
    ]

    # add a combined orientation
    for i in range(1, num_extra + 1):
        x_sign = 1 if i % 2 == 0 else -1
        y_sign = 1 if i % 4 == 0 else -1
        z_sign = 1 if i % 8 == 0 else -1
        orientations.append((10 * x_sign, 10 * y_sign, 45 * z_sign))
    
    cams = [get_rotated_camera(K, C, *o, deg=True) for o in orientations]

    name = f'combo{num_extra}'

    results = []
    errors = []
    for _ in range(50):
        # generate points
        points_3dh = generate_points_random(100)

        points_2dh = [points_3dh @ cam.T for cam in cams]
        points_2d = [pts[:, :2] / pts[:, 2].reshape(-1, 1) for pts in points_2dh]
        points_2d_jittered = [jitter_image_points(pts, sigma=4) for pts in points_2d]

        
        recovered_k = calibrate(*points_2d_jittered)
        error = get_relative_error(K, recovered_k)

        results.append(recovered_k)
        errors.append(error)
    
    write_results(results, f'results/combo/results_{name}.csv')
    write_errors(errors, f'results/combo/errors_{name}.csv')

def main():
    # for z_orientation in range(10, 91, 10):
    #     run_z_angle_exp(z_orientation)
    # for count in range(2, 11, 2):
    #     run_range_exp(count, 10, 10, 90)
    
    for num_extra in range(1, 9):
        run_extra_combo_exp(num_extra)


if __name__ == "__main__":
    main()

