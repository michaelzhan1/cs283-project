import numpy as np

from utils.calibration import calibrate
from utils.camera import build_camera_matrix, get_rotated_camera
from utils.plotting import get_title_from_orientation
from utils.points import generate_points_random, jitter_image_points
from utils.results import get_relative_error, write_results, write_errors

# experiment loop

# original camera
P, K, C = build_camera_matrix(1000, 1000, 0, 0, 0, np.array([0, 0, 0, 1]))

# orientations (degree rotations in (x, y, z))
orientations = [
    (0, 0, 0),
    (0, 0, 5),
    (0, 0, -5),
    (0, -5, 0),
    (0, 5, 0),
    (5, 0, 0),
    (-5, 0, 0),
]

name = 'big_combine'

cams = [get_rotated_camera(K, C, *o, deg=True) for o in orientations]
titles = [get_title_from_orientation(o) for o in orientations]

results = []
errors = []
for i in range(10):
    # generate points
    points_3dh = generate_points_random(100)

    points_2dh = [points_3dh @ cam.T for cam in cams]
    points_2d = [pts[:, :2] / pts[:, 2].reshape(-1, 1) for pts in points_2dh]
    points_2d_jittered = [jitter_image_points(pts, sigma=1) for pts in points_2d]

    
    recovered_k = calibrate(*points_2d_jittered)
    error = get_relative_error(K, recovered_k)

    results.append(recovered_k)
    errors.append(error)

write_results(results, f'results/results_{name}.csv')
write_errors(errors, f'results/errors_{name}.csv')
