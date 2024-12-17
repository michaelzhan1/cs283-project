import numpy as np
from utils.points import generate_points_random
from utils.calibration import calibrate
from utils.camera import build_camera_matrix, get_rotated_camera
from utils.results import write_errors, write_results, get_relative_error


def run_reconstruct():
    results = []
    errors = []

    for _ in range(50):
        fx = np.random.randint(500, 1000)
        fy = np.random.randint(500, 1000)
        u = np.random.randint(-100, 100)
        v = np.random.randint(-100, 100)
        s = np.random.uniform(0, 1)

        # original camera
        P, K, C = build_camera_matrix(fx, fy, u, v ,s, np.array([0, 0, 0, 1]))

        # orientations (degree rotations in (x, y, z))
        orientations = [
            (0, 0, 0),
            (0, -10, 0),
            (0, 10, 0),
            (10, 0, 0),
            (-10, 0, 0),
            (0, 0, 45),
            (0, 0, -45)
        ]

        cameras = [get_rotated_camera(K, C, *o, deg=True) for o in orientations]

        points_3dh = generate_points_random(100)
        points_2dh = [points_3dh @ cam.T for cam in cameras]
        points_2d = [pts[:, :2] / pts[:, 2].reshape(-1, 1) for pts in points_2dh]

        recovered_k = calibrate(*points_2d)
        error = get_relative_error(K, recovered_k)

        results.append(recovered_k)
        errors.append(error)

    write_results(results, "./results/noiseless/results.csv")
    write_errors(errors, "./results/noiseless/errors.csv")
        

def main():
    run_reconstruct()

if __name__ == "__main__":
    main()