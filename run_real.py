import glob
import numpy as np
from utils.calibration import calibrate_from_real_images
from utils.results import get_relative_error, write_results, write_errors

def main():
    # ground truth camera matrix from april calibration
    K = np.array([
        [2997, 0, 1553],
        [0, 2977, 1882],
        [0, 0, 1]
    ])

    results = []
    errors = []

    for i in range(7, 18):
        print(f"Running on {i} images")
        max_base_image = min(i, 14)
        images = [f'./data/real/scene/img{i}.jpg' for i in range(max_base_image)]
        images += [f'./data/real/scene/img_extra{i}.jpg' for i in range(max(i - 13, 0))]
        recovered_k = calibrate_from_real_images(images)
        error = get_relative_error(K, recovered_k)
        results.append(recovered_k)
        errors.append(error)
    
    write_results(results, 'results/real/results.csv')
    write_errors(errors, 'results/real/errors.csv')


if __name__ == "__main__":
    main()