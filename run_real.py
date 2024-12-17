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

    # run a combo first then z
    results = []
    errors = []

    print("Starting combo + z experiment")
    for i in range(7, 19):
        print(f"Running on {i} images")
        max_base_image = min(i, 14)
        images = [f'./data/real/scene/img{j}.jpg' for j in range(max_base_image)]
        images += [f'./data/real/scene/img_extra{j}.jpg' for j in range(max(i - 14, 0))]
        recovered_k = calibrate_from_real_images(images)
        error = get_relative_error(K, recovered_k)
        results.append(recovered_k)
        errors.append(error)
    
    write_results(results, 'results/real/results_combo_first.csv')
    write_errors(errors, 'results/real/errors_combo_first.csv')

    # run z first then combo
    results = []
    errors = []

    print("Starting z + combo experiment")
    for i in range(7, 19):
        print(f"Running on {i} images")
        images = [f'./data/real/scene/img{j}.jpg' for j in range(7)]

        # load z rotations first
        for j in range(min(max(i - 7, 0), 4)):
            images.append(f'./data/real/scene/img_extra{j}.jpg')

        # load combo rotations
        for j in range(max(i - 11, 0)):
            images.append(f'./data/real/scene/img{j}.jpg')
        
        recovered_k = calibrate_from_real_images(images)
        error = get_relative_error(K, recovered_k)
        results.append(recovered_k)
        errors.append(error)
    
    write_results(results, 'results/real/results_z_first.csv')
    write_errors(errors, 'results/real/errors_z_first.csv')


if __name__ == "__main__":
    main()