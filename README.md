# Rotating Camera Self-Calibration
An implementation of rotating camera self-calibration following [this paper](https://link.springer.com/chapter/10.1007/3-540-57956-7_52) (Hartley 1994) and section 19.6 of Multiple View Geometry in Computer Vision by Hartley and Zisserman.

# At a Glance
The main experiment scripts are `run_april_calibration.py`, `run_real.py`, and `run_synthetic.py`. They run calibration of AprilTag boards, self-calibration of real images, and self-calibration on synthetic data, respectively. All inputs are stored in the `data/` folder, and all outputs are in the `results/` folder. `analysis.ipynb` was used for plot generation and analysis of results from the experiments. All other Jupyter notebooks are out-of-date and were used for developing utility modules. See more information regarding the repository components below.

**Note**: Due to potential issues with the AprilTag library, `run_april_calibration.py` is unreliable to run. After running it enough times, it should give an output.

# Repo Structure
The repository is structured as follows:

## Top level files
#### `run_*.py`
These are the scripts used to run ground-truth calibrations on AprilTags, or self-calibrations on real or synthetic data.

#### `requirements.txt`
A list of Python dependencies

#### `analysis.ipynb`
A notebook used for result analysis

## `assignment/`
This folder holds the notebook used to mimic a typical assignment problem from CS283, to be potentially used in a future assignment.

## `data/`
This holds the data used in the experiments. `AprilBoards.pkl` at the top level holds binary information regarding the AprilTags themselves, used in ground-truth calibration. The `synthetic/` folder holds pickled 3D points used in old notebooks. `real/aprilboards/` and `real/scene` hold images of AprilTags and images taken from the rotating camera, respectively, used in real data experiments.

## `old_notebooks/`
This contains old notebooks used during testing. These notebooks are **not** expected to work immediately, and may require moving back to the top level or additional editing to work. All important contents of these folders have been moved to scripts or modules, and what remains is mostly plots.

## `results/`
This contains the recovered intrinsic parameter matrices from the synthetic experiments, as well as the errors in those matrices from the original.

## `utils/`
This contains all of the modules/scripts that are used in the calibrations.

#### `april.py`
Contains functions to run ground-truth calibration on multiple images of an AprilTag board.

#### `calibration.py`
Contains functions for self-calibration in both synthetic and real settings.

#### `camera.py`
Contains functions for setting up a virtual camera in synthetic experiments.

#### `homography.py`
Contains functions for solving and applying homographies between two images or sets of points.

#### `plotting.py`
Mainly used in the notebooks. Contains functions for plotting multiple images or sets of points.

#### `points.py`
Used to generate and add noise to synthetic data.

#### `results.py`
Used to handle and calculate useful statistics for synthetic data outputs.