import numpy as np

def generate_points_random(n, seed=None):
    """Generate n random points in 3D space"""
    if seed is not None:
        np.random.seed(seed)
    # generate random x and y coordinates
    points_xy = np.random.uniform(-5, 5, (n, 2))
    # add z and w coordinates
    points_z = np.random.uniform(15, 25, (n, 1))
    points_w = np.ones((n, 1))
    points = np.hstack([points_xy, points_z, points_w])
    return points

def generate_points(rows, cols):
    """Generate nxn grid of points in 3D space around Z axis"""
    center = np.array([0, 0, 5, 1])
    # make an n by n grid surrounding the center
    x = np.linspace(-0.1 * cols, 0.1 * cols, cols)
    y = np.linspace(-0.1 * rows, 0.1 * rows, rows)
    yy, xx = np.meshgrid(x, y)
    points = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    points = np.hstack([points, np.zeros((rows * cols, 1)), np.ones((rows * cols, 1))])
    points += center
    return points

def jitter_image_points(points, sigma=1, seed=None):
    """Add gaussian noise"""
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(scale=sigma, size=points.shape)
    return points + noise