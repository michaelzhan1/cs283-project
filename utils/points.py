import numpy as np

def generate_points_random(n, seed=0):
    """Generate n random points in 3D space"""
    np.random.seed(seed)
    points = np.random.uniform(-1, 1, (n, 4))
    points[:, 0] *= 2
    points[:, 1] *= 2
    points[:, 2] += 5
    points[:, 3] = 1
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

def jitter_image_points(points, sigma=1, seed=0):
    np.random.seed(seed)
    noise = np.random.normal(scale=sigma, size=points.shape)
    return points + noise