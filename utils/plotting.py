import numpy as np
import matplotlib.pyplot as plt

def points_to_image(points2d, h=300, w=300, r=5):
    """Convenience function that converts matplotlib points to an image"""
    arr = np.ones((h, w, 3))
    for i in range(h):
        for j in range(w):
            # check if pixel is within r of any point
            for k in range(points2d.shape[0]):
                point = points2d[k, :]
                if np.linalg.norm(np.array([i, j]) - point) < r:
                    if k == 0:
                        arr[i, j] = np.array([1, 0, 0])
                    elif k == 1:
                        arr[i, j] = np.array([0, 1, 0])
                    else:
                        arr[i, j] = np.array([0, 0, 1])
                    break
    return arr

def plot_points_as_image(*points2d, h=300, w=300, titles=None):
    """Plot points with same coordinate system as plt.imshow (inverted y axis, row-first)"""
    n = len(points2d)
    fig = plt.figure(figsize=(5 * n, 5))
    for i in range(n):
        points = points2d[i]
        ax = fig.add_subplot(1, n, i + 1)
        ax.scatter(points[:, 1], points[:, 0])
        ax.scatter([points[0, 1]], [points[0, 0]], c='r')
        ax.scatter([points[1, 1]], [points[1, 0]], c='g')
        ax.set_aspect('equal')
        ax.set_xlim(-w // 2, w // 2)
        ax.set_ylim(h // 2, -h // 2)
        if titles is not None:
            ax.set_title(titles[i])
    plt.show()

def plot_multiple_images(*images, titles=None):
    """Plot multiple images in a row"""
    n = len(images)
    fig = plt.figure(figsize=(5 * n, 5))
    for i in range(n):
        ax = fig.add_subplot(1, n, i + 1)
        ax.imshow(images[i])
        if titles is not None:
            ax.set_title(titles[i])
    plt.show()

def get_title_from_orientation(orientation):
    """Get a plot title from an (x,y,z) orientation"""
    if orientation == (0, 0, 0):
        return "Original"
    title = "Rotation: "
    if orientation[0] != 0:
        title += f"x={orientation[0]} "
    if orientation[1] != 0:
        title += f"y={orientation[1]} "
    if orientation[2] != 0:
        title += f"z={orientation[2]} "
    return title[:-1]
