import numpy as np
import matplotlib.pyplot as plt

def visualize_voxel_grid(loaded_voxel_grid):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Get the coordinates of the voxels
    filled = loaded_voxel_grid > 0.5
    x, y, z = filled.nonzero()
    ax.scatter(x, y, z, zdir='z', c='red', marker='s')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


loaded_voxel_grid = np.load('generated_shape.npy')
visualize_voxel_grid(loaded_voxel_grid)


