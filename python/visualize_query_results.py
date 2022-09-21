import matplotlib.pyplot as plt
import numpy as np
from util import *

# This script uses example data. You will have to modify the
# loading code below to suit how you structure your data.


def visualize_query_result(I, X_matched, X_model, T_m2q, inliers, uv_m, K):
    model = '../visualization_sample_data'

    # If you have colors for your point cloud model, then you can use this.
    #c = np.loadtxt(f'{model}/c.txt')  # RGB colors [shape: num_points x 3].
    # Otherwise you can use this, which colors the points according to their Y.
    c = None

    # These control the location and the viewing target
    # of the virtual figure camera, in the two views.
    # You will probably need to change these to work
    # with your scene.
    lookfrom1 = np.array((0, -20, 5))
    lookat1 = np.array((0, 0, 6))
    lookfrom2 = np.array((25, -5, 10))
    lookat2 = np.array((0, 0, 10))

    # 'matches' is assumed to be a Nx2 array, where the
    # first column is the index of the 2D point in the
    # query image and the second column is the index of
    # its matched 3D point.


    # 'inliers' is assumed to be a 1D array of indices
    # of the good matches, e.g. as identified by your
    # PnP+RANSAC strategy.

    u_inliers = uv_m.T
    X_inliers = X_matched

    u_hat = project(K, T_m2q@X_inliers)

    e = np.linalg.norm(u_hat - u_inliers, axis=0)
    print(f'Mean reprojection error: {np.mean(e)}')
    fig = plt.figure(figsize=(10, 8))

    plt.subplot(221)
    plt.imshow(I)
    plt.scatter(*u_hat, marker='+', c=e)
    plt.xlim([0, I.shape[1]])
    plt.ylim([I.shape[0], 0])
    plt.colorbar(label='Reprojection error (pixels)')
    plt.title('Query image and reprojected points')

    plt.subplot(222)
    plt.hist(e, bins=50)
    plt.xlabel('Reprojection error (pixels)')
    plt.title(f'Mean reprojection error: {np.round(np.mean(e), 4)}')

    plt.subplot(223)
    draw_model_and_query_pose(X_model, T_m2q, K, lookat1,
                            lookfrom1, c=c, frame_size=0.6)
    plt.xlim([-1000, 4000])
    plt.ylim([3000, -1000])
    plt.title('Model and localized pose (top view)')

    plt.subplot(224)
    draw_model_and_query_pose(X_model, T_m2q, K, lookat2,
                            lookfrom2, c=c, frame_size=0.6)
    plt.xlim([-1000, 3000])
    plt.ylim([3000, 0])
    plt.title('Model and localized pose (side view)')

    plt.tight_layout()
    plt.show()
