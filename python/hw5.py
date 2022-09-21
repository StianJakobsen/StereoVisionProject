import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def epipolar_distance(F, uv1, uv2):
    """
    F should be the fundamental matrix (use F_from_E)
    uv1, uv2 should be 3 x n homogeneous pixel coordinates
    """
    n = uv1.shape[1]
    l2 = F@uv1
    l1 = F.T@uv2
    e = np.sum(uv2*l2, axis=0)
    norm1 = np.linalg.norm(l1[:2, :], axis=0)
    norm2 = np.linalg.norm(l2[:2, :], axis=0)
    return 0.5*e*(1/norm1 + 1/norm2)


def rotate_x(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]])


def rotate_y(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]])


def rotate_z(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c, -s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])


def project(K, X):
    """
    Computes the pinhole projection of a (3 or 4)xN array X using
    the camera intrinsic matrix K. Returns the pixel coordinates
    as an array of size 2xN.
    """
    uvw = K@X[:3, :]
    uvw /= uvw[2, :]
    return uvw[:2, :]


def draw_frame(K, T, scale=1):
    """
    Visualize the coordinate frame axes of the 4x4 object-to-camera
    matrix T using the 3x3 intrinsic matrix K.

    Control the length of the axes by specifying the scale argument.
    """
    X = T @ np.array([
        [0, scale, 0, 0],
        [0, 0, scale, 0],
        [0, 0, 0, scale],
        [1, 1, 1, 1]])
    u, v = project(K, X)
    plt.plot([u[0], u[1]], [v[0], v[1]], color='red')  # X-axis
    plt.plot([u[0], u[2]], [v[0], v[2]], color='green')  # Y-axis
    plt.plot([u[0], u[3]], [v[0], v[3]], color='blue')  # Z-axis


# From assignment 4

def estimate_H(xy, XY):
    # Tip: U,s,VT = np.linalg.svd(A) computes the SVD of A.
    # The column of V corresponding to the smallest singular value
    # is the last column, as the singular values are automatically
    # ordered by decreasing magnitude. However, note that it returns
    # V transposed.

    n = XY.shape[1]

    A = np.vstack((np.array([XY[0], XY[1], np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n), -XY[0]*xy[0], -XY[1]*xy[0], -xy[0]]).T,
                   np.array([np.zeros(n), np.zeros(n), np.zeros(n), XY[0], XY[1], np.ones(n), -XY[0]*xy[1], -XY[1]*xy[1], -xy[1]]).T))

    _, _, VT = np.linalg.svd(A)
    H = VT.T[:, -1].reshape((3, 3))

    return H


def decompose_H(H):
    # Tip: Use np.linalg.norm to compute the Euclidean length
    T1 = np.eye(4)
    T2 = np.eye(4)

    k1 = np.linalg.norm(H[:, 0])
    k2 = -k1

    r1 = H[:, 0] / k1
    r2 = H[:, 1] / k1
    r3 = np.cross(r1, r2)
    t = H[:, 2] / k1

    T1[:3, :4] = np.column_stack((r1, r2, r3, t))
    T1[:3, :3] = closest_rotation_matrix(np.column_stack((r1, r2, r3)))

    r1 = H[:, 0] / k2
    r2 = H[:, 1] / k2
    r3 = np.cross(r1, r2)
    t = H[:, 2] / k2

    T2[:3, :4] = np.column_stack((r1, r2, r3, t))
    T2[:3, :3] = closest_rotation_matrix(np.column_stack((r1, r2, r3)))

    if T1[2, 3] >= 0:
        T = T1
    else:
        T = T2

    return T


def closest_rotation_matrix(Q):
    U, _, V = np.linalg.svd(Q)
    R = U @ V
    # print(np.linalg.norm(R @ R.T - np.eye(3)))
    # print(np.linalg.norm(Q @ Q.T - np.eye(3)))
    return R


def SE3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def decompose_E(E):
    """
    Computes the four possible decompositions of E into a relative
    pose, as described in Szeliski 7.2.

    Returns a list of 4x4 transformation matrices.
    """
    U, _, VT = np.linalg.svd(E)
    R90 = np.array([[0, -1, 0], [+1, 0, 0], [0, 0, 1]])
    R1 = U @ R90 @ VT
    R2 = U @ R90.T @ VT
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    t1, t2 = U[:, 2], -U[:, 2]
    return [SE3(R1, t1), SE3(R1, t2), SE3(R2, t1), SE3(R2, t2)]


def choose_pose(E, xy1, xy2):
    T4 = decompose_E(E)
    best_num_visible = 0
    for i, T in enumerate(T4):
        P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        P2 = T[:3, :]
        X1 = triangulate_many(xy1, xy2, P1, P2)
        X2 = T@X1
        num_visible = np.sum((X1[2, :] > 0) & (X2[2, :] > 0))
        if num_visible > best_num_visible:
            best_num_visible = num_visible
            best_T = T
            best_X1 = X1
    T = best_T
    X = best_X1

    return T, X


def get_num_ransac_trials(sample_size, confidence, inlier_fraction):
    return int(np.log(1 - confidence)/np.log(1 - inlier_fraction**sample_size))


def estimate_E_ransac(xy1, xy2, K, distance_threshold, num_trials):

    # Tip: The following snippet extracts a random subset of 8
    # correspondences (w/o replacement) and estimates E using them.
    #   sample = np.random.choice(xy1.shape[1], size=8, replace=False)
    #   E = estimate_E(xy1[:,sample], xy2[:,sample])

    uv1 = K@xy1
    uv2 = K@xy2

    print('Running RANSAC with %g inlier threshold and %d trials...' %
          (distance_threshold, num_trials), end='')
    best_num_inliers = -1
    for i in range(num_trials):
        sample = np.random.choice(xy1.shape[1], size=8, replace=False)
        E_i = estimate_E(xy1[:, sample], xy2[:, sample])
        d_i = epipolar_distance(F_from_E(E_i, K), uv1, uv2)
        inliers_i = np.absolute(d_i) < distance_threshold
        num_inliers_i = np.sum(inliers_i)
        if num_inliers_i > best_num_inliers:
            best_num_inliers = num_inliers_i
            E = E_i
            inliers = inliers_i
    print('Done!')
    print('Found solution with %d/%d inliers' %
          (np.sum(inliers), xy1.shape[1]))
    return E, inliers


def estimate_E(xy1, xy2):
    n = xy1.shape[1]
    A = np.empty((n, 9))
    for i in range(n):
        x1, y1 = xy1[:2, i]
        x2, y2 = xy2[:2, i]
        A[i, :] = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1]

    _, _, VT = np.linalg.svd(A)
    return np.reshape(VT[-1, :], (3, 3))


def F_from_E(E, K):
    K_inv = np.linalg.inv(K)
    F = K_inv.T@E@K_inv
    return F


def draw_correspondences(I1, I2, uv1, uv2, F, sample_size=8):
    """
    Draws a random subset of point correspondences and their epipolar lines.
    """

    assert uv1.shape[0] == 3 and uv2.shape[0] == 3, 'uv1 and uv2 must be 3 x n arrays of homogeneous 2D coordinates.'
    sample = np.random.choice(
        range(uv1.shape[1]), size=sample_size, replace=False)
    uv1 = uv1[:, sample]
    uv2 = uv2[:, sample]
    n = uv1.shape[1]
    uv1 /= uv1[2, :]
    uv2 /= uv2[2, :]

    l1 = F.T@uv2
    l2 = F@uv1

    colors = plt.cm.get_cmap('Set2', n).colors
    plt.figure('Correspondences', figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(I1)
    plt.xlabel('Image 1')
    plt.scatter(*uv1[:2, :], s=100, marker='x', c=colors)
    for i in range(n):
        hline(l1[:, i], linewidth=1, color=colors[i], linestyle='--')
    plt.xlim([0, I1.shape[1]])
    plt.ylim([I1.shape[0], 0])

    plt.subplot(122)
    plt.imshow(I2)
    plt.xlabel('Image 2')
    plt.scatter(*uv2[:2, :], s=100, marker='o', zorder=10,
                facecolor='none', edgecolors=colors, linewidths=2)
    for i in range(n):
        hline(l2[:, i], linewidth=1, color=colors[i], linestyle='--')
    plt.xlim([0, I2.shape[1]])
    plt.ylim([I2.shape[0], 0])
    plt.tight_layout()
    plt.suptitle(
        'Point correspondences and associated epipolar lines (showing %d randomly drawn pairs)' % sample_size)


def draw_point_cloud(X, I1, uv1, xlim, ylim, zlim):
    assert uv1.shape[1] == X.shape[
        1], 'If you get this error message in Task 4, it probably means that you did not extract the inliers of all the arrays (uv1,uv2,xy1,xy2) before calling draw_point_cloud.'

    # We take I1 and uv1 as arguments in order to assign a color to each
    # 3D point, based on its pixel coordinates in one of the images.
    #c = I1[uv1[1, :].astype(np.int32), uv1[0, :].astype(np.int32), :]
    c = None
    # Matplotlib doesn't let you easily change the up-axis to match the
    # convention we use in the course (it assumes Z is upward). So this
    # code does a silly rearrangement of the Y and Z arguments.
    plt.figure('3D point cloud', figsize=(6, 6))
    ax = plt.axes(projection='3d')
    ax.scatter(X[0, :], X[2, :], X[1, :], c=c, marker='.', depthshade=False)
    ax.grid(False)
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_zlim([ylim[1], ylim[0]])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    plt.title('[Click, hold and drag with the mouse to rotate the view]')


def hline(l, **args):
    """
    Draws a homogeneous 2D line.
    You must explicitly set the figure xlim, ylim before or after using this.
    """

    # Surely you don't have a figure bigger than this!
    lim = np.array([-1e8, +1e8])
    a, b, c = l
    if np.absolute(a) > np.absolute(b):
        x, y = -(c + b*lim)/a, lim
    else:
        x, y = lim, -(c + a*lim)/b
    plt.plot(x, y, **args)


def triangulate_many(xy1, xy2, P1, P2):
    """
    Arguments
        xy: Calibrated image coordinates in image 1 and 2
            [shape 3 x n]
        P:  Projection matrix for image 1 and 2
            [shape 3 x 4]
    Returns
        X:  Dehomogenized 3D points in world frame
            [shape 4 x n]
    """
    n = xy1.shape[1]
    X = np.empty((4, n))
    for i in range(n):
        A = np.empty((4, 4))
        A[0, :] = P1[0, :] - xy1[0, i]*P1[2, :]
        A[1, :] = P1[1, :] - xy1[1, i]*P1[2, :]
        A[2, :] = P2[0, :] - xy2[0, i]*P2[2, :]
        A[3, :] = P2[1, :] - xy2[1, i]*P2[2, :]
        U, s, VT = np.linalg.svd(A)
        X[:, i] = VT[3, :]/VT[3, 3]
    return X


def project_camera_frame(K, T, scale):
    """
    Draw the axes of T and a pyramid, representing the camera.
    """
    s = scale
    X = []
    X.append(np.array([0, 0, 0, 1]))
    X.append(np.array([-s, -s, 1.5*s, 1]))
    X.append(np.array([+s, -s, 1.5*s, 1]))
    X.append(np.array([+s, +s, 1.5*s, 1]))
    X.append(np.array([-s, +s, 1.5*s, 1]))
    X.append(np.array([5.0*s, 0, 0, 1]))
    X.append(np.array([0, 5.0*s, 0, 1]))
    X.append(np.array([0, 0, 5.0*s, 1]))
    X = np.array(X).T
    u, v = project(K, T@X)
    lines = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    plt.plot([u[0], u[5]], [v[0], v[5]], color='#ff5555', linewidth=2)
    plt.plot([u[0], u[6]], [v[0], v[6]], color='#33cc55', linewidth=2)
    plt.plot([u[0], u[7]], [v[0], v[7]], color='#44aaff', linewidth=2)
    for (i, j) in lines:
        plt.plot([u[i], u[j]], [v[i], v[j]], color='black')


def draw_model_and_query_pose(X, T_m2q, K,
                              lookat=np.array((0.0, 0.0, 0.0)),
                              lookfrom=np.array((0.0, 0.0, -15.0)),
                              point_size=10,
                              frame_size=0.5,
                              c=None):
    """
              X: Point cloud model of [shape (3 or 4)xN].
          T_m2q: Transformation from model to query camera coordinates (e.g. as obtained from OpenCV's solvePnP).
              K: Intrinsic matrix for the virtual 'figure camera'.
    lookat|from: The viewing target and origin of the virtual figure camera.
     point_size: Radius of a point (in pixels) that is 1 unit away. (Points further away will appear smaller.)
     frame_size: The length (in model units) of the camera and coordinate frame axes.
              c: Color associated with each point in X [shape Nx3].
    """

    assert X.ndim == 2, 'X must be a (3 or 4)xN array'
    assert X.shape[1] > 0, 'X must have at least one point'

    X = X.copy()
    if X.shape[0] == 3:
        X = np.vstack([X, np.ones_like(X[0, :])])
    else:
        X = X/X[3, :]

    if c is None:
        c = X[1, :]
    else:
        c = c.copy()
        if np.max(c) > 1.0:
            c = c / 256.0

    # Create transformation from model to 'figure camera'
    T_f2m = np.eye(4)
    T_f2m[:3, 2] = (lookat - lookfrom)
    T_f2m[:3, 2] /= np.linalg.norm(T_f2m[:3, 2])
    T_f2m[:3, 0] = np.cross(np.array((0, 1, 0)), T_f2m[:3, 2])
    T_f2m[:3, 0] /= np.linalg.norm(T_f2m[:3, 0])
    T_f2m[:3, 1] = np.cross(T_f2m[:3, 2], T_f2m[:3, 0])
    T_f2m[:3, 1] /= np.linalg.norm(T_f2m[:3, 1])
    T_f2m[:3, 3] = lookfrom
    T_m2f = np.linalg.inv(T_f2m)

    # Transform point cloud model into 'figure camera' frame
    X_f = T_m2f@X
    visible = X_f[2, :] > 0
    not_visible = X_f[2, :] < 0
    X_f = X_f[:, visible]
    if np.sum(not_visible) > np.sum(visible):
        print(
            '[draw_model_and_query_pose] Most of point cloud is behind camera, is that intentional?')

    # Project point cloud with depth sorting
    T_q2m = np.linalg.inv(T_m2q)
    u = K@X_f[:3, :]
    u = u[:2, :]/u[2, :]
    i = np.argsort(-X_f[2, :])
    plt.scatter(*u[:, i], c=c[i], s=(point_size**2)/X_f[2, :], rasterized=True)

    # Project the coordinate frame axes of the localized query camera
    project_camera_frame(K, T_m2f@T_q2m, scale=frame_size)

    plt.axis('image')

    # Arbitrarily set axis limits to 2*principal point
    plt.xlim([0.0, K[0, 2]*2])
    plt.ylim([K[1, 2]*2, 0.0])
