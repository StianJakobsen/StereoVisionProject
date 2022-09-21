import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from cv2 import cv2 as cv
from hw5 import *
from util import get_Rt


def residuals(p, uv1, uv2, K, R0, n):
    X = p[6:].reshape((n, 3))
    X = np.hstack((X, np.ones((n, 1))))

    R = rotate_x(p[0]) @ rotate_y(p[1]) @ rotate_z(p[2]) @ R0
    Rt = np.vstack(
        (np.hstack((R[:3, :3], np.reshape(p[3:6], (3, 1)))), np.array([0, 0, 0, 1])))

    uv1_hat = project(K, X.T)
    uv2_hat = project(K, Rt @ X.T)

    r = np.hstack((uv1_hat - uv1, uv2_hat - uv2))

    return (r.T).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points):
    m = n_points*2*n_cameras
    n = 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    A[2*n_points:, :6] = 1
    for s in range(n_points):
        for cam in range(n_cameras):
            A[s*2 + cam*2*n_points:(s+1)*2 + cam*2 *
              n_points, s*3 + 6:(s+1)*3 + 6] = 1
    return A


def bundle_adjustment(p, uv1, uv2, K, R0, n):
    def fun(p): return residuals(p, uv1[:2, :], uv2[:2, :], K, R0, n)

    n_cameras, n_points = 2, n
    A = bundle_adjustment_sparsity(n_cameras, n_points)
    print(f'\nMean reprojection error before bundle adjustment: {np.mean(np.sqrt(residuals(p, uv1[:2, :], uv2[:2, :], K, R0, n)**2))}\n')
    res = least_squares(fun, p, jac_sparsity=A, x_scale='jac', ftol=1e-4, method='trf', verbose=2)
    print(f'\nMean reprojection error after bundle adjustment: {np.mean(np.sqrt(residuals(res.x, uv1[:2, :], uv2[:2, :], K, R0, n)**2))}\n')
    return res


def residuals_pose(p, uv2, X, K, R0, weights):
    R = rotate_x(p[0]) @ rotate_y(p[1]) @ rotate_z(p[2]) @ R0
    Rt = np.vstack((np.hstack((R[:3, :3], np.reshape(p[3:6], (3, 1)))), np.array([0, 0, 0, 1])))

    uv2_hat = project(K, Rt @ X)
    r = uv2_hat - uv2.T
    if weights is not None:
        r = r.reshape((1, 2*r.shape[1]))
        r = r @ weights

    return r.ravel()


def refine_pose(p, uv2, X, K, R0, weights=None):
    def fun(p): return residuals_pose(p, uv2, X, K, R0, weights)

    res = least_squares(fun, p, method='trf')
    return res


# For task 4.1
def residuals_pose2(p, uv2, X, K, sigma_K, weights):
    Rt = get_Rt(p[:3], p[3:6])
    uv2_hat = project(K, Rt @ X)
    r = uv2_hat - uv2.T
    if weights is not None:
        r = r.reshape((1, 2*r.shape[1]))
        r = r @ weights

    return r.ravel()


def refine_pose2(p, uv2, X, K, sigma_K, weights=None):
    def fun(p): return residuals_pose2(p, uv2, X, K, sigma_K, weights)

    res = least_squares(fun, p, x_scale='jac', method='trf')

    return res