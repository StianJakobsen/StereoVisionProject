import numpy as np
from cv2 import cv2 as cv
import matplotlib.pyplot as plt

from matching import match_query_and_model
from task2 import create_3D_model
from bundle_adjustment import refine_pose
from util import get_Rt
from visualize_query_results import visualize_query_result
from hw5 import *

from decimal import *

def localize(des_model, X, queryImage, K, threshold=0.75, weighted=False, use_ROOT=False):
    uv2, inliers = match_query_and_model(des_model, X.shape[1], queryImage, threshold, use_ROOT=use_ROOT)

    X = X[:, inliers]

    _, rvec, tvec, inliers = cv.solvePnPRansac(X[:3].T, uv2, K, np.zeros(4), reprojectionError=2)

    X = X[:, inliers[:, 0]]
    uv2 = uv2[inliers[:, 0]]

    weights = None
    if weighted:
        n = X.shape[1]
        weights = calculate_weights(n)

    R0 = get_Rt(rvec, tvec)

    p1 = p2 = p3 = 0
    p = np.array([p1, p2, p3])
    p = np.concatenate((p, tvec.ravel()))

    np.savetxt('./data/monte_carlo_data/p.txt', p)
    np.savetxt('./data/monte_carlo_data/uv2.txt', uv2)
    np.savetxt('./data/monte_carlo_data/X.txt', X)
    np.savetxt('./data/monte_carlo_data/R0.txt', R0)

    res = refine_pose(p, uv2, X, K, R0, weights)

    R = rotate_x(res.x[0]) @ rotate_y(res.x[1]) @ rotate_z(res.x[2]) @ R0
    Rt = np.vstack((np.hstack((R[:3, :3], np.reshape(res.x[3:6], (3, 1)))), np.array([0, 0, 0, 1])))

    return Rt, X, inliers, uv2, res.jac



def estimate_uncertainty(K, des_model, X, queryImage, weighted=False):
    _, X, _, _, J = localize(des_model, X, queryImage, K, weighted=weighted)

    n = X.shape[1]
    
    sigma_r = 1
    Sigma_r = (1 / sigma_r) * np.eye(2*n) # Avoid inverting this since it is a very large matrix and takes a lot of time
    Sigma_p = np.linalg.inv(J.T @ Sigma_r @ J)

    stds = np.sqrt(np.diag(Sigma_p))
    stds[:3] = stds[:3]*180/np.pi # Convert angles from rad to degrees
    stds[3:] *= 1000 # convert to mm

    return stds

def calculate_weights(n):
    sigma_u = 50
    sigma_v = 0.1
    sigma_diag = np.hstack((sigma_u**2*np.ones(n), sigma_v**2*np.ones(n)))
    sigma_r = np.diag(sigma_diag)

    L = np.linalg.cholesky(sigma_r)
    L_inv = np.linalg.inv(L)

    return L_inv


def Monte_Carlo(K_mean, sigma_K, p, uv2, X, R0, m=500):
    poses = np.zeros((m, 6))
    for i in range(m):
        K_params = np.random.normal(np.zeros(3), sigma_K)
        K_var = np.array([[K_params[0], 0, K_params[1]], [
                         0, K_params[0], K_params[2]], [0, 0, 0]])

        K = K_mean + K_var

        res = refine_pose(p, uv2, X, K, R0)
        poses[i] = res.x[:6]

    return poses


if __name__ == '__main__':
    img3 = cv.imread('../3DModelPictures/IMG_0513.jpeg')  # New query image
    img4 = cv.imread('../3DModelPictures/IMG_0514.jpeg')  # New query image
    img5 = cv.imread('../3DModelPictures/IMG_0511.jpeg')  # New query image

    # Load model
    des_model = np.loadtxt('./data/model/des_model').astype('float32')
    X = np.loadtxt('./data/model/X.txt')
    K = np.loadtxt('./data/calibration_data/newcameramtx.txt')

    # Select which subtask to run
    Task31 = False
    Task32 = True
    Task33 = True
    Task34 = True

    if Task31:
        query_img = img5
        T_m2q, X_matched, inliers, uv2, _ = localize(des_model, X, query_img, K)
        visualize_query_result(query_img, X_matched, X, T_m2q, inliers, uv2, K)
        
    if Task32:
        stds = estimate_uncertainty(K, des_model, X, img3)
        stds2 = estimate_uncertainty(K, des_model, X, img4)
        stds3 = estimate_uncertainty(K, des_model, X, img5)
        print(list(map(lambda l: float(round(Decimal(l),7)), stds))) # Print on decimal form instead of standard form, to easier analyze the differences
        print(list(map(lambda l: float(round(Decimal(l),7)), stds2)))
        print(list(map(lambda l: float(round(Decimal(l),7)), stds3)))
        print('----------------------------------------')

    if Task33:
        stds = estimate_uncertainty(K, des_model, X, img3, weighted=True)
        stds2 = estimate_uncertainty(K, des_model, X, img4, weighted=True)
        stds3 = estimate_uncertainty(K, des_model, X, img5, weighted=True)
        print(list(map(lambda l: float(round(Decimal(l),7)), stds))) # Print on decimal form instead of standard form, to easier analyze the differences
        print(list(map(lambda l: float(round(Decimal(l),7)), stds2)))
        print(list(map(lambda l: float(round(Decimal(l),7)), stds3)))
        print('----------------------------------------')

    if Task34:
        sigma_K1 = np.array([50, 0.1, 0.1])
        sigma_K2 = np.array([0.1, 50, 0.1])
        sigma_K3 = np.array([0.1, 0.1, 50])

        # Run localize to get and save initial estimates for the pose
        localize(des_model, X, img3, K)
        p = np.loadtxt('./data/monte_carlo_data/p.txt')
        uv2 = np.loadtxt('./data/monte_carlo_data/uv2.txt')
        X = np.loadtxt('./data/monte_carlo_data/X.txt')
        R0 = np.loadtxt('./data/monte_carlo_data/R0.txt')

        poses1 = Monte_Carlo(K, sigma_K1, p, uv2, X, R0)
        poses2 = Monte_Carlo(K, sigma_K2, p, uv2, X, R0)
        poses3 = Monte_Carlo(K, sigma_K3, p, uv2, X, R0)

        pose_cov1 = np.cov(poses1.T)
        pose_cov2 = np.cov(poses2.T)
        pose_cov3 = np.cov(poses3.T)

        print(list(map(lambda l: float(round(Decimal(l),7)), np.sqrt(np.diag(pose_cov1)))))
        print(list(map(lambda l: float(round(Decimal(l),7)), np.sqrt(np.diag(pose_cov2)))))
        print(list(map(lambda l: float(round(Decimal(l),7)), np.sqrt(np.diag(pose_cov3)))))
        print('----------------------------------------')
