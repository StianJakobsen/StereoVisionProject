import numpy as np
from cv2 import cv2 as cv
import matplotlib.pyplot as plt

from hw5 import *


def match_pictures(img1, img2, threshold, use_ROOT=False):
    
    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if use_ROOT:
        des1 /= des1.sum(axis=1, keepdims=True)
        des1 = np.sqrt(des1)
        des2 /= des2.sum(axis=1, keepdims=True)
        des2 = np.sqrt(des2)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < threshold*n.distance:
            matchesMask[i] = [1, 0]
            good_matches.append([m])

    final_matches = []
    match_indexes = np.zeros(len(kp1), dtype=bool)
    for match in good_matches:
        p1 = kp1[match[0].queryIdx].pt
        p2 = kp2[match[0].trainIdx].pt
        final_matches.append([p1[0], p1[1], p2[0], p2[1]])
        match_indexes[match[0].queryIdx] = True

    des_model = np.array(des1)[match_indexes]
    kp_model = np.array(kp1)[match_indexes]

    print("Number of matches: ", len(final_matches))

    return des_model, kp_model, np.array(final_matches)


def extract_pose_and_points(matches, K, use_ransac=True):
    uv1 = np.vstack([matches[:, :2].T, np.ones(matches.shape[0])])
    uv2 = np.vstack([matches[:, 2:4].T, np.ones(matches.shape[0])])

    xy1 = np.linalg.inv(K) @ uv1
    xy2 = np.linalg.inv(K) @ uv2

    if use_ransac:
        confidence = 0.99
        inlier_fraction = 0.5
        distance_threshold = 4.0
        num_trials = get_num_ransac_trials(8, confidence, inlier_fraction)
        print('Running RANSAC: %d trials, %g pixel threshold' %
            (num_trials, distance_threshold))
        E, inliers = estimate_E_ransac(xy1, xy2, K, distance_threshold, num_trials)

        uv1 = uv1[:, inliers]
        uv2 = uv2[:, inliers]
        xy1 = xy1[:, inliers]
        xy2 = xy2[:, inliers]
    else:
        inliers = np.ones(uv1.shape[1], dtype=bool)
        
    E = estimate_E(xy1, xy2)
    P1 = np.hstack((np.eye(3), np.zeros(3).reshape(3, 1))) # Image 1 is chosen as origin of the world frame
    P2, X = choose_pose(E, xy1, xy2)

    return P1, P2, X, uv1, uv2, inliers


def match_query_and_model(des_model, N, queryImage, threshold, use_ROOT=False):
    sift = cv.SIFT_create()
    kp3, des3 = sift.detectAndCompute(queryImage, None)

    if use_ROOT:
        des_model /= des_model.sum(axis=1, keepdims=True)
        des_model = np.sqrt(des_model)
        des3 /= des3.sum(axis=1, keepdims=True)
        des3 = np.sqrt(des3)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_model, des3, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    good_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < threshold*n.distance:
            matchesMask[i] = [1, 0]
            good_matches.append([m])

    final_matches = []
    match_indexes = np.zeros(N, dtype=bool)
    for match in good_matches:
        p2 = kp3[match[0].trainIdx].pt
        final_matches.append([p2[0], p2[1]])
        match_indexes[match[0].queryIdx] = True

    return np.array(final_matches), match_indexes


def find_3D_correspondences(uv1, X1, X2, Rt, K):
    uv2 = project(K, X2)
    uv1 = uv1[:2]

    distances = np.array([])
    distances1 = np.array([])
    distances2 = np.array([]) 
    for i in range(uv2.shape[1]):
        for j in range(uv1.shape[1]):
            if np.linalg.norm(uv2[:,i] - uv1[:,j]) < 1:
                distances = np.append(distances, np.linalg.norm(X2[:3,i] - X1[:3,j]))
                distances1 = np.append(distances1, X2[:3,i])
                distances2 = np.append(distances2, X1[:3,j])
                if distances.shape[0] >= 10:
                    print("enough correspondences ", i, " ", j)
                    print(distances1.shape)
                    return distance_ratio(distances1, distances2)
    print(distances1.shape)
    return distance_ratio(distances1, distances2)


def distance_ratio(distances1, distances2):
    print(distances1.shape)
    if distances1.shape[0] <= 1: # Return 0 if list is empty
        return 0
    rs = np.array([])
    for i in range(1, distances1.shape[0]):
        r = np.linalg.norm(distances1[i] - distances1[i-1]) / np.linalg.norm(distances2[i] - distances2[i-1])
        rs = np.append(rs, r)

    return np.mean(rs)
