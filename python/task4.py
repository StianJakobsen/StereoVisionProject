import numpy as np
from cv2 import cv2 as cv
import matplotlib.pyplot as plt

from matching import extract_pose_and_points, match_pictures, find_3D_correspondences
from bundle_adjustment import bundle_adjustment, refine_pose
from hw5 import *
from task3 import localize
from visualize_query_results import visualize_query_result
from util import get_Rt
from task2 import create_3D_model
from util import get_limits

def create_3D_model_ORB(img1, img2, K, scalar):
    # ORB Detector
    orb = cv.ORB_create(nfeatures=50000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Brute Force Matching
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    print(f'Number of matches: {len(matches)}')

    final_matches = []
    match_indexes = np.zeros(len(kp1), dtype=bool)
    for match in matches:
        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt
        final_matches.append([p1[0], p1[1], p2[0], p2[1]])
        match_indexes[match.queryIdx] = True

    des_model = np.array(des1)[match_indexes]
    kp_model = np.array(kp1)[match_indexes]

    _, P2, X, uv1, uv2, inliers = extract_pose_and_points(np.array(final_matches), K)

    des_model = des_model[inliers]
    kp_model = kp_model[inliers]

    # Bundle adjustment
    n = X.shape[1]

    R0 = np.eye(4)
    R0[:3, :3] = P2[:3, :3]

    t = P2[:3, 3]
    p1 = p2 = p3 = 0
    p = np.array([p1, p2, p3])
    p = np.concatenate((p, t, np.ravel(X[:3, :n].T)))

    res = bundle_adjustment(p, uv1, uv2, K, R0, n)

    x = res.x[6:].reshape((n, 3)).T
    X = np.vstack((x, np.ones(n)))
    X[:3] *= scalar

    np.savetxt('./data/model_orb/des_model', np.array(des_model))
    np.savetxt('./data/model_orb/X.txt', X)

    return des_model, X, uv1, uv2


def localize_ORB(des_model, X, img3, K):
    orb = cv.ORB_create(nfeatures=50000)
    kp3, des3 = orb.detectAndCompute(img3, None)

    # Brute Force Matching
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_model, des3)
    print("Num matches: ", len(matches))

    final_matches = []
    match_indexes = np.zeros(X.shape[1], dtype=bool)
    for match in matches:
        p2 = kp3[match.trainIdx].pt
        final_matches.append([p2[0], p2[1]])
        match_indexes[match.queryIdx] = True

    uv2 = np.array(final_matches)
    inliers = match_indexes

    X = X[:, inliers]

    _, rvec, tvec, inliers = cv.solvePnPRansac(X[:3].T, uv2, K, np.zeros(4), iterationsCount=10000)

    if inliers is not None:
        print("Num inliers: ", len(inliers))
        X = X[:, inliers[:, 0]]
        uv2 = uv2[inliers[:, 0]]
    else:
        print("No inliers found")

    R0 = get_Rt(rvec, tvec)

    p1 = p2 = p3 = 0
    p = np.array([p1, p2, p3])
    p = np.concatenate((p, tvec.ravel()))

    res = refine_pose(p, uv2, X, K, R0)

    R = rotate_x(res.x[0]) @ rotate_y(res.x[1]) @ rotate_z(res.x[2]) @ R0
    Rt = np.vstack((np.hstack((R[:3, :3], np.reshape(res.x[3:6], (3, 1)))), np.array([0, 0, 0, 1])))

    return Rt, X, inliers, uv2, res.jac

#used in task 4.2
def create_3D_model_large(image_list, K, scalar, threshold=0.75):
    Xs = np.empty((4, 0))
    des_model_tot = np.empty((0, 128))
    n = len(image_list)
    T = np.eye(4)

    for i in range(n-1):
        des_model, _, matches = match_pictures(image_list[i], image_list[i+1], threshold)
        P1, P2, X, uv1, uv2, inliers = extract_pose_and_points(matches, K)

        # Bundle adjustment
        n = X.shape[1]

        R0 = np.eye(4)
        R0[:3, :3] = P2[:3, :3]

        t = P2[:3, 3]
        p1 = p2 = p3 = 0

        p = np.array([p1, p2, p3])
        p = np.concatenate((p, t, np.ravel(X[:3, :n].T)))

        res = bundle_adjustment(p, uv1, uv2, K, R0, n)

        X = res.x[6:].reshape((n, 3)).T
        X = np.vstack((X, np.ones(n)))

        R = rotate_x(res.x[0]) @ rotate_y(res.x[1]) @ rotate_z(res.x[2]) @ R0
        Rt = np.vstack((np.hstack((R[:3, :3], np.reshape(res.x[3:6], (3, 1)))), np.array([0, 0, 0, 1])))

        des_model = des_model[inliers]

        if i == 0:
            uv0 = uv1
            X0 = X
            relative_scale = 1

        T_inv = np.eye(4)
        T_inv[:3,:3] =  T[:3,:3].T
        T_inv[:3,3] = T[:3,3] * -1
        X = T_inv @ X

        if i != 0:
              relative_scale = find_3D_correspondences(uv0, X0, X, P1, K)

        if relative_scale != 0:
            print("scaling up: ", relative_scale)
            X[:3] /= relative_scale
            Xs = np.hstack((Xs, X))
            des_model_tot = np.vstack((des_model_tot, des_model))
        else:
            print('No correspondence')
            print(f'i: {i}, j={i+1}')
        
        T = Rt @ T
                
    # Bundle adjustment?
    Xs[:3] *= scalar

    np.savetxt('./data/model_large/des_model', np.array(des_model_tot))
    np.savetxt('./data/model_large/X.txt', Xs)

    return des_model_tot, Xs, uv1



if __name__ == '__main__':
    img1 = cv.imread('../3DModelPictures/IMG_0509.jpeg')
    img2 = cv.imread('../3DModelPictures/IMG_0510.jpeg')
    I2 = plt.imread('../3DModelPictures/IMG_0510.jpeg') / 255.0  # To draw point cloud
    I2 = I2.reshape((I2.shape[1], I2.shape[0], I2.shape[2]))
    img3 = cv.imread('../3DModelPictures/IMG_0513.jpeg')  # New query image to localize
    
    K = np.loadtxt('./data/calibration_data/newcameramtx.txt')
    scalar = 2.47

 
    Task42 = False
    Task43 = False
    Task44 = True
    Task45 = True

    if Task42:
        img1 = cv.imread('../3DModelPictures/IMG_0509.jpeg')  # queryImage
        img2 = cv.imread('../3DModelPictures/IMG_0510.jpeg')  # trainImage
        img3 = cv.imread('../3DModelPictures/IMG_0511.jpeg')  # queryImage
        img4 = cv.imread('../3DModelPictures/IMG_0512.jpeg')  # trainImage
        img5 = cv.imread('../3DModelPictures/IMG_0513.jpeg')  # trainImage

        image_list = [img1, img2, img3, img4, img5]

        # Comment if existing model exist and uncomment line under
        des_model, X, _ = create_3D_model_large(image_list, K, scalar, threshold=0.5)
        #X = np.loadtxt('./data/model_large/X.txt')

        uv = np.vstack((project(K, X), np.ones(X.shape[1])))

        xmin, xmax, ymin, ymax, zmin, zmax = get_limits(X)

        draw_point_cloud(X, I2, uv, xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
        plt.show()

        #Localization
        des_model = np.loadtxt('./data/model_large/des_model').astype('float32')

        Rt, X_matched, inliers, uv2, jac = localize(des_model, X, img3, K)

        visualize_query_result(img3, X_matched, X, Rt, inliers, uv2, K)

    if Task43:
        #Model creation (Can comment if model exist and load using code under)
        des_model, X, uv1, uv2 = create_3D_model_ORB(img1, img2, K, scalar)
        #X = np.loadtxt('./data/model_orb/X.txt')

        draw_point_cloud(X, I2, uv2, xlim=[-10, +10], ylim=[-10, +10], zlim=[10, 20])
        plt.show()

        #Localization 
        des_model = np.loadtxt('./data/model_orb/des_model').astype('uint8')

        Rt, X_matched, inliers, uv2, jac = localize_ORB(des_model, X, img3, K)

        visualize_query_result(img3, X_matched, X, Rt, inliers, uv2, K)
     
    
    if Task44:
        
        des_model, kp_model, X, uv1, uv2 = create_3D_model(img1, img2, K, scalar, threshold=0.45, use_ROOT=True)
        
        xmin, xmax, ymin, ymax, zmin, zmax = get_limits(X)
        draw_point_cloud(
            X, I2, uv2, xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
        plt.show()
    
    if Task45:
        #Model creation (Can comment if model exist and load using code under)
        des_model, kp_model, X, uv1, uv2 = create_3D_model(img1, img2, K, scalar, threshold=0.5, use_pydegensac=True)
        #X = np.loadtxt('./data/model_degensac/X.txt')

        xmin, xmax, ymin, ymax, zmin, zmax = get_limits(X)

        draw_point_cloud(
            X, I2, uv2, xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
        plt.show()

        des_model = np.loadtxt('./data/model_degensac/des_model').astype('float32')
        
        Rt, X_matched, inliers, uv2, jac = localize(des_model, X, img3, K)

        visualize_query_result(img3, X_matched, X, Rt, inliers, uv2, K)




