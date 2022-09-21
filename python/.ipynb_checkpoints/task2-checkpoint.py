import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from hw5 import *
from matching import match_pictures, extract_pose_and_points, find_3D_correspondences
from bundle_adjustment import bundle_adjustment
from util import get_limits
from match_pydegensac import match_pictures_pydegensac

def create_3D_model(img1, img2, K, scalar, threshold=0.75, use_pydegensac=False, use_ROOT=False):
    #used in task 4.5
    if use_pydegensac:
        des_model, kp_model, matches = match_pictures_pydegensac(img1, img2, threshold)
        P1, P2, X, uv1, uv2, inliers = extract_pose_and_points(matches, K, use_ransac=False)
    else:
        des_model, kp_model, matches = match_pictures(img1, img2, threshold, use_ROOT=use_ROOT)
        P1, P2, X, uv1, uv2, inliers = extract_pose_and_points(matches, K)

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
    x = np.vstack((x, np.ones(n)))

    #scale model by the scaling factor from 2.3
    x[:3] *= scalar
    if use_pydegensac:
        np.savetxt('./data/model_degensac/des_model', np.array(des_model))
        np.savetxt('./data/model_degensac/X.txt', x)
        np.savetxt('./data/model_degensac/uv1.txt', uv1)
        np.savetxt('./data/model_degensac/uv2.txt', uv2)
    else:
        np.savetxt('./data/model/des_model', np.array(des_model))
        np.savetxt('./data/model/X.txt', x)
        np.savetxt('./data/model/uv1.txt', uv1)
        np.savetxt('./data/model/uv2.txt', uv2)

    return des_model, kp_model, x, uv1, uv2




if __name__ == '__main__':
    img1 = cv.imread('../3DModelPictures/IMG_0509.jpeg')  # queryImage
    img2 = cv.imread('../3DModelPictures/IMG_0510.jpeg')  # trainImage
    I1 = plt.imread('../3DModelPictures/IMG_0509.jpeg') / 255.0  # To draw point cloud
    I2 = plt.imread('../3DModelPictures/IMG_0510.jpeg') / 255.0  # To draw point cloud
    I2 = I2.reshape((I2.shape[1], I2.shape[0], I2.shape[2]))
    K = np.loadtxt('./data/calibration_data/newcameramtx.txt')
    
    scalar = 2.47 # Found in task 2.3

    des_model, kp_model, X, uv1, uv2 = create_3D_model(img1, img2, K, scalar, threshold=0.45)

    xmin, xmax, ymin, ymax, zmin, zmax = get_limits(X)

    draw_point_cloud(
        X, I2, uv2, xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    plt.show()

        


