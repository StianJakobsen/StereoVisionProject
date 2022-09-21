import numpy as np
from cv2 import cv2 as cv
import glob
import matplotlib.pyplot as plt

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

chessboard_dim = (7, 10)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_dim[0]*chessboard_dim[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_dim[0],
                       0:chessboard_dim[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob('../CalibrationPictures/*.jpeg')

for i, fname in enumerate(images):
    print(i, fname)
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(
        gray, (chessboard_dim[0], chessboard_dim[1]), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(
            img, (chessboard_dim[0], chessboard_dim[1]), corners2, ret)
        img = cv.resize(img, (80, 104))
        cv.imshow('img', img)
        cv.waitKey(10)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

total_error = 0
errors = []
error_vecs = []
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    error_vec = imgpoints[i] - imgpoints2
    error_vecs.extend(error_vec)
    total_error += error
    errors.append(error)
print("mean error: {}".format(total_error/len(objpoints)))

plt.bar(np.arange(len(errors)), errors)
plt.savefig("../Figures/Task1_1_bar_plot.png")
plt.show()

error_vecs = np.array(error_vecs).reshape((len(error_vecs), 2))
plt.scatter(error_vecs[:, 0], error_vecs[:, 1])
plt.savefig("../Figures/Task1_1_scatter_plot.png")
plt.show()

ret, mtx, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv.calibrateCameraExtended(
    objpoints, imgpoints, gray.shape[::-1], None, None)


print("------- Task 1.2 ---------")

img = cv.imread('../3DModelPictures/IMG_0509.jpeg')
h, w = img.shape[:2]
std_mtx = np.array([[float(stdDeviationsIntrinsics[0]), 0, float(stdDeviationsIntrinsics[2])], [
                   0, float(stdDeviationsIntrinsics[1]), float(stdDeviationsIntrinsics[3])], [0, 0, 1]])
std_dist = np.array(stdDeviationsIntrinsics[4:9]).reshape((5,))
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))


#Test undistortion from normal distribution
for i in range(10):
    dist_rand = np.random.normal(dist, std_dist)
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        mtx, dist_rand, (w, h), 1, (w, h))
    
    # undistort
    dst = cv.undistort(img, mtx, dist_rand, None, newcameramtx)
    x, y, w_dst, h_dst = roi
    dst = dst[y:y+h_dst, x:x+w_dst]
    cv.imshow('dst', dst)
    cv.waitKey(0)
    
## Undistort images for task 2
images = glob.glob('../3DModelPictures/*jpeg')
for img_path in images:
    img = cv.imread(img_path)
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w_dst, h_dst = roi
    dst = dst[y:y+h_dst, x:x+w_dst]
    
    path = f"../3DModelPicturesUndistorted/{img_path.split('/')[-1]}"
    cv.imwrite(path, dst)

np.savetxt("./data/calibration_data/stdDeviationsIntrinsics.txt", stdDeviationsIntrinsics)
np.savetxt("./data/calibration_data/mtx.txt", mtx)
np.savetxt("./data/calibration_data/dist.txt", dist)
np.savetxt("./data/calibration_data/newcameramtx.txt", newcameramtx)