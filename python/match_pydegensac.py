import numpy as np
import matplotlib.pyplot as plt
import cv2
import pydegensac
from time import time
from copy import deepcopy

#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
#We will draw correspondences found and the geometric transformation between the images.

def draw_matches(kps1, kps2, tentatives, img1, img2, H, mask):
    if H is None:
        print ("No homography found")
        return
    matchesMask = mask.ravel().tolist()
    h,w,ch = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, H)
    #print (dst)
    #Ground truth transformation
    dst_GT = cv2.perspectiveTransform(pts, H_gt)
    img2_tr = cv2.polylines(img2,[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA)
    img2_tr = cv2.polylines(deepcopy(img2_tr),[np.int32(dst_GT)],True,(0,255,0),3, cv2.LINE_AA)
    # Blue is estimated, green is ground truth homography
    draw_params = dict(matchColor = (255,255,0), # draw matches in yellow color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img_out = cv2.drawMatches(img1.astype(np.uint8),kps1,img2_tr.astype(np.uint8),kps2,tentatives,None,**draw_params)
    plt.figure()
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(img_out, interpolation='nearest')
    plt.show()
    return

def verify_pydegensac(kps1, kps2, tentatives, th = 4.0,  n_iter = 2000):
    src_pts = np.float32([ kps1[m.queryIdx].pt for m in tentatives ]).reshape(-1,2)
    dst_pts = np.float32([ kps2[m.trainIdx].pt for m in tentatives ]).reshape(-1,2)
    H, mask = pydegensac.findHomography(src_pts, dst_pts, th, 1, n_iter)
    print ('pydegensac found {} inliers'.format(int(deepcopy(mask).astype(np.float32).sum())))
    return H, mask



def match_pictures_pydegensac(img1, img2, threshold=0.75):
    th = 100
    n_iter = 5000

    det = cv2.SIFT_create()
    kp1, des1 = det.detectAndCompute(img1,None)
    kp2, des2 = det.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [False for i in range(len(matches))]

    # SNN ratio test
    for i,(m,n) in enumerate(matches):
        if m.distance < threshold*n.distance:
            matchesMask[i]=True
    good_matches = [m[0] for i, m in enumerate(matches) if matchesMask[i] ]


    cv2_H, cv2_mask = verify_pydegensac(kp1, kp2, good_matches, th, n_iter)

    good_matches = np.array(good_matches)[cv2_mask]

    final_matches = []
    match_indexes = np.zeros(len(kp1), dtype=bool)
    for match in good_matches:
        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt
        final_matches.append([p1[0], p1[1], p2[0], p2[1]])
        match_indexes[match.queryIdx] = True

    des_model = np.array(des1)[match_indexes]
    kp_model = np.array(kp1)[match_indexes]
    

    return des_model, kp_model, np.array(final_matches)
