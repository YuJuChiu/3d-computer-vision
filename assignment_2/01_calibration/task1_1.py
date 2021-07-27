import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,7,0)
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)
objp = objp * 0.02
# col=False
# ir1=False
# ir2=True

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('col/*.png')

for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 9), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7, 9), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(30)

cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None,
                                                       flags=cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3)

print(mtx, dist)
#
# if ir1 == False:
#     # Arrays to store object points and image points from all the images.
#     objpoints = [] # 3d point in real world space
#     imgpoints = [] # 2d points in image plane.
#
#     images = glob.glob('ir1/*.png')
#     count = 0
#     valid_sensed = []
#     for fname in images:
#         valid_sensed.append(fname)
#
#     valid_sensed_idx = [0,3,4,8,13,14,15,65,66,68,69,71,74,75]
#     # for fname in images:
#     for idx in valid_sensed_idx:
#         print(valid_sensed[idx])
#         img = cv2.imread(valid_sensed[idx])
#         # img = cv2.imread(fname)
#         gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
#         # Find the chess board corners
#         ret, corners = cv2.findChessboardCorners(gray,(7,9),None)
#
#         # If found, add object points, image points (after refining them)
#         if ret == True:
#             objpoints.append(objp)
#             print(count)
#             print(fname)
#             corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#             imgpoints.append(corners2)
#
#             # Draw and display the corners
#             img = cv2.drawChessboardCorners(img, (7,9), corners2,ret)
#             cv2.imshow('img',img)
#             cv2.waitKey(3000)
#         count = count +1
#
#     cv2.destroyAllWindows()
#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None, flags = cv2.CALIB_ZERO_TANGENT_DIST+ cv2.CALIB_FIX_K3)
#
#     print(mtx,dist)

# # for ir2
# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.
#
# images = glob.glob('ir2/*.png')
# count = 0
# valid_sensed = []
# for fname in images:
#     valid_sensed.append(fname)
#
# valid_sensed_idx = [0,3,4,7,8,10,11,12,13,14,15,17,18,19,20,21,23,24,28,29,32,33,34,35,37,38,39,41,42,43,44,45,47,48,49,50,53,54,57,59,60,61,64,65,66,68,69,71,72,74,75,76,77]
# # for fname in images:
# for idx in valid_sensed_idx:
#     # print(valid_sensed[idx])
#     img = cv2.imread(valid_sensed[idx])
#     # img = cv2.imread(fname)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
#     # Find the chess board corners
#     ret, corners = cv2.findChessboardCorners(gray,(7,9),None)
#     print(ret)
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#         print(count)
#         # print(fname)
#         corners2 = cv2.cornerSubPix(gray,corners,(7,7),(-1,-1),criteria)
#         imgpoints.append(corners2)
#
#         # Draw and display the corners
#         img = cv2.drawChessboardCorners(img, (7,9), corners2,ret)
#         cv2.imshow('img',img)
#         cv2.waitKey(30)
#     count = count +1
#
# cv2.destroyAllWindows()
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None, flags = cv2.CALIB_ZERO_TANGENT_DIST+ cv2.CALIB_FIX_K3)
#
# print(mtx,dist)
