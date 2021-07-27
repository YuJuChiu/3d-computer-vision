import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# flags = (cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST)
flags = (cv2.CALIB_FIX_K3)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,7,0)
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)
objp = objp * 0.02
# print(objp)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints_l = [] # 2d points in image plane.
imgpoints_r = [] # 2d points in image plane

images_right = glob.glob('ir1/*.png')
images_left = glob.glob('ir2/*.png')


valid_sensed_idx = [0,3,4,8,13,14,15,65,66,68,69,71,74,75]
for idx in valid_sensed_idx:
        img_l = cv2.imread(images_left[idx])
        img_r = cv2.imread(images_right[idx])

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, (7, 9), None)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, (7, 9), None)

        # If found, add object points, image points (after refining them)
        if ret_l == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray_l,corners_l,(11,11),(-1,-1),criteria)
            imgpoints_l.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img_l, (7,9), corners2,ret_r)
            cv2.imshow(images_left[idx], img_l)
            cv2.waitKey(20)

        if ret_r == True:
            # objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray_r,corners_r,(11,11),(-1,-1),criteria)
            imgpoints_r.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img_r, (7,9), corners2,ret_r)
            cv2.imshow(images_right[idx], img_r)
            cv2.waitKey(20)

cv2.destroyAllWindows()

K_ir1 = np.array([[388.425466037048, 0.0, 321.356734811229],
                [0.0, 387.559412128232, 244.543659354387],
                [0.0, 0.0, 1.000000]])
d_ir1 = np.array([0.00143845958426059,-0.00410315309358759,0.0,0.0,0.0])

K_ir2 = np.array([[390.034619271096, 0.0, 321.390633361907],
                [0.0, 389.119919973996, 244.648608218415],
                [0.0, 0.0, 1.000000]])
d_ir2 = np.array([0.00241762888488943,-0.00118610336539317,0.0,0.0,0.0])

img_shape = gray_l.shape[::-1]
# print(img_shape)
ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, K_ir1, d_ir1, K_ir2, d_ir2, img_shape, criteria=criteria, flags=flags)

print('F = {}\nR = {}\nT = {}'.format(F, R, T))