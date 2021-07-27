AT3DCV 2021 - Challenge 02 - Task 1

You are given
- images of 3 different sensors (colour = col, infrared (left) = ir1, infrared (right) = ir2) acquired with a RealSense D435
- the 3 x 78 images include a 7x9 checkerboard with 20mm x 20mm squares
- corresponding files (acquired at the same time) have the same file name

They are in the folders:
.\col\
.\ir1\
.\ir2\

=========================================

Your task is:

1. Calibrate each sensor individually (intrinsic calibration) where you calculate the camera matrix K and the distortion coefficients d. Use a pinhole camera model and two radial distortion coefficients. Use subpixel corner refinement.
Provide the code you were using.

Hint: You should check the detection of the checkerboard to not include outliers / wrong detections in your calibration routine.
You may be interested to use the following functions:
cv2.findChessboardCorners(...)
cv2.cornerSubPix(...)
# for visualization and debugging
cv2.drawChessboardCorners(...)
cv2.calibrateCamera(...)
# the following flags may be interesting for you
cv2.CALIB_ZERO_TANGENT_DIST
cv2.CALIB_FIX_K3

2. Perform two extrinsic calibrations where you calculate the rotation (R), translation (t) and the fundamental matrix (F) of

2.1. A stereo set with "ir1" and "ir2"
2.2. A stereo set with "col" and "ir1" [Optional]

Provide the code you were using.

=========================================

For 1. and 2. you can compare some results with the "task_01_result_hint.txt"
