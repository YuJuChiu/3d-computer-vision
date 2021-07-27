AT3DCV 2021 - Challenge 02 - Task 2

You are given
- images of 3 different sensors (colour = col, infrared (left) = ir1, infrared (right) = ir2) acquired with a RealSense D435
- the 3 x 2 images include two scene with multiple objects. The first scene is acquired with an additional IR dot pattern projection, the second without.
- corresponding files (acquired at the same time) have the same file name

They are in the folders:
.\col\
.\ir1\
.\ir2\

=========================================

Your task is:

1. Undistort the images with the help of the calibration parameters from Task 1 and save the images as PNGs to the folders
	.\col\undist\
	.\ir1\undist\
	.\ir2\undist\

Hint: If you did not do Task 1, please use the calibration parameters provided in the file "task_01_result_hint.txt".
You may be interested to use the following functions:
img_undist = cv2.undistort(img, K, distortion, None, K)

2. Draw epipolar lines for "0000000040.png" [Optional]
2.1. Randomly select 5 points in col (and 5 points in ir1) in the undistorted images.
2.2. Draw the corresponding 5 epipolar lines on ir1 (and 5 epipolar lines on col) on these images.
2.3. Rectify the images and draw them side by side with their (horizontal) epipolar lines.

You may be interested to use the following functions:
cv2.computeCorrespondEpilines(...)
cv2.stereoRectify(...)
cv2.initUndistortRectifyMap(...)
cv2.remap(...)

3. Disparity estimation comparison
3.1. Rectify the images of
	.\ir1\undist\
	.\ir2\undist\
3.2. Estimate the disparity with the stereo pair "0000000039.png" and the stereo pair "0000000040.png"
3.3. What difference do you observe and why? Write a short explanation (2-3 sentences).
The disparity map for "0000000039.png" is less noisy than for "0000000040.png". As the projectors add some artificially structures on "0000000039.png", the image could be better recovered.
Hint: If you did not do Task 1, please use the calibration parameters provided in the file "task_01_result_hint.txt".
You may be interested to use the following functions:
stereo = cv2.StereoBM_create(...)
