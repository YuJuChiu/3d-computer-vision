AT3DCV 2021 - Challenge 02 - Task 3

You are given
- images of 3 different sensors (colour = col, infrared (left) = ir1, infrared (right) = ir2) acquired with a RealSense D435
- the 3 x 5 images include two scene with multiple objects and an ArUcO marker as world anchor. All scenes are acquired with an additional IR dot pattern projection.
- corresponding files (acquired at the same time) have the same file name

They are in the folders:
.\col\
.\ir1\
.\ir2\

=========================================

Your task is:

1. Disparity estimation
1.1. Undistort the images of ir1 and ir2 with the help of the calibration parameters ("task_01_result_hint.txt") and save the images as PNGs to the folders
	.\ir1\undist\
	.\ir2\undist\
1.2. Rectify the images of
	.\ir1\undist\
	.\ir2\undist\
1.3. Estimate the disparity with the rectified image pairs and save the disparities in numpy files as
	.\disparity_<X>.npy
	where <X> = 0, 1, 2, 3, 4.

2. Point cloud fusion
2.1. Calculate an individual 3D point cloud for each image with the disparity estimates of part 1.3. and save each point cloud in a *.ply file (function is provided in "ply_writer.py") as
	.\out_individual_<X>.ply
	where <X> = 0, 1, 2, 3, 4.
Hint: If you did not do part 1, please use the disparity given in the numpy files .\disparity_<X>.npy
Hint: To ease visualization, you may want to mask your disparity between [200, 800]
You may be interested to use the following functions:
cv2.reprojectImageTo3D(...)
cv2.cvtColor(...)

2.2. Implement ArUcO marker detection and get the poses between camera and ArUcO marker for
	.\ir1\
Hint: Take care with the metric units from the ArUcO translation and use the following dictionary:
aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
You may be interested to have a look into this tutorial:
https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html

2.3. Bring the point clouds from "0000000029.png", ..., "0000000039.png" into the ArUcO reference frame and save each point cloud in a *.ply file as
	.\out_<X>.ply
	where <X> = 0, 1, 2, 3, 4.
Hint: Take care of inverse transformation (lecture 1)

2.4. Observe your results in Meshlab (or your favourite *.ply viewer). What do you observe for point cloud "0000000039.png"? Write a short explanation (1-2 sentences).
The floor in front of the stuffs is represented more fragmental, while the wall behind is recovered better. The reason might be that, the floor has several sections of reflections.
2.5. [Optional] Marker detection is more stable on the colour images (without the dot pattern). Use the ArUcO detection on the colour images
	.\col\
to bring all point clouds "0000000029.png", ..., "0000000039.png" in a common reference system and save each point cloud in a *.ply file as
	.\out_corrected<X>.ply
	where <X> = 0, 1, 2, 3, 4.
Hint: Drawing a pose graph can help to get the transformation directions right (lecture 1)
