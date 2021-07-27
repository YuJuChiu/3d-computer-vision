import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utilities import *

#################################################################
###        Read dataset  :  Don't modify anything             ###
#################################################################

# read data from camera 0
img_0 = plt.imread("data_for_task2/rgb_full/0.png")[:,:,:3]

depth_0 = plt.imread("data_for_task2/depth_full/0.png")[:,:,0]
depth_0 = depth_0 * (6-0.1) + 0.1 # rescale depth range from [0,1] to [0.1,4]

pose_0 = np.loadtxt("data_for_task2/pose_cam/0.txt")

pose_0_inv = np.linalg.inv(pose_0)
K_0 = np.loadtxt("data_for_task2/intrinsic_cam/0.txt")
K_0_inv = np.linalg.inv(K_0)

# read data from camera 1
img_1 = plt.imread("data_for_task2/rgb_full/1.png")[:,:,:3]
depth_1 = plt.imread("data_for_task2/depth_full/1.png")[:,:,0]
depth_1 = depth_1 * (6-0.1) + 0.1 # rescale depth range from [0,1] to [0.1,4]

pose_1 = np.loadtxt("data_for_task2/pose_cam/1.txt")
pose_1_inv = np.linalg.inv(pose_1)

K_1 = np.loadtxt("data_for_task2/intrinsic_cam/1.txt")
K_1_inv = np.linalg.inv(K_1)

######################################################################
###     To do 1: Complete the code for per pixel forward warping   ###
###              with Nearest Neighbor interpolation               ###
######################################################################

#### Note : when we play with coordinate, order is (x,y)
####        while when we access the image, order is (y,x)

x,y = 100, 40
point_0 = [x,y,1]
depth = depth_0[y,x]

####  step 1. Convert image coordinate into camera_0's coordinate 
####          system of z = 1 by using inverse intrinsic

point_0 = np.matmul(K_0_inv, point_0)

####  step 2. Scale the coordinate by depth to bring it to 3D point
####          and make it as homogeneous 3D coordinate (i.e. [x,y,z,1])

point_0 = depth * point_0
point_0 = np.hstack((point_0, 1))

####  step 3. Camera's extrinsic is recorded in pyrender's orientation
####          Re-orient the camera's orientation to pyrender's orientation
####          by mutliplying correct rotational factor
factor = np.array([[1,0,0,0],
                   [0,-1,0,0],
                   [0,0,-1,0],
                   [0,0,0,1]])
point_0 = np.matmul(factor, point_0)

####  step 4. Express the point into world coordinate system by using 
####          pose of camera_0 Here, pose_0's direction is *Cam0 -> World*

point_0 = np.matmul(pose_0, point_0)

####  step 5. Express the point into camera_1's coordinate system.
####          Here, pose_1's direction is *Cam1 -> World*

point_1 = np.matmul(pose_1_inv, point_0)

####  step 6. Re-orient the camera to pyrender's orientation
####          by mutliplying correct rotational factor
factor = np.array([[1,0,0,0],
                   [0,-1,0,0],
                   [0,0,-1,0],
                   [0,0,0,1]])
point_1 = np.matmul(factor, point_1)

####  step 7. Discard the homogeneous point (i.e. [x,y,z,1] -> [x,y,z] 
####          and divid by it's z to make z = 1 (i.e. [x',y',1])
####          --> now point is expressed in plane in front of camera
####              with distance of z = 1

point_1 = point_1 / point_1[2]
point_1 = point_1[:3]

####  step 8. Express the point into pixel coordinate system by using 
####          intrinsic K

point_1 = np.matmul(K_1, point_1)
print(point_1)

####  step 9. Change x,y to test whether warping gives similar result 

####  x, y = 100, 40 -> result = [113.577, 103.909]
####  x, y =  30, 60 -> result = [ 56.368, 139.554]
####  x, y = 260,340 -> result = [305.768, 346.573]
####  x, y = 100,100 -> result = [124.168, 160.247]

#########################################################################
###  To do 2: Complete the code for bilinear_interpolation_per_pixel  ###
###           from assignment_utilities.py                            ###
#########################################################################

coord = [320.3,130.712]

interpolated =  bilinear_interpolation_per_pixel(coord,img_0)
print(interpolated)

####  step . check with different coordinates to test whether the interpolation
####          gives reasonably similar result 

####  coord = [130.3,130.2]   -> result = [0.87482354 0.82745099 0.23890197]
####  coord = [20.33,230.212] -> result = [0.51291703 0.48206229 0.43240691]
####  coord = [320.3,130.712] -> result = [0.98243451 0.87909334 0.78184472]
####  coord = [111.11,222.22] -> result = [0.70980394 0.63529414 0.14901961]


