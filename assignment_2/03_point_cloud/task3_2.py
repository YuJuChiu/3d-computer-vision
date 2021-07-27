import cv2
from cv2 import aruco
import numpy as np
import glob
from ply_writer import write_ply
import numpy as np
import matplotlib.pyplot as plt


K_ir1 = np.array([[388.425466037048, 0.0, 321.356734811229],
                [0.0, 387.559412128232, 244.543659354387],
                [0.0, 0.0, 1.000000]])
d_ir1 = np.array([0.00143845958426059,-0.00410315309358759,0.0,0.0,0.0])

R = np.array([[0.999999506646425, -3.18339774658664e-05, 0.000992820983631579],
                [3.15905844318835e-05, 0.999999969447414, 0.000245167709718939],
                [-0.000992828757961677, -0.000245136224969466, 0.999999477099508]])
T = np.array([-49.9430087222935, 0.0126441058712290, -0.0678600809461142])

K_ir2 = np.array([[390.034619271096, 0.0, 321.390633361907],
                [0.0, 389.119919973996, 244.648608218415],
                [0.0, 0.0, 1.000000]])
d_ir2 = np.array([0.00241762888488943,-0.00118610336539317,0.0,0.0,0.0])

size = (640,480)
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K_ir1, d_ir1,
                                                                  K_ir2, d_ir2,
                                                                  size, R, T)


images = glob.glob('ir1/*.png')

marker_length = 0.4
marker_spacing = 0.2
aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
aruco_params = aruco.DetectorParameters_create()

board = cv2.aruco.GridBoard_create(5, 7, marker_length, marker_spacing, aruco_dict)

images_ir1 = glob.glob('ir1/undist/*.png')
images_ir2 = glob.glob('ir2/undist/*.png')

for idx, fname in enumerate(images):

    img = cv2.imread(fname)
    file_name = 'disparity_' + str(idx) + '.npy'
    disp = np.load(file_name)
    # print(disp)
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = np.logical_and(disp < 800, disp > 200)
    points = points[mask]
    colors = colors[mask]

    out_fn = 'out_individual_' + str(idx) + '.ply'
    out_points = points
    # print(points.shape)
    out_colours = colors
    write_ply(out_fn, out_points, out_colours)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    marker_corners, ids, _ = aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        size_of_marker = 0.004 # side lenght of the marker in meter,
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(marker_corners, size_of_marker , K_ir1, d_ir1,)
        # _, H = aruco.find_homograpy
        # R, _ = cv2.Rodrigues(rvecs)
        # T = np.squeeze(tvecs, axis=0)
        # print(R.shape, T.shape)
        length_of_axis = 0.01
        imaxis = aruco.drawDetectedMarkers(img, marker_corners, ids)
        for i in range(len(tvecs)):
            imaxis = aruco.drawAxis(imaxis, K_ir1, d_ir1, rvecs[i], tvecs[i], length_of_axis)
        plt.figure()
        plt.imshow(imaxis)
        plt.show()

    Trans = np.identity(4)
    Trans[:3, :3] = R
    Trans[:3, 3] = T
    # print(Trans.shape, points.T.shape)
    out_fn = 'out_corrected_' + str(idx) + '.ply'
    n,m = points.shape
    z = np.ones((n, 1))
    points = np.append(points, z, axis=1)
    out_points = np.matmul(Trans, points.T)
    out_points_corrected = out_points.T[:,:3]
    out_colours_corrected = out_colours
    # print(out_colours_corrected.shape)
    write_ply(out_fn, out_points_corrected, out_colours_corrected)


