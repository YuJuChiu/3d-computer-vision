import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import os

# undistort the images
# ir1
images = glob.glob('ir1/*.png')

K_ir1 = np.array([[388.425466037048, 0.0, 321.356734811229],
                [0.0, 387.559412128232, 244.543659354387],
                [0.0, 0.0, 1.000000]])
d_ir1 = np.array([0.00143845958426059,-0.00410315309358759,0.0,0.0,0.0])

R = np.array([[0.999999506646425, -3.18339774658664e-05, 0.000992820983631579],
                [3.15905844318835e-05, 0.999999969447414, 0.000245167709718939],
                [-0.000992828757961677, -0.000245136224969466, 0.999999477099508]])
T = np.array([-49.9430087222935, 0.0126441058712290, -0.0678600809461142])

for fname in images:
    img = cv2.imread(fname)
    # undistort
    img_undist = cv2.undistort(img, np.float64(K_ir1), np.float64(d_ir1), None, np.float64(K_ir1))
    filename = './ir1/undist' + fname[3:]
    cv2.imwrite(filename, img_undist)


# ir2
images = glob.glob('ir2/*.png')

K_ir2 = np.array([[390.034619271096, 0.0, 321.390633361907],
                [0.0, 389.119919973996, 244.648608218415],
                [0.0, 0.0, 1.000000]])
d_ir2 = np.array([0.00241762888488943,-0.00118610336539317,0.0,0.0,0.0])

for fname in images:
    img = cv2.imread(fname)
    # undistort
    img_undist = cv2.undistort(img, np.float64(K_ir2), np.float64(d_ir2), None, np.float64(K_ir2))
    filename = './ir2/undist' + fname[3:]
    cv2.imwrite(filename, img_undist)

# rectify the images

size = (640,480)
#相机坐标系转换
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K_ir1, d_ir1,
                                                                 K_ir2, d_ir2,
                                                                 size, R, T)
#减小畸变
left_map1, leftmap2 = cv2.initUndistortRectifyMap(K_ir1, d_ir1, R1,
                                                P1, size=size, m1type=cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(K_ir2, d_ir2, R2,
                                                P2, size=size, m1type=cv2.CV_16SC2)

images_ir1 = glob.glob('ir1/undist/*.png')
images_ir2 = glob.glob('ir2/undist/*.png')

#测试图片
for idx, fname in enumerate(images):
    org = cv2.imread(images_ir1[idx])
    org2 = cv2.imread(images_ir2[idx])

    #显示线条，方便比较
    dst = cv2.remap(org, left_map1, leftmap2, cv2.INTER_LINEAR)
    for i in range(20):
        cv2.line(dst, (0, i*24), (640, i*24), (0,255,0), 1)

    dst2 = cv2.remap(org2, right_map1, right_map2, cv2.INTER_LINEAR)
    for i in range(20):
        cv2.line(dst2, (0, i*24), (640, i*24), (0,255,0), 1)

    dst_new = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    dst2_new = cv2.cvtColor(dst2,cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=23)
    disparity = stereo.compute(dst_new,dst2_new)

    file_name = 'disparity_' + str(idx) + '_test.npy'
    np.save(file_name, disparity)