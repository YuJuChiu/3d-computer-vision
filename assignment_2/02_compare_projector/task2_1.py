import numpy as np
import cv2
import glob

# col
images = glob.glob('col/*.png')

K_col = np.array([[662.593701688052, 0.0, 324.857607968018],
                  [0.0, 658.422641634482, 224.715217487322],
                  [0.0, 0.0, 1.000000]])
d_col = np.array([0.155208391239907,-0.360250096753537,0.0,0.0,0.0])

for fname in images:
    img = cv2.imread(fname)
    # undistort
    img_undist = cv2.undistort(img, np.float64(K_col), np.float64(d_col), None, np.float64(K_col))
    filename = './col/undist' + fname[3:]
    cv2.imwrite(filename, img_undist)


# ir1
images = glob.glob('ir1/*.png')

K_ir1 = np.array([[388.425466037048, 0.0, 321.356734811229],
                [0.0, 387.559412128232, 244.543659354387],
                [0.0, 0.0, 1.000000]])
d_ir1 = np.array([0.00143845958426059,-0.00410315309358759,0.0,0.0,0.0])

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