import numpy as np
import matplotlib.pyplot as plt

def generate_homogeneous_grid(h,w):
    yv, xv = np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
    hv = np.ones((h,w))
    homogeneous_grid = np.stack([xv,yv,hv],-1)
    
    return homogeneous_grid
    
def bilinear_interpolation_per_pixel(coord,img_source):
    
    # 00---------10
    # |    |      |
    # | S00| S10  |   value_x = S00 * value[11] + S11 * value[00] +
    # |----X------|             S10 * value[01] + S01 * value[10]
    # | S01| S11  |
    # 01---------11
    
    # get grid value for each corners
    coord_x_floor = np.floor(coord[0])
    coord_x_ceil  =  coord_x_floor + 1
    coord_y_floor = np.floor(coord[1])
    coord_y_ceil  =  coord_y_floor + 1
    
    coord_00 = np.stack([coord_y_floor,coord_x_floor],-1).astype(int)
    coord_01 = np.stack([coord_y_ceil,coord_x_floor],-1).astype(int)
    coord_10 = np.stack([coord_y_floor,coord_x_ceil],-1).astype(int)
    coord_11 = np.stack([coord_y_ceil,coord_x_ceil],-1).astype(int)

    # skip grid value outside of image range to keep away from range error
    if (coord_x_floor < 0 or coord_x_floor >= 640) or (coord_x_ceil < 0 or coord_x_ceil >= 640) \
       or (coord_y_floor < 0 or coord_y_floor >= 480) or (coord_y_ceil < 0 or coord_y_ceil >= 480):
           return 0
    
    #################################################################
    ###     To do : implement code to calculate S00,11,01,10      ###
    #################################################################
    
    x,y = coord
    
    S00 = np.abs(x - coord_x_floor) * np.abs(y - coord_y_floor)
    S11 = np.abs(x - coord_x_ceil) * np.abs(y - coord_y_ceil)
    S10 = np.abs(x - coord_x_floor) * np.abs(y - coord_y_ceil)
    S01 = np.abs(x - coord_x_ceil) * np.abs(y - coord_y_floor)
    
    
    # image is RGB, 3 channel
    rgb = []
    for chan in range(3):

        val_00 = img_source[coord_00[0],coord_00[1],chan]
        val_01 = img_source[coord_01[0],coord_01[1],chan]
        val_10 = img_source[coord_10[0],coord_10[1],chan]
        val_11 = img_source[coord_11[0],coord_11[1],chan]
        
        ##################################################################
        ###     To do : Use the formular to get interpolated value     ###
        ###             by summing with proper weights (S00 - S11)     ###
        ##################################################################
    
        each_chan = S00 * val_11 + S11 * val_00 + S10 * val_01 + S01 * val_10
        
        rgb.append(each_chan)
        
    interpolated = np.stack(rgb,0)
    return interpolated
    
    
def bilinear_interpolation_per_grid(grid,img_source):
    
    # 00---------10
    # |    |      |
    # | S00| S10  |   value_x = S00 * value[11] + S11 * value[00] +
    # |----X------|             S10 * value[01] + S01 * value[10]
    # | S01| S11  |
    # 01---------11
    
    n_pixels,_ = grid.shape # n_pixels = 480*640
    
    # get grid value for each corners
    grid_x_floor = np.floor(grid[:,0])
    grid_x_ceil  = grid_x_floor + 1
    grid_y_floor = np.floor(grid[:,1])
    grid_y_ceil =  grid_y_floor + 1
    
    grid_00 = np.stack([grid_x_floor, grid_y_floor],-1)
    grid_01 = np.stack([grid_x_floor, grid_y_ceil],-1)
    grid_10 = np.stack([grid_x_ceil, grid_y_floor],-1)
    grid_11 = np.stack([grid_x_ceil, grid_y_ceil],-1)
    
    # calculate mask to zero out grids pointing outside of image range
    mask = np.ones(grid_00.shape[0])
    mask *= (0 <= grid_00[:,0]) * (grid_00[:,0] < 640) * (0 <= grid_00[:,1]) * (grid_00[:,1] < 480)
    mask *= (0 <= grid_01[:,0]) * (grid_01[:,0] < 640) * (0 <= grid_01[:,1]) * (grid_01[:,1] < 480)
    mask *= (0 <= grid_10[:,0]) * (grid_10[:,0] < 640) * (0 <= grid_10[:,1]) * (grid_10[:,1] < 480)
    mask *= (0 <= grid_11[:,0]) * (grid_11[:,0] < 640) * (0 <= grid_11[:,1]) * (grid_11[:,1] < 480)
    mask = mask.reshape(n_pixels,1)
    
    # mask out grid values outside of image range to keep away from range error
    grid_00 = (grid_00*mask).astype(int)
    grid_01 = (grid_01*mask).astype(int)
    grid_10 = (grid_10*mask).astype(int)
    grid_11 = (grid_11*mask).astype(int)
        
    #################################################################
    ###     To do : implement code to calculate S00,11,01,10      ###
    #################################################################

    x = grid[:,0]
    y = grid[:,1]

    S00 = np.abs(x - grid_x_floor) * np.abs(y - grid_y_floor)
    S11 = np.abs(x - grid_x_ceil) * np.abs(y - grid_y_ceil)
    S10 = np.abs(x - grid_x_floor) * np.abs(y - grid_y_ceil)
    S01 = np.abs(x - grid_x_ceil) * np.abs(y - grid_y_floor)


    
    img = []
    
    # image is RGB, 3channel
    for chan in range(3):
        val_00 = img_source[grid_00[:,1],grid_00[:,0],np.ones(grid.shape[0],np.int)*chan]
        val_11 = img_source[grid_11[:,1],grid_11[:,0],np.ones(grid.shape[0],np.int)*chan]
        val_01 = img_source[grid_01[:,1],grid_01[:,0],np.ones(grid.shape[0],np.int)*chan]
        val_10 = img_source[grid_10[:,1],grid_10[:,0],np.ones(grid.shape[0],np.int)*chan]
        
        ###########################$######################################
        ###     To do : Use the formular to get interpolated value     ###
        ###             by summing with proper weights (S00 - S11)     ###
        ##########################$#######################################
        
        each_chan = S00 * val_11 + S11 * val_00 + S10 * val_01 + S01 * val_10
        
        img.append(each_chan)

    # mask out values which are pointed from masked grid values
    interpolated = np.stack(img,-1) * mask

    return interpolated
    
