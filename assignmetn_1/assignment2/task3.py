import pyrender,trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


# scenario = 'test'
scenario = '0'

# scene
scene = pyrender.Scene(bg_color=[0,0,0], ambient_light=[1.0,1.0,1.0])
r = pyrender.OffscreenRenderer(640,480)

# light
light = pyrender.SpotLight(color=np.ones(3), intensity=1.0,
                           innerConeAngle=np.pi/16.0,
                           outerConeAngle=np.pi/6.0)
scene.add(light,pose=np.identity(4))

# lumbar mesh
trimesh_lumbar = trimesh.load('dataset_for_task3/SK55M_texturized_bonemesh.ply')
lumbar_mesh = pyrender.Mesh.from_trimesh(trimesh_lumbar)
lumbar_node = scene.add(lumbar_mesh)

# tip marker mesh
sm = trimesh.creation.uv_sphere(radius=0.005)
sm.visual.vertex_colors = [1.0, 0.0, 0.0]
tip_mesh = pyrender.Mesh.from_trimesh(sm)
tip_node = scene.add(tip_mesh)

######################################################################
###     To do 1: Setup intirnsic matrix K obtained from task 1     ###
######################################################################

# camera intrinsic K from task1
K = np.array([[500.52156682,   0,         320.29761927],
              [  0,         500.31160744, 239.58855425],
              [  0,           0,           1        ]])
              
camera = pyrender.IntrinsicsCamera(fx=K[0,0],fy=K[1,1],cx=K[0,2],cy=K[1,2],znear=0.001,zfar=3)
scene.add(camera,pose=np.identity(4))

######################################################################
###          To do 2: Calculate "world_to_camera" pose             ###
###    from "marker_to_world" pose and "camera_to_marker" pose     ###
######################################################################

# marker to world pose
cam_marker_to_world = np.loadtxt("dataset_for_task3/pose_cam/{}.txt".format(scenario),delimiter=" ")
# cam_to_marker_pose
cam_to_marker = np.loadtxt("dataset_for_task3/pose_head/{}.txt".format(scenario),delimiter=" ")

###  step 1. calculate "camera_to_world" pose 

camera_to_world = np.dot(cam_marker_to_world, cam_to_marker)

###  step 2. calculate "world_to_camera" pose
###          Hint : inverting direction is inversing the matrix

world_to_camera = np.linalg.inv(camera_to_world)

######################################################################
###             To do 3: By using "world_to_camera" pose,          ###
###   calculate "lumbar_to_camrea" pose and "tip_to_camera" pose   ###
######################################################################

# marker to world pose
lumbar_marker_to_world = np.loadtxt("dataset_for_task3/pose_phantom/{}.txt".format(scenario),delimiter=" ")
# lumbar_to_marker_pose
lumbar_to_marker = np.loadtxt("dataset_for_task3/pose_lumbar/{}.txt".format(scenario),delimiter=" ")

### step 1. calculate "lumbar_to_world" pose

lumbar_to_world = np.dot(lumbar_marker_to_world, lumbar_to_marker)

### step 2. calculate "lumbar_to_camera" pose by using
###         "lumbar_to_world" pose and "world_to_camera" pose

lumbar_to_cam = np.dot(world_to_camera, lumbar_to_world)

scene.set_pose(lumbar_node,pose=lumbar_to_cam)
# color,depth = r.render(scene)
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(color)
# plt.subplot(1,2,2)
# plt.imshow(depth)
# plt.show()
# marker to world pose
tooltip_marker_to_world = np.loadtxt("dataset_for_task3/pose_tool/{}.txt".format(scenario),delimiter=" ")
# tooltip_to_marker_pose
tooltip_to_marker = np.loadtxt("dataset_for_task3/pose_tooltip/{}.txt".format(scenario),delimiter=" ")

### step 1. calculate "tooltip_to_world" pose

tooltip_to_world = np.dot(tooltip_marker_to_world, tooltip_to_marker)
# scene.set_pose(tip_node,pose=tooltip_to_world)
# color,depth = r.render(scene)
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(color)
# plt.subplot(1,2,2)
# plt.imshow(depth)
# plt.show()
### step 2. calculate "tooltip_to_camera" pose by using
###         "tooltip_to_world" pose and "world_to_camera" pose

tooltip_to_camera = np.dot(world_to_camera, tooltip_to_world)

scene.set_pose(tip_node,pose=tooltip_to_camera)

# render
color,depth = r.render(scene)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(color)
plt.subplot(1,2,2)
plt.imshow(depth)
plt.show()
# load image
img = plt.imread("dataset_for_task3/rgb_full/{}.png".format(scenario))[:,:,:3]


##############################################################################
### To do 4 : augment the rendered obj on the image ** WITHOUT FOR LOOP ** ###
##############################################################################

mask = np.logical_not(depth).astype(float)
color = color/255
img_aug = img * mask.reshape(480,640,-1)
img_aug = img_aug + color.reshape(480,640,-1)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img_aug)
plt.show()


#  Once everything is setup, you will see tooltip and lumbar are augmented properly
#  Now we can check out how this augmentation will help in opaque phantom.

if False:

    for scenario in range(4):

        #################################################################
        ###  To do 5: copy paste codes from To do2 to To do4 here     ###
        ###           to look at 4 results of training scenarios.      ###
        #################################################################

        ######################################################################
        ###          To do 2: Calculate "world_to_camera" pose             ###
        ###    from "marker_to_world" pose and "camera_to_marker" pose     ###
        ######################################################################

        # marker to world pose
        cam_marker_to_world = np.loadtxt("dataset_for_task3/pose_cam/{}.txt".format(scenario), delimiter=" ")
        # cam_to_marker_pose
        cam_to_marker = np.loadtxt("dataset_for_task3/pose_head/{}.txt".format(scenario), delimiter=" ")

        ###  step 1. calculate "camera_to_world" pose

        camera_to_world = np.dot(cam_marker_to_world, cam_to_marker)

        ###  step 2. calculate "world_to_camera" pose
        ###          Hint : inverting direction is inversing the matrix

        world_to_camera = np.linalg.inv(camera_to_world)

        ######################################################################
        ###             To do 3: By using "world_to_camera" pose,          ###
        ###   calculate "lumbar_to_camrea" pose and "tip_to_camera" pose   ###
        ######################################################################

        # marker to world pose
        lumbar_marker_to_world = np.loadtxt("dataset_for_task3/pose_phantom/{}.txt".format(scenario), delimiter=" ")
        # lumbar_to_marker_pose
        lumbar_to_marker = np.loadtxt("dataset_for_task3/pose_lumbar/{}.txt".format(scenario), delimiter=" ")

        ### step 1. calculate "lumbar_to_world" pose

        lumbar_to_world = np.dot(lumbar_marker_to_world, lumbar_to_marker)

        ### step 2. calculate "lumbar_to_camera" pose by using
        ###         "lumbar_to_world" pose and "world_to_camera" pose

        lumbar_to_cam = np.dot(world_to_camera, lumbar_to_world)

        scene.set_pose(lumbar_node, pose=lumbar_to_cam)

        # marker to world pose
        tooltip_marker_to_world = np.loadtxt("dataset_for_task3/pose_tool/{}.txt".format(scenario), delimiter=" ")
        # lumbar_to_marker_pose
        tooltip_to_marker = np.loadtxt("dataset_for_task3/pose_tooltip/{}.txt".format(scenario), delimiter=" ")

        ### step 1. calculate "tooltip_to_world" pose

        tooltip_to_world = np.dot(tooltip_marker_to_world, tooltip_to_marker)

        ### step 2. calculate "tooltip_to_camera" pose by using
        ###         "tooltip_to_world" pose and "world_to_camera" pose

        tooltip_to_camera = np.dot(world_to_camera, tooltip_to_world)

        scene.set_pose(tip_node, pose=tooltip_to_camera)

        # render
        color, depth = r.render(scene)

        # load image
        img = plt.imread("dataset_for_task3/rgb_full/{}.png".format(scenario))[:, :, :3]

        ##############################################################################
        ### To do 4 : augment the rendered obj on the image ** WITHOUT FOR LOOP ** ###
        ##############################################################################

        mask = np.logical_not(depth).astype(float)
        color = color / 255
        img_aug = img * mask.reshape(480, 640, -1)
        img_aug = img_aug + color.reshape(480, 640, -1)


        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(img_aug)
        plt.show()
