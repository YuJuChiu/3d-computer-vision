#Only intentend for academic use for the course "Advanced Topics in 3D Computer Vision" at TUM in SS2021
#Do not distribute or share!!!

import os
import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """


    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return semi, desc


class SuperPointFrontend(object):
    """ Wrapper around pytorch net to help with pre and post image processing. """

    def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh,
                 cuda=False):
        self.name = 'SuperPoint'
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh  # L2 descriptor distance for good match.
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.

        # Load the network in inference mode.
        self.net = SuperPointNet()
        if cuda:
            # Train on GPU, deploy on GPU.
            self.net.load_state_dict(torch.load(weights_path))
            self.net = self.net.cuda()
        else:
            # Train on GPU, deploy on CPU.
            self.net.load_state_dict(torch.load(weights_path,
                                                map_location=lambda storage, loc: storage))
        self.net.eval()

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def run(self, img):
        """ Process a numpy image to extract points and descriptors.
        Input
          img - HxW numpy float32 input image in range [0,1].
        Output
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
          """
        assert img.ndim == 2, 'Image must be grayscale.'
        assert img.dtype == np.float32, 'Image must be float32.'
        H, W = img.shape[0], img.shape[1]
        inp = img.copy()
        inp = (inp.reshape(1, H, W))
        inp = torch.from_numpy(inp)
        inp = torch.autograd.Variable(inp).view(1, 1, H, W)
        if self.cuda:
            inp = inp.cuda()
        # Forward pass of network.
        outs = self.net.forward(inp)
        semi, coarse_desc = outs[0], outs[1]
        # Convert pytorch -> numpy.
        semi = semi.data.cpu().numpy().squeeze()
        # --- Process points.
        dense = np.exp(semi)  # Softmax.
        dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # --- Process descriptor.
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            if self.cuda:
                samp_pts = samp_pts.cuda()
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return pts, desc, heatmap


class PointTracker(object):
    """ Class to manage a fixed memory of points and descriptors that enables
    sparse optical flow point tracking.
    Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
    tracks with maximum length L, where each row corresponds to:
    row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
    """

    def __init__(self, max_length, nn_thresh):
        if max_length < 2:
            raise ValueError('max_length must be greater than or equal to 2.')
        self.maxl = max_length
        self.nn_thresh = nn_thresh
        self.all_pts = []
        for n in range(self.maxl):
            self.all_pts.append(np.zeros((2, 0)))
        self.last_desc = None
        self.tracks = np.zeros((0, self.maxl + 2))
        self.track_count = 0
        self.max_score = 9999

    def nn_match_two_way(self, desc1, desc2, nn_thresh):
        """
        Performs two-way nearest neighbor matching of two sets of descriptors, such
        that the NN match from descriptor A->B must equal the NN match from B->A.
        Inputs:
          desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
          desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
          nn_thresh - Optional descriptor distance below which is a good match.
        Returns:
          matches - 3xL numpy array, of L matches, where L <= N and each column i is
                    a match of two descriptors, d_i in image 1 and d_j' in image 2:
                    [d_i index, d_j' index, match_score]^T
        """
        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))
        if nn_thresh < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')
        # Compute L2 distance. Easy since vectors are unit normalized.
        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
        # Get NN indices and scores.
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < nn_thresh
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
        scores = scores[keep]
        # Get the surviving point indices.
        m_idx1 = np.arange(desc1.shape[1])[keep]
        m_idx2 = idx
        # Populate the final 3xN match data structure.
        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores
        return matches

    def get_offsets(self):
        """ Iterate through list of points and accumulate an offset value. Used to
        index the global point IDs into the list of points.
        Returns
          offsets - N length array with integer offset locations.
        """
        # Compute id offsets.
        offsets = []
        offsets.append(0)
        for i in range(len(self.all_pts) - 1):  # Skip last camera size, not needed.
            offsets.append(self.all_pts[i].shape[1])
        offsets = np.array(offsets)
        offsets = np.cumsum(offsets)
        return offsets

    def update(self, pts, desc):
        """ Add a new set of point and descriptor observations to the tracker.
        Inputs
          pts - 3xN numpy array of 2D point observations.
          desc - DxN numpy array of corresponding D dimensional descriptors.
        """
        if pts is None or desc is None:
            print('PointTracker: Warning, no points were added to tracker.')
            return
        assert pts.shape[1] == desc.shape[1]
        # Initialize last_desc.
        if self.last_desc is None:
            self.last_desc = np.zeros((desc.shape[0], 0))
        # Remove oldest points, store its size to update ids later.
        remove_size = self.all_pts[0].shape[1]
        self.all_pts.pop(0)
        self.all_pts.append(pts)
        # Remove oldest point in track.
        self.tracks = np.delete(self.tracks, 2, axis=1)
        # Update track offsets.
        for i in range(2, self.tracks.shape[1]):
            self.tracks[:, i] -= remove_size
        self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
        offsets = self.get_offsets()
        # Add a new -1 column.
        self.tracks = np.hstack((self.tracks, -1 * np.ones((self.tracks.shape[0], 1))))
        # Try to append to existing tracks.
        matched = np.zeros((pts.shape[1])).astype(bool)
        matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
        for match in matches.T:
            # Add a new point to it's matched track.
            id1 = int(match[0]) + offsets[-2]
            id2 = int(match[1]) + offsets[-1]
            found = np.argwhere(self.tracks[:, -2] == id1)
            if found.shape[0] > 0:
                matched[int(match[1])] = True
                row = int(found)
                self.tracks[row, -1] = id2
                if self.tracks[row, 1] == self.max_score:
                    # Initialize track score.
                    self.tracks[row, 1] = match[2]
                else:
                    # Update track score with running average.
                    # NOTE(dd): this running average can contain scores from old matches
                    #           not contained in last max_length track points.
                    track_len = (self.tracks[row, 2:] != -1).sum() - 1.
                    frac = 1. / float(track_len)
                    self.tracks[row, 1] = (1. - frac) * self.tracks[row, 1] + frac * match[2]
        # Add unmatched tracks.
        new_ids = np.arange(pts.shape[1]) + offsets[-1]
        new_ids = new_ids[~matched]
        new_tracks = -1 * np.ones((new_ids.shape[0], self.maxl + 2))
        new_tracks[:, -1] = new_ids
        new_num = new_ids.shape[0]
        new_trackids = self.track_count + np.arange(new_num)
        new_tracks[:, 0] = new_trackids
        new_tracks[:, 1] = self.max_score * np.ones(new_ids.shape[0])
        self.tracks = np.vstack((self.tracks, new_tracks))
        self.track_count += new_num  # Update the track count.
        # Remove empty tracks.
        keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
        self.tracks = self.tracks[keep_rows, :]
        # Store the last descriptors.
        self.last_desc = desc.copy()
        return

    def get_tracks(self, min_length):
        """ Retrieve point tracks of a given minimum length.
        Input
          min_length - integer >= 1 with minimum track length
        Output
          returned_tracks - M x (2+L) sized matrix storing track indices, where
            M is the number of tracks and L is the maximum track length.
        """
        if min_length < 1:
            raise ValueError('\'min_length\' too small.')
        valid = np.ones((self.tracks.shape[0])).astype(bool)
        good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
        # Remove tracks which do not have an observation in most recent frame.
        not_headless = (self.tracks[:, -1] != -1)
        keepers = np.logical_and.reduce((valid, good_len, not_headless))
        returned_tracks = self.tracks[keepers, :].copy()
        return returned_tracks

    def draw_tracks(self, out, tracks):
        """ Visualize tracks all overlayed on a single image.
        Inputs
          out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
          tracks - M x (2+L) sized matrix storing track info.
        """
        # Store the number of points per camera.
        pts_mem = self.all_pts
        N = len(pts_mem)  # Number of cameras/images.
        # Get offset ids needed to reference into pts_mem.
        offsets = self.get_offsets()
        # Width of track and point circles to be drawn.
        stroke = 1
        # Iterate through each track and draw it.
        for track in tracks:
            clr = myjet[int(np.clip(np.floor(track[1] * 10), 0, 9)), :] * 255
            for i in range(N - 1):
                if track[i + 2] == -1 or track[i + 3] == -1:
                    continue
                offset1 = offsets[i]
                offset2 = offsets[i + 1]
                idx1 = int(track[i + 2] - offset1)
                idx2 = int(track[i + 3] - offset2)
                pt1 = pts_mem[i][:2, idx1]
                pt2 = pts_mem[i + 1][:2, idx2]
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
                # Draw end points of each track.
                if i == N - 2:
                    clr2 = (255, 0, 0)
                    cv2.circle(out, p2, stroke, clr2, -1, lineType=16)


class VideoStreamer(object):
    """ Class to help process image streams. Three types of possible inputs:"
      1.) USB Webcam.
      2.) A directory of images (files in directory matching 'img_glob').
      3.) A video file, such as an .mp4 or .avi file.
    """

    def __init__(self, basedir, camid, height, width, skip, img_glob):
        self.cap = []
        self.camera = False
        self.video_file = False
        self.listing = []
        self.sizer = [height, width]
        self.i = 0
        self.skip = skip
        self.maxlen = 1000000
        # If the "basedir" string is the word camera, then use a webcam.
        if basedir == "camera/" or basedir == "camera":
            print('==> Processing Webcam Input.')
            self.cap = cv2.VideoCapture(camid)
            self.listing = range(0, self.maxlen)
            self.camera = True
        else:
            # Try to open as a video.
            self.cap = cv2.VideoCapture(basedir)
            lastbit = basedir[-4:len(basedir)]
            if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
                raise IOError('Cannot open movie file')
            elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
                print('==> Processing Video Input.')
                num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.listing = range(0, num_frames)
                self.listing = self.listing[::self.skip]
                self.camera = True
                self.video_file = True
                self.maxlen = len(self.listing)
            else:
                print('==> Processing Image Directory Input.')
                search = os.path.join(basedir, img_glob)
                self.listing = glob.glob(search)
                self.listing.sort()
                self.listing = self.listing[::self.skip]
                self.maxlen = len(self.listing)
                if self.maxlen == 0:
                    raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')

    def read_image(self, impath, img_size):
        """ Read image as grayscale and resize to img_size.
        Inputs
          impath: Path to input image.
          img_size: (W, H) tuple specifying resize size.
        Returns
          grayim: float32 numpy array sized H x W with values in range [0, 1].
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        # Image is resized via opencv.
        interp = cv2.INTER_AREA
        grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
        grayim = (grayim.astype('float32') / 255.)
        return grayim

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
        Returns
           image: Next H x W image.
           status: True or False depending whether image was loaded.
        """
        if self.i == self.maxlen:
            return (None, False)
        if self.camera:
            ret, input_image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
                return (None, False)
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
            input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                                     interpolation=cv2.INTER_AREA)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            input_image = input_image.astype('float') / 255.0
        else:
            image_file = self.listing[self.i]
            input_image = self.read_image(image_file, self.sizer)
        # Increment internal counter.
        self.i = self.i + 1
        input_image = input_image.astype('float32')
        return (input_image, True)


#HINT: Needed for Pose Graph Optimization
# from posegraphoptimizer import PoseGraphOptimizer, getGraphNodePose

# Util function
def T_from_R_t(R, t):
    R = np.array(R).reshape(3, 3)
    t = np.array(t).reshape(3)
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[0, 3] = t[0]
    T[1, 3] = t[1]
    T[2, 3] = t[2]
    T[3, 3] = 1
    return T


def draw_feature_tracked(second_frame, first_frame,
                        second_keypoints, first_keypoints,
                        color_line=(0, 255, 0), color_circle=(255, 0, 0)):
    mask_bgr = np.zeros_like(cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR))
    frame_bgr = cv2.cvtColor(second_frame, cv2.COLOR_GRAY2BGR)
    for i, (second, first) in enumerate(zip(second_keypoints, first_keypoints)):
        a, b = second.ravel() # flatten
        c, d = first.ravel()
        mask_bgr = cv2.line(mask_bgr, (int(a), int(b)), (int(c), int(d)), color_line, 1)
        frame_bgr = cv2.circle(frame_bgr, (int(a), int(b)), 3, color_circle, 1)
    return cv2.add(frame_bgr, mask_bgr)


def getGT(file_context, frame_id):
    ss = file_context[frame_id].strip().split()
    x = float(ss[3])
    y = float(ss[7])
    z = float(ss[11])
    return [x, y, z]


class Camera:
    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


# Major functions for VO computation
class VO:
    def __init__(self, camera):
        self.camera = camera
        self.focal = self.camera.fx
        self.center = (self.camera.cx, self.camera.cy)

        self.curr_R = None
        self.curr_t = None

        self.T = None
        self.relative_T = None

        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

    def featureTracking(self, curr_frame, old_frame, old_kps):
        # ToDo
        # Not: There is a optical flow method in OpenCV that can help ;) input the old_kps and track them

        # Set parameters for KLT (shape: [k,2] [k,1] [k,1])
        klt_params = dict(winSize=(21, 21),
                          maxLevel=3,
                          criteria=(cv2.TERM_CRITERIA_EPS |
                                    cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        curr_kps, matches, _ = cv2.calcOpticalFlowPyrLK(old_frame,
                                                        curr_frame,
                                                        old_kps,
                                                        None,
                                                        **klt_params)

        ##

        # Remove nono-matched keypoints
        matches = matches.reshape(matches.shape[0])
        return curr_kps[matches == 1], old_kps[matches == 1], matches

    def featureMatching(self, curr_frame, old_frame, weights_path, nms_dist, conf_thresh, nn_thresh, max_length, orb=True, sup=True):
        if orb:
            # ToDo
            # Hint: again, OpenCV is your friend ;) Tip: maybe you want to improve the feature matching by only taking the best matches...
            # print("orb")
            # Initiate ORB detector
            orb = cv2.ORB_create()

            # find the keypoints and descriptors with ORB
            kp1, des1 = orb.detectAndCompute(curr_frame, None)
            kp2, des2 = orb.detectAndCompute(old_frame, None)

            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match descriptors
            matches = bf.match(des1, des2)

            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)

            # # Draw first 10 matches.
            # img3 = cv2.drawMatches(old_frame, kp1, curr_frame, kp2, matches[:10], None, flags=2)
            #
            # plt.imshow(img3), plt.show()

            kp1_match = np.array([kp1[mat.queryIdx].pt for mat in matches])
            kp2_match = np.array([kp2[mat.trainIdx].pt for mat in matches])
            ###

        elif sup:
            # print("sup")
            # path = "/Users/yeelu/Desktop/AT3DCV/AT3DCV_Challenge3/Challenge3Data/superpoint_v1.pth"
            sup = SuperPointFrontend(weights_path, nms_dist, conf_thresh, nn_thresh)
            spt = PointTracker(max_length, nn_thresh)

            curr_frame = np.float32(curr_frame)
            old_frame = np.float32(old_frame)
            # print(curr_frame.shape)
            curr_frame = curr_frame/255
            old_frame = old_frame/255
            # print(old_frame)
            kp1, des1, heatmap1 = sup.run(curr_frame)
            kp2, des2, heatmap2 = sup.run(old_frame)
            # print(kp1.shape)
            matches = spt.nn_match_two_way(des1, des2, 0.7)
            # print(matches)
            m_idx1 = matches[0, :].astype(int).tolist()
            m_idx2 = matches[1, :].astype(int).tolist()

            # print(type(m_idx2), type(kp1), kp1.shape)
            kp1_match = np.array([kp1[:, mat] for mat in m_idx1])
            kp1_match = kp1_match[:, :2]
            kp2_match = np.array([kp2[:, mat] for mat in m_idx2])
            kp2_match = kp2_match[:, :2]
            # print(kp1_match.shape)
        else:  # use SIFT
            # ToDo
            # Hint: Have you heared about the Ratio Test for SIFT?
            # print("sift")
            # find the keypoints and descriptors with SIFT
            sift = cv2.xfeatures2d.SIFT_create()

            kp1, des1 = sift.detectAndCompute(curr_frame, None)
            kp2, des2 = sift.detectAndCompute(old_frame, None)
            # print("type(kp1)", type(kp1), "type(des1)", type(des1))
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            # print("type(matches)", type(matches))
            # Apply ratio test
            good = []
            good_list = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
                    good_list.append(m)

            # # cv2.drawMatchesKnn expects list of lists as matches.
            # img3 = cv2.drawMatchesKnn(old_frame, kp1, curr_frame, kp2, good, None, flags=2)
            #
            # plt.imshow(img3), plt.show()

            kp1_match = np.array([kp1[mat.queryIdx].pt for mat in good_list])
            kp2_match = np.array([kp2[mat.trainIdx].pt for mat in good_list])

        return kp1_match, kp2_match, matches

    def initialize(self, first_frame, sceond_frame, w, nm, cf, nn, ml, of=True, orb=False, sup=True):
        if of:
            first_keypoints = self.detector.detect(first_frame)
            first_keypoints = np.array([x.pt for x in first_keypoints], dtype=np.float32)
            second_keypoints_matched, first_keypoints_matched, _ = self.featureTracking(sceond_frame, first_frame,
                                                                                        first_keypoints)
        else:
            second_keypoints_matched, first_keypoints_matched, _ = self.featureMatching(sceond_frame, first_frame, w, nm, cf, nn, ml,
                                                                                        orb=orb, sup=sup)

        # ToDo
        # Hint: Remember the lecture: given the matched keypoints you can compute the Essential matrix and from E you can recover R and t...

        E, mask = cv2.findEssentialMat(second_keypoints_matched, first_keypoints_matched, focal=self.focal, pp=self.center, method=cv2.RANSAC, prob=0.999, threshold=1)
        _, self.curr_R, self.curr_t, mask = cv2.recoverPose(E, second_keypoints_matched, first_keypoints_matched, focal=self.focal, pp=self.center)

        ###

        self.relative_T = T_from_R_t(self.curr_R, self.curr_t)
        self.T = self.relative_T
        return second_keypoints_matched, first_keypoints_matched

    def processFrame(self, curr_frame, old_frame, old_kps, w, nm, cf, nn, ml, of=True, orb=False, sup=True):

        if of:
            curr_kps_matched, old_kps_matched, matches = self.featureTracking(curr_frame, old_frame,
                                                                                           old_kps)
        else:
            curr_kps_matched, old_kps_matched, matches = self.featureMatching(curr_frame, old_frame, w, nm, cf, nn, ml,
                                                                                           orb=orb, sup=sup)

        # ToDo
        # Hint: Here we only do the naive way and do everything based on Epipolar Geometry (Essential Matrix). No need for PnP in this tutorial

        # Check the size of input images
        assert (curr_frame.ndim == 2 and curr_frame.shape[0] == self.camera.height and curr_frame.shape[1] == self.camera.width)
        assert (old_frame.ndim == 2 and old_frame.shape[0] == self.camera.height and old_frame.shape[1] == self.camera.width)

        # Find Essential Matrix (by RANSAC)
        E, _ = cv2.findEssentialMat(curr_kps_matched, old_kps_matched, focal=self.focal, pp=self.center, method=cv2.RANSAC, prob=0.999, threshold=1)
        # Recover Pose (the translation t is set to 1)
        _, R, t, mask = cv2.recoverPose(E, curr_kps_matched, old_kps_matched, focal=self.focal, pp=self.center)

        ###


        inliners = len(mask[mask == 255])
        if (inliners > 20):
            self.relative_T = T_from_R_t(R, t)
            self.curr_t = self.curr_t + self.curr_R.dot(t)
            self.curr_R = R.dot(self.curr_R)
            self.T = T_from_R_t(self.curr_R, self.curr_t)

        # Get new KPs if too few
        if (old_kps_matched.shape[0] < 1000):
            curr_kps_matched = self.detector.detect(curr_frame)
            curr_kps_matched = np.array([x.pt for x in curr_kps_matched], dtype=np.float32)
        return curr_kps_matched, old_kps_matched



def main():
    argument = argparse.ArgumentParser()
    argument.add_argument("--o", help="use ORB", action="store_true")
    argument.add_argument("--f", help="use Optical Flow", action="store_true")
    argument.add_argument("--l", help="use Loop Closure for PGO", action="store_true")
    argument.add_argument("--s", help="use super_points", action="store_true")
    argument.add_argument('--w', type=str, default='/Users/yeelu/Desktop/AT3DCV/AT3DCV_Challenge3/Challenge3Data/superpoint_v1.pth',
                        help='Path to pretrained weights file (default: superpoint_v1.pth).')
    argument.add_argument('--nm', type=int, default=4,
                        help='Non Maximum Suppression (NMS) distance (default: 4).')
    argument.add_argument('--cf', type=float, default=0.015,
                        help='Detector confidence threshold (default: 0.015).')
    argument.add_argument('--nn', type=float, default=0.7,
                        help='Descriptor matching threshold (default: 0.7).')
    argument.add_argument('--ml', type=int, default=5,
                        help='Maximum length of point tracks (default: 5).')
    args = argument.parse_args()
    orb = args.o
    of = args.f
    sup = args.s
    w = args.w
    nm = args.nm
    cf = args.cf
    nn = args.nn
    ml = args.ml
    loop_closure = args.l

    #Hard-coded Loop closure estimates (Needed for PGO); We only take these 2 for now
    # lc_ids = [1572, 3529]
    # lc_dict = {1572: 125, 3529: 553}

    #HINT: Adapt path
    image_dir = os.path.realpath('./Challenge3Data/data/')
    pose_path = os.path.realpath('./Challenge3Data/00.txt')

    with open(pose_path) as f:
        poses_context = f.readlines()

    image_list = []
    for file in os.listdir(image_dir):
        if file.endswith("png"):
            image_list.append(image_dir + '/' + file)

    image_list.sort()

    # Initial VisualOdometry Object
    camera = Camera(1241.0, 376.0, 718.8560,
                    718.8560, 607.1928, 185.2157)
    vo = VO(camera)
    traj_plot = np.zeros((1000,1000,3), dtype=np.uint8)

    # # ToDo (PGO)
    # #Hint: Initialize Pose Graph Optimizer
    # # Hint: have a look in the PGO class and what methods are provided. The first frame should be static (addPriorFactor)


    ###


    first = 0
    second = first + 3  # For wider baseline with better initialization...
    first_frame = cv2.imread(image_list[first], 0)
    second_frame = cv2.imread(image_list[second], 0)

    second_keypoints, first_keypoints = vo.initialize(first_frame, second_frame, w, nm, cf, nn, ml, of=of, orb=orb, sup=sup)

    # ToDo (PGO)
    # Hint: fill the Pose Graph: There is a difference between the absolute pose and the relative pose



    ###


    old_frame = second_frame
    old_kps = second_keypoints


    for index in range(second+1, len(image_list)):
        curr_frame = cv2.imread(image_list[index], 0)
        # print(image_list[index])
        true_pose = getGT(poses_context, index)
        true_x, true_y = int(true_pose[0])+290, int(true_pose[2])+90

        curr_kps, old_kps = vo.processFrame(curr_frame, old_frame, old_kps, w, nm, cf, nn, ml, of=of, orb=orb, sup=sup)

        # ToDo (PGO)
        # Hint: keep filling new poses



        ###

        if loop_closure:
            if index in lc_ids:
                loop_idx = lc_dict[index]
                print("Loop: ", PGO.curr_node_idx, loop_idx)

                # ToDo (PGO)
                # Hint: just use Identity pose for Loop Closure np.eye(4)



                ###

                # #Plot trajectory after PGO
                # for k in range(index):
                #     try:
                #         pose_trans, pose_rot = getGraphNodePose(PGO.graph_optimized, k)
                #         print(pose_trans)
                #         print(pose_rot)
                #         cv2.circle(traj_plot, (int(pose_trans[0])+290, int(pose_trans[2])+90), 1, (255, 0, 255), 5)
                #     except:
                #         #catch error for first few missing poses...
                #         print("Pose not available for frame # ", k)


        #Utilities for Drawing
        curr_t = vo.curr_t
        if(index > 2):
            x, y, z = curr_t[0], curr_t[1], curr_t[2]
        else:
            x, y, z = 0., 0., 0.
        odom_x, odom_y = int(x)+290, int(z)+90

        cv2.circle(traj_plot, (odom_x,odom_y), 1, (index*255/4540,255-index*255/4540,0), 1)
        cv2.circle(traj_plot, (true_x,true_y), 1, (0,0,255), 2)
        cv2.rectangle(traj_plot, (10, 20), (600, 60), (0,0,0), -1)
        text = "FrameID: %d  Coordinates: x=%1fm y=%1fm z=%1fm"%(index,x,y,z)
        cv2.putText(traj_plot, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
        cv2.imshow('Trajectory', traj_plot)
        show_image = draw_feature_tracked(curr_frame, old_frame,
                                         curr_kps, old_kps)
        cv2.imshow('Mono', show_image)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break



        # Update old data
        old_frame = curr_frame
        old_kps = curr_kps

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
