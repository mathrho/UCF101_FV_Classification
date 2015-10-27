#
# Handles a improved trajectory feature point
# Check http://lear.inrialpes.fr/people/wang/improved_trajectories
#

import numpy as np
import os
INFO_DIM=10
TRAJ_DIM=30
TRAJ_SHAPE_DIM=30
HOG_DIM=96
HOF_DIM=108
MBHX_DIM=96
MBHY_DIM=96
ALL_DIM=INFO_DIM+TRAJ_DIM+TRAJ_SHAPE_DIM+HOG_DIM+HOF_DIM+MBHX_DIM+MBHY_DIM

#Represents a set of IDTF feature (columns as points)
class IDTFeature(object):
    def __init__(self, data):

        if data.size:
        
            data = np.reshape(data, (ALL_DIM, -1), order='F')

            traj_start = 10
            traj_shape_start = traj_start + TRAJ_DIM
            hog_start = traj_shape_start + TRAJ_SHAPE_DIM
            hof_start = hog_start + HOG_DIM
            mbhx_start = hof_start + HOF_DIM
            mbhy_start = mbhx_start + MBHX_DIM
            mbhy_end = mbhy_start + MBHY_DIM

            self.info = data[0:9,:]
            self.traj = data[traj_start:traj_shape_start,:]
            self.traj_shape = data[traj_shape_start:hog_start,:]
            self.hog = data[hog_start:hof_start,:]
            self.hof = data[hof_start:mbhx_start,:]
            self.mbhx = data[mbhx_start:mbhy_start,:]
            self.mbhy = data[mbhy_start:mbhy_end,:]

        else:

            self.info = np.array([])
            self.traj = np.array([])
            self.traj_shape = np.array([])
            self.hog = np.array([])
            self.hof = np.array([])
            self.mbhx = np.array([])
            self.mbhy = np.array([])


# Parses a video's IDTF (binary) file and returns IDTF feature
def read_IDTF_file(vid_feature):

    feature = None
    data = np.fromfile(vid_feature, dtype=np.float32)
    feature = IDTFeature(data)

    return feature

