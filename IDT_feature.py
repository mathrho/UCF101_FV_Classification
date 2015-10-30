#
# Handles a improved trajectory feature point
# Check http://lear.inrialpes.fr/people/wang/improved_trajectories
#

import numpy as np
import sys, os, random

INFO_DIM=10
TRAJ_DIM=30
TRAJ_SHAPE_DIM=30
HOG_DIM=96
HOF_DIM=108
MBHX_DIM=96
MBHY_DIM=96
MBH_DIM=96*2
ALL_DIM=INFO_DIM+TRAJ_DIM+TRAJ_SHAPE_DIM+HOG_DIM+HOF_DIM+MBHX_DIM+MBHY_DIM


# Represents a single IDTF feature
class IDTFeature(object):
    def __init__(self, data):

        if data.size:

            traj_start = 10
            traj_shape_start = traj_start + TRAJ_DIM
            hog_start = traj_shape_start + TRAJ_SHAPE_DIM
            hof_start = hog_start + HOG_DIM
            mbhx_start = hof_start + HOF_DIM
            mbhy_start = mbhx_start + MBHX_DIM
            mbhy_end = mbhy_start + MBHY_DIM

            self.info = data[0:9]
            self.traj = data[traj_start:traj_shape_start]
            self.traj_shape = data[traj_shape_start:hog_start]
            self.hog = data[hog_start:hof_start]
            self.hof = data[hof_start:mbhx_start]
            #self.mbhx = data[mbhx_start:mbhy_start]
            #self.mbhy = data[mbhy_start:mbhy_end]
            self.mbh = data[mbhx_start:mbhy_end]

        else:

            self.info = np.array([])
            self.traj = np.array([])
            self.traj_shape = np.array([])
            self.hog = np.array([])
            self.hof = np.array([])
            #self.mbhx = np.array([])
            #self.mbhy = np.array([])
            self.mbh = np.array([])


# Populates an np.ndarray for each of the descriptors in an IDTF feature
# traj, hog, hof, mbh
class vid_descriptors(object):
    # input is a list of IDTFs objects (as specified by the above IDTFeature class) which represent the features of a video.
    def __init__(self, IDTFeatures):

        trajs = []
        hogs = []
        hofs = []
        mbhs = []
        for feature in IDTFeatures:
            trajs.append(np.ndarray(shape=(1,TRAJ_DIM), buffer=feature.traj, dtype=np.float32))
            hogs.append(np.ndarray(shape=(1,HOG_DIM), buffer=feature.hog, dtype=np.float32))
            hofs.append(np.ndarray(shape=(1,HOF_DIM), buffer=feature.hof, dtype=np.float32))
            mbhs.append(np.ndarray(shape=(1,MBH_DIM), buffer=feature.mbh, dtype=np.float32))

        self.traj = np.vstack(trajs)
        self.hog = np.vstack(hogs)
        self.hof = np.vstack(hofs)
        self.mbh = np.vstack(mbhs)


################################################################
# Useful Helper functions                                      #
################################################################


# Returns a tuple (trajs, hogs, hofs, mbhs) where each element is a list of of np.ndarray of type of descriptor.
# Each np.ndarray in the list is a matrix concatenating together all of the descriptors of that particular type for a
# given video.
# So the length of each list will be the number of videos, names provided in the vid_features input list
#
# directory: Directory where the input videos are located
# vid_features: a list of names of videos.
def populate_descriptors(directory, vid_features):
    vid_trajs = []
    vid_hogs = []
    vid_hofs = []
    vid_mbhs = []
    for vid_feature in vid_features:
        vid_desc = vid_descriptors(read_IDTF_file(os.path.join(directory,vid_feature)))
        vid_trajs.append(vid_desc.traj)
        vid_hogs.append(vid_desc.hog)
        vid_hofs.append(vid_desc.hof)
        vid_mbhs.append(vid_desc.mbh)

    return (vid_trajs, vid_hogs, vid_hofs, vid_mbhs)


# returns a list of vid_descriptors objects.
def list_descriptors(directory, vid_features):
    vid_descs = []
    for vid_feature in vid_features:
      vid_descs.append(vid_descriptors(read_IDTF_file(os.path.join(directory,vid_feature))))
    return vid_descs


# returns a list of vid_descriptors objects, sample #nr_sampels_pvid IDTFeatures per video
def list_descriptors_sampled(directory, vid_features, nr_sampels_pvid):
    vid_descs = []
    for vid_feature in vid_features:
        print vid_feature
        points = read_IDTF_file(os.path.join(directory,vid_feature))
        if points:
            nr_points = len(points)
            sample_size = min(nr_points,nr_sampels_pvid)
            idx_sampled = random.sample(xrange(nr_points),sample_size)
            idx_sampled.sort()
            points_sampled = [points[i] for i in idx_sampled]
            vid_descs.append(vid_descriptors(points_sampled))
    return vid_descs


# Provided a list of vid_descriptors objects, concatenates the
# each np.ndarray descriptor type (e.g. each trajs, hogs, hofs, ...)
# into a large np.ndarray matrix.
# Returns a list of 4 large np.ndarray matrices.
def bm_descriptors(descriptors_list):
    vid_trajs = []
    vid_hogs = []
    vid_hofs = []
    vid_mbhs = []
    for desc in descriptors_list:
        vid_trajs.append(desc.traj)
        vid_hogs.append(desc.hog)
        vid_hofs.append(desc.hof)
        vid_mbhs.append(desc.mbh)
    # make each of the descriptor lists into a big matrix.
    bm_list = []
    # The indices of the elements in the list are as follows:
    # bm_list[0] >>> trajs
    # bm_list[1] >>> hogs
    # bm_list[2] >>> hofs
    # bm_list[3] >>> mbhs
    bm_list.append(np.vstack(vid_trajs))
    bm_list.append(np.vstack(vid_hogs))
    bm_list.append(np.vstack(vid_hofs))
    bm_list.append(np.vstack(vid_mbhs))
    return bm_list


# Parses a video's IDTF (binary) file and returns IDTF feature
def read_IDTF_file(vid_feature):

    points = []
    data = np.fromfile(vid_feature, dtype=np.float32)

    if data.size:
        data = np.reshape(data, (-1,ALL_DIM))
        for p in range(0,data.shape[0]):
            points.append(IDTFeature(data[p,:]))

    return points

