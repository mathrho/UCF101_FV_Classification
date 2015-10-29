import os, sys, collections
import numpy as np
from yael import ynumpy
import IDT_feature
from tempfile import TemporaryFile

"""
Encodes a fisher vector.

"""


def create_fisher_vector(gmm_list, video_desc, fv_file, fv_sqrt=False, fv_l2=False):
    """
    expects a single video_descriptors object. videos_desciptors objects are defined in IDT_feature.py
    fv_file is the full path to the fisher vector that is created.

    this single video_desc contains the (trajs, hogs, hofs, mbhs) np.ndarrays
    """
    vid_desc_list = []
    vid_desc_list.append(video_desc.traj)
    vid_desc_list.append(video_desc.hog)
    vid_desc_list.append(video_desc.hof)
    vid_desc_list.append(video_desc.mbh)
    # For each video create and normalize a fisher vector for each of the descriptors. Then, concatenate the
    # fisher vectors together to get an extra long fisher vector.
    # Return a list of all of these long fisher vectors. The list should be the same length as the number
    # of input videos.
    fvs = []
    for descriptor,gmm_mean_pca in zip(vid_desc_list,gmm_list):
        gmm, mean, pca_transform = gmm_mean_pca
        # apply the PCA to the vid_trajectory descriptor
        # each image_desc is of size (X,TRAJ_DIM). Pca_tranform is of size (TRAJ_DIM,TRAJ_DIM/2)
        descrip = descriptor.astype('float32') - mean
        if pca_transform != None:
            descrip = np.dot(descrip, pca_transform)
        
        # compute the Fisher vector, using the derivative w.r.t mu and sigma
        fv = ynumpy.fisher(gmm, descrip, include = ['mu', 'sigma'])

        # normalizations are done on each descriptor individually
        if fv_sqrt:
            # power-normalization
            fv = np.sign(fv) * (np.abs(fv) ** 0.5)

        if fv_l2:
            # L2 normalize
            # sum along the rows.
            norms = np.sqrt(np.sum(fv ** 2))
            # -1 allows reshape to infer the length. So it just solidifies the dimensions to (274,1)
            fv /= norms
            # handle images with 0 local descriptor (100 = far away from "normal" images)
            fv[np.isnan(fv)] = 100
        
        fvs.append(fv.T)

    # concatenate fvs
    # output_fv = np.hstack(fvs)

    # L2 normalize the entire fv.
    # norm = np.sqrt(np.sum(output_fv ** 2))
    # output_fv /= norm

    # example name:
    #   'v_Archery_g01_c01.fisher.npz'
    # subdirectory name
    # np.savez(fv_file, fv=output_fv)
    # print fv_file
    # return output_fv

    # fvs[0] >>> traj.fv
    # fvs[1] >>> hog.fv
    # fvs[2] >>> hof.fv
    # fvs[3] >>> mbh.fv
    np.savez(fv_file, fv=fvs)
    print fv_file
    return fvs
