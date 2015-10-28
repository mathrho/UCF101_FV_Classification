"""
Can execute this as a script to populate the GMM or load it as a module

PCA reduction on each descriptor is set to false by default.
"""
import IDT_feature
import numpy as np
import sys, os
from yael import ynumpy
import argparse


def populate_gmms(VID_DIR, sample_vids, gmm_file, k_gmm, sample_size=1500000, PCA=False):
    """
    sample_size is the number of IDTFs that we sample from the total_lines number of IDTFs
    that were computed previously.

    gmm_file is the output file to save the list of GMMs.
    Saves the GMMs in the gmm_file file as the gmm_list attribute.

    Returns the list of gmms.
    """
    nr_vids = len(sample_vids)
    nr_samples_pvid = np.ceil(sample_size/nr_vids)

    sample_descriptors = IDT_feature.list_descriptors_sampled(VID_DIR, sample_vids, nr_samples_pvid)
    bm_list = IDT_feature.bm_descriptors(sample_descriptors)

    #Construct gmm models for each of the different descriptor types.
    gmm_list = [gmm_model(bm, k_gmm, PCA=PCA) for bm in bm_list]
    np.savez(gmm_file, gmm_list=gmm_list)
    
    return gmm_list


def gmm_model(sample, k_gmm, PCA=False):
    """
    Returns a tuple: (gmm,mean,pca_transform)
    gmm is the ynumpy gmm model fro the sample data. 
    pca_tranform is None if PCA is True.
    Reduces the dimensions of the sample (by 50%) if PCA is true
    """

    print "Building GMM model"
    # convert to float32
    sample = sample.astype('float32')
    # compute mean and covariance matrix for the PCA
    mean = sample.mean(axis = 0) # for rows
    sample = sample - mean
    pca_transform = None
    if PCA:
        cov = np.dot(sample.T, sample)

        # decide to keep 1/2 of the original components, so shape[1]/2
        # compute PCA matrix and keep only 1/2 of the dimensions.
        orig_comps = sample.shape[1]
        pca_dim = orig_comps/2
        # eigvecs are normalized.
        eigvals, eigvecs = np.linalg.eig(cov)
        perm = eigvals.argsort() # sort by increasing eigenvalue 
        pca_transform = eigvecs[:, perm[orig_comps-pca_dim:orig_comps]]   # eigenvectors for the 64 last eigenvalues
        # transform sample with PCA (note that numpy imposes line-vectors,
        # so we right-multiply the vectors)
        sample = np.dot(sample, pca_transform)
    # train GMM
    gmm = ynumpy.gmm_learn(sample, k_gmm)
    toReturn = (gmm,mean,pca_transform)
    return toReturn


def computeIDTFs(training_samples, VID_DIR):
    """
    Computes the IDTFs specifically used for constructing the GMM
    training_samples is a list of videos located at the VID_DIR directory.
    The IDT features are output in the GMM_dir.
    """
    for video in training_samples:
        videoLocation = os.path.join(VID_DIR,video)
        featureOutput = os.path.join(GMM_dir,os.path.basename(video)[:-4]+".features")
        print "Computing IDTF for %s" % (video)
        computeIDTF.extract(videoLocation, featureOutput)
        print "complete."  


def sampleVids(vid_list, nr_pcls=1):
    """
    vid_list is a text file of video names and their corresponding
    class.
    This function reads the video names and creates a list with one video
    from each class.
    """
    f = open(vid_list, 'r')
    videos = f.readlines()
    f.close()
    videos = [video.rstrip() for video in videos]
    vid_dict = {}
    for line in videos:
        l = line.split()
        key = int(l[1])
        if key not in vid_dict:
            vid_dict[key] = []
        vid_dict[key].append(l[0].split('/')[1])
    
    samples = []
    for k,v in vid_dict.iteritems():
        #samples.extend(v[:1])
        samples.extend(v[:min(nr_pcls,len(v))])
    return samples


#python gmm.py 120 UCF101_dir train_list
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--gmmk', help='Number of GMM modes', type=int, required=True)
    parser.add_argument('-v', '--videos', help="Directory of the input videos or features", type=str)
    parser.add_argument('-l', '--vidlist', help="List of input videos from which to sample", type=str)
    parser.add_argument('-o', '--gmmfile', help="Output file to save the list of gmms", type=str)

   # parser.add_argument("-p", "--pca", type=float, help="percent of original descriptor components to retain after PCA")
    parser.add_argument("-p", "--pca", action="store_true",
        help="Reduce each descriptor dimension by 50 percent using PCA")
    args = parser.parse_args()

    print args.gmmk
    print args.videos
    print args.vidlist
    print args.gmmfile

    VID_DIR = args.videos
    input_list = args.input_list

    #vid_samples = sampleVids(input_list)
    vid_samples = input_list

    #computeIDTFs(vid_samples, VID_DIR)
    features = []
    for vidname in vid_samples:
        vidname_ = os.path.splitext(vidname)[0]
        features.append(vidname_+'.bin')
    populate_gmms(VID_DIR,features,args.gmmfile,args.gmmk,PCA=args.pca)

