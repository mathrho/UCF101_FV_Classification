
"""
Script to train a basic action classification system.

Trains a One vs. Rest SVM classifier on the fisher vector video outputs.
This script is used to experimentally test different parameter settings for the SVMs.

"""

import os, sys, collections, random, string
import numpy as np
import scipy.io
from tempfile import TemporaryFile
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics as metrics
import classify_library


class_index_file = "./class_index.npz"
train_list = '/home/zhenyang/Workspace/data/UCF101/train1.txt'
test_list = '/home/zhenyang/Workspace/data/UCF101/test1.txt'
fv_dir = '/home/zhenyang/Workspace/data/UCF101/features/fv'

#class_index_file_loaded = np.load(class_index_file)
#class_index = class_index_file_loaded['class_index'][()]
#index_class = class_index_file_loaded['index_class'][()]


def make_FV_matrix(videos, fv_dir, labels):

    matrix = []
    target = []
    for i,video in enumerate(videos):
        vid_file = os.path.join(fv_dir,os.path.splitext(video)[0])
        matfile = scipy.io.loadmat(vid_file+'.fv.mat')
        fvlist = matfile['fv']

        if fvlist:
            fvs = []
            for fv in fvlist[0,:]:

                # power-normalization
                fv = np.sign(fv) * (np.abs(fv) ** 0.5)
                # L2 normalize
                norms = np.sqrt(np.sum(fv ** 2))
                fv /= norms
                fv[np.isnan(fv)] = 100

                fvs.append(fv)

            if fvs:
                # concatenate fvs
                output_fv = np.hstack(fvs)

                # L2 normalize the entire fv.
                norm = np.sqrt(np.sum(output_fv ** 2))
                output_fv /= norm

                matrix.append(output_fv)
                target.append(labels[i])

    X = np.vstack(matrix)
    Y = np.array(target)

    return (X,Y)


if __name__ == '__main__':

    f = open(train_list, 'r')
    videos = f.readlines()
    f.close()
    videos_train = [line.split()[0] for line in [video.rstrip() for video in videos]]
    labels_train = [int(line.split()[1]) for line in [video.rstrip() for video in videos]]

    f = open(test_list, 'r')
    videos = f.readlines()
    f.close()
    videos_test = [line.split()[0] for line in [video.rstrip() for video in videos]]
    labels_test = [int(line.split()[1]) for line in [video.rstrip() for video in videos]]

    # GET THE TRAINING AND TESTING DATA.
    X_train, Y_train = make_FV_matrix(videos_train, fv_dir, labels_train)
    X_test, Y_test = make_FV_matrix(videos_test, fv_dir, labels_test)
    flname = '/home/zhenyang/Workspace/data/UCF101/features/UCF101_train1.fv'
    np.savez(flname, data_train=(X_train, Y_train))
    flname = '/home/zhenyang/Workspace/data/UCF101/features/UCF101_test1.fv'
    np.savez(flname, data_test=(X_test, Y_test))

    # TRAINING
    estimator = OneVsRestClassifier(LinearSVC(random_state=0, C=100, loss='l1', penalty='l2'))
    classifier = estimator.fit(X_train, Y_train)
    metrics = classify_library.metric_scores(classifier, X_test, Y_test, verbose=True)
    print metrics

