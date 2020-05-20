#!/bin/python 

import numpy
import os
from sklearn.svm import SVC
import pickle
import sys
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.metrics import average_precision_score
from sklearn.kernel_approximation import AdditiveChi2Sampler


# Performs K-means clustering and save the model to a local file

def read_label(file_path, event_name):
    data = [line.strip() for line in open(file_path, 'r')]
    data = [d.split() for d in data]
    label = numpy.zeros(len(data))
    for idx, d in enumerate(data):
        label[idx] = 1 if d[1] == event_name else 0
    return label

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: {0} event_name feat_dir feat_dim output_file'.format(sys.argv[0]))
        print("event_name -- name of the event (P001, P002 or P003 in Homework 1)")
        print("feat_type -- mfcc or asr")
        print("output_file -- path to save the svm model")
        exit(1)

    event_name = sys.argv[1]
    feat_type = sys.argv[2]
    feat_dim = sys.argv[3]
    output_file = sys.argv[4]

    if feat_type == 'mfcc':
        X = numpy.loadtxt(os.path.join('kmeans', 'result_{}.csv'.format(feat_dim)), delimiter=',')
    elif feat_type == 'asr':
        X = numpy.loadtxt(os.path.join('asrfeat', 'result_{}.csv'.format(feat_dim)), delimiter=',')
    else:    # soundnet feature
        # X = numpy.loadtxt(os.path.join('soundnetfeat', 'result_{}.csv'.format(feat_type.split('_')[1])), delimiter=',')
        X = numpy.loadtxt(os.path.join('soundnetfeat', 'result_08.csv'), delimiter=',')
#        for c in ['04', '06', '08']:
#            X_cur = numpy.loadtxt(os.path.join('soundnetfeat', 'result_{}.csv'.format(c)), delimiter=',')
#            X = numpy.concatenate((X, X_cur), axis=1)
#
    train_y = read_label('../all_trn.lst', event_name)
    train_X = X[:len(train_y)]

    ros = RandomOverSampler(random_state=42)
    train_X_res, train_y_res = ros.fit_resample(train_X, train_y)
    # print('Resampled dataset {}'.format(Counter(train_y_res)))
  
    if feat_type == 'mfcc':    
        chi_feature = AdditiveChi2Sampler(sample_steps=2)
        train_X_res = chi_feature.fit_transform(train_X_res)

    clf = SVC(probability=True)
    clf.fit(train_X_res, train_y_res)

    output_dir = output_file.split('/')[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pickle.dump(clf, open(output_file, 'wb'))
    print('SVM trained successfully for event {}!'.format(event_name))
