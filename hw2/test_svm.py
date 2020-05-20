#!/bin/python 

import numpy as np
import os
from sklearn.svm import SVC
import pickle
import sys
from sklearn.kernel_approximation import AdditiveChi2Sampler

# Apply the SVM model to the testing videos; Output the score for each video


def read_feature(feature_type, data_type):
    dim = 200 if feature_type == 'surf' else 100
    data_list = [l.strip() for l in open('list/{}.video'.format(data_type), 'r').readlines()]
    x = np.zeros((len(data_list), dim))

    for idx, d in enumerate(data_list):
        x[idx] = np.loadtxt('kmeans/{}/{}'.format(feature_type, d), dtype='float', delimiter=',')

    return x    


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0]))
        print("model_file -- path of the trained svm file")
        print("feat_dir -- dir of feature files")
        print("feat_dim -- dim of features; provided just for debugging")
        print("output_file -- path to save the prediction score")
        exit(1)

    model_file = sys.argv[1]
    feature_type = sys.argv[2]
    data_type = sys.argv[3]
    output_file = sys.argv[4]

    test_X = read_feature(feature_type, data_type)
    chi_feature = AdditiveChi2Sampler(sample_steps=2)
    test_X = chi_feature.fit_transform(test_X)

    clf = pickle.load(open(model_file, 'rb'))

    test_conf = clf.decision_function(test_X)

    output_dir = output_file.split('/')[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    np.savetxt(output_file , test_conf, fmt='%2.4f')

    print('complete prediction!')
