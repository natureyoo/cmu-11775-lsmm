#!/bin/python 

import numpy
import os
from sklearn.svm import SVC
import pickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0]))
        print("model_file -- path of the trained svm file")
        print("feat_dir -- dir of feature files")
        print("feat_dim -- dim of features; provided just for debugging")
        print("output_file -- path to save the prediction score")
        exit(1)

    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]

    X = numpy.loadtxt(os.path.join(feat_dir, 'result.csv'), delimiter=',')
    test_file = [line.strip() for line in open('../all_test_fake.lst', 'r')]
    test_X = X[- len(test_file):]

    clf = pickle.load(open(model_file, 'rb'))
    test_predict = clf.predict(test_X)

    numpy.savetxt(output_file, test_predict, fmt='%s')
