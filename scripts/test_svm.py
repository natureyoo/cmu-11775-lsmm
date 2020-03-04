#!/bin/python 

import numpy
import os
from sklearn.svm import SVC
import pickle
import sys
from sklearn.kernel_approximation import AdditiveChi2Sampler

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
    feat_type = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]

    if feat_type == 'mfcc':    
        X = numpy.loadtxt(os.path.join('kmeans', 'result_{}.csv'.format(feat_dim)), delimiter=',')
    elif feat_type == 'asr':
        X = numpy.loadtxt(os.path.join('asrfeat', 'result_{}.csv'.format(feat_dim)), delimiter=',')
    else:    # soundnet feature
        #  X = numpy.loadtxt(os.path.join('soundnetfeat', 'result_{}.csv'.format(feat_type.split('_')[1])), delimiter=',')
        X = numpy.loadtxt(os.path.join('soundnetfeat', 'result_08.csv'), delimiter=',')
#        for c in ['04', '06', '08']:
#            X_cur = numpy.loadtxt(os.path.join('soundnetfeat', 'result_{}.csv'.format(c)), delimiter=',')
#            X = numpy.concatenate((X, X_cur), axis=1)
#    
    val_file = [line.strip() for line in open('../all_val.lst', 'r')]
    test_file = [line.strip() for line in open('../all_test_fake.lst', 'r')]
    val_X = X[-len(val_file)-len(test_file):-len(test_file)]
    test_X = X[- len(test_file):]

    clf = pickle.load(open(model_file, 'rb'))
    if feat_type == 'mfcc':
        chi_feature = AdditiveChi2Sampler(sample_steps=2)
        val_X = chi_feature.fit_transform(val_X)
        test_X = chi_feature.fit_transform(test_X)
    
    val_conf = clf.decision_function(val_X)
    test_conf = clf.decision_function(test_X)

    output_dir = output_file.split('/')[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    numpy.savetxt(output_file, val_conf, fmt='%2.4f')

    test_file_name = '_'.join(output_file.split('/')[1].split('_')[:2]).upper()

    test_output_file = os.path.join(output_dir, '{}.lst'.format(test_file_name))
    numpy.savetxt(test_output_file , test_conf, fmt='%2.4f')
