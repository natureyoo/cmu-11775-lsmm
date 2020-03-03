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
    if len(sys.argv) != 4:
        print('Usage: {0} event_name feat_dir feat_dim output_file'.format(sys.argv[0]))
        print("event_name -- name of the event (P001, P002 or P003 in Homework 1)")
        print("feat_type -- mfcc or asr or both")
        print("output_file -- path to save the svm model")
        exit(1)

    event_name = sys.argv[1]
    feat_type = sys.argv[2]
    output_file = sys.argv[3]

    if feat_type == 'mfcc':
        X = numpy.loadtxt(os.path.join('kmeans', 'result_100.csv'), delimiter=',')
    elif feat_type == 'asr':
        X = numpy.loadtxt(os.path.join('asrfeat', 'result.csv'), delimiter=',')
    else:
        X1 = numpy.loadtxt(os.path.join('kmeans', 'result.csv'), delimiter=',')
        X2 = numpy.loadtxt(os.path.join('asrfeat', 'result.csv'), delimiter=',')
        X = numpy.concatenate((X1, X2), axis=1)
        del X1, X2
    train_y = read_label('../all_trn.lst', event_name)
    train_X = X[:len(train_y)]
    val_y = read_label('../all_val.lst', event_name)
    val_X = X[len(train_y):len(train_y) + len(val_y)]

    ros = RandomOverSampler(random_state=42)
    train_X_res, train_y_res = ros.fit_resample(train_X, train_y)
    # print('Resampled dataset {}'.format(Counter(train_y_res)))
  
    if feat_type == 'mfcc':    
        chi_feature = AdditiveChi2Sampler(sample_steps=2)
        train_X_res = chi_feature.fit_transform(train_X_res)
        val_X = chi_feature.fit_transform(val_X)

    clf = SVC(probability=True)
    clf.fit(train_X_res, train_y_res)

    val_predict = clf.predict(val_X)
    val_conf = clf.decision_function(val_X)

    numpy.savetxt('{}/{}_{}.lst'.format(output_file.split('/')[0], event_name, feat_type), val_conf, fmt='%2.4f')
    # val_accuracy = float((val_predict == val_y).sum()) / len(val_y)
    # val_precision = float(((val_predict == 1) & (val_y == 1)).sum()) / (val_predict == 1).sum()
    # val_recall = float(((val_predict == 1) & (val_y == 1)).sum()) / (val_y == 1).sum()
    # print('Accuracy: {:.2f} Precision: {:.2f} Recall: {:.2f}'.format(val_accuracy, val_precision, val_recall))
    # print('MAP: {:.2f}'.format(average_precision_score(val_y, val_prob)))

    pickle.dump(clf, open(output_file, 'wb'))
    print('SVM trained successfully for event {}!'.format(event_name))
