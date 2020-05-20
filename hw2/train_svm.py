import numpy as np
import os
from sklearn.svm import SVC
import pickle
import sys
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.metrics import average_precision_score
from sklearn.kernel_approximation import AdditiveChi2Sampler


def read_label(use_data, event_name):
    data_list = []
    for data_type in use_data:
        file_path = '../all_{}.lst'.format('trn' if data_type=='train' else 'val')    
        data_list.extend([line.strip() for line in open(file_path, 'r')])
    data = [d.split() for d in data_list]
    label = np.zeros(len(data))
    for idx, d in enumerate(data):
        label[idx] = 1 if d[1] == event_name else 0
    return label


def read_feature(feature_type, use_data):
    data_list = []    
    dim = 200 if feature_type == 'surf' else 100
    for data_type in use_data:
        data_list.extend([l.strip() for l in open('list/{}.video'.format(data_type), 'r').readlines()])
    x = np.zeros((len(data_list), dim))

    for idx, d in enumerate(data_list):
        x[idx] = np.loadtxt('kmeans/{}/{}'.format(feature_type, d), dtype='float', delimiter=',')

    return x    


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: {0} event_name feat_dir output_file'.format(sys.argv[0]))
        print("event_name -- name of the event (P001, P002 or P003 in Homework 1)")
        print("feat_type -- surf or cnn")
        print("output_path -- path to save the svm model")
        print("val_use -- use validation data")
        exit(1)

    event_name = sys.argv[1]
    feat_type = sys.argv[2]
    output_path = sys.argv[3]
    use_data = ['train', 'val'] if sys.argv[4] == 'true' else ['train']

    train_X = read_feature(feat_type, use_data)
    train_y = read_label(use_data, event_name)
    ros = RandomOverSampler(random_state=42)
    train_X_res, train_y_res = ros.fit_resample(train_X, train_y)
    chi_feature = AdditiveChi2Sampler(sample_steps=2)
    train_X_res = chi_feature.fit_transform(train_X_res)

    clf = SVC(probability=True)
    clf.fit(train_X_res, train_y_res)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = '{}/{}_{}_{}'.format(output_path, feat_type, event_name, 'with_val' if len(use_data) > 1 else 'only_train')
    pickle.dump(clf, open(output_file, 'wb'))
    print('SVM trained successfully for event {}!'.format(event_name))
