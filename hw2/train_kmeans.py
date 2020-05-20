import numpy as np
import os
from sklearn.cluster.k_means_ import KMeans
import sys
import pickle
import time

if __name__ == '__main__':
    feat_param = sys.argv[1]
    feature_type = 'surf' if feat_param  == 'surf' else 'cnn'
    dim = 64 if feat_param == 'surf' else (512*7*7 if feat_param == 'cnn2' else 512)
    file_type = 'pkl' if feature_type == 'surf' else 'npy'
    jump = 20 if feature_type == 'surf' else 1
    train_list = [l.strip() for l in open('list/train.video', 'r').readlines()]
    cnt = 0
    for t in train_list:
        if cnt % 100 == 0:
            print(cnt)
        file_path = '{}/{}.{}'.format(feat_param, t, file_type)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            if feature_type =='surf':
                s = pickle.load(open(file_path, 'rb'))
            else:
                s = np.load(file_path)
            if len(s) > 0:
                if feature_type == 'surf':
                    tmp = [s[i * jump].reshape(-1, dim) for i in range(len(s) // jump) if s[i * jump][0] is not None]
                    tmp = np.concatenate(tmp, axis=0)
                else:
                    idx = np.arange((len(s) - 1) // jump + 1) * jump
                    tmp = s[idx, :]
                try:
                    sampled_arr = np.concatenate((sampled_arr, tmp), axis=0)
                except:
                    sampled_arr = tmp
        cnt += 1

    print('training kmeans model with sampled training data')
    print(sampled_arr.shape)
    start_time = time.time()
    kmeans = KMeans(n_clusters=100, random_state=0).fit(sampled_arr[[i * jump for i in range(sampled_arr.shape[0] // jump)]])
    print('complete training kmeans model in {}s'.format(time.time() - start_time))
    print('unload kmeans model')
    pickle.dump(kmeans, open('kmeans_{}'.format(feat_param), 'wb'))


