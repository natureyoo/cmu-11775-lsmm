import numpy as np
import os
from sklearn.cluster.k_means_ import KMeans
import sys
import pickle


def kmeans_feature(file_list, feature_type, kmeans_model, dim=100):
    feat_dim = 64 if feature_type == 'surf' else (512*7*7 if feature_type == 'cnn2' else 512)
    file_type = 'pkl' if feature_type == 'surf' else 'npy'
    base_surf = [np.zeros((1, 1, feat_dim))] if feature_type == 'surf' else np.zeros((1, feat_dim))
    cnt = 0
    for f in file_list:
        f_path = '{}/{}.{}'.format(feature_type, f, file_type)
        if cnt % 10 == 0:
            print(cnt)
            
        if os.path.exists(f_path) and os.path.getsize(f_path) > 0:
            try:
                s = pickle.load(open(f_path, 'rb')) if feature_type == 'surf' else np.load(f_path)
            except:
                print('invalid feature on {}'.format(f))
                s = base_surf
            if len(s) == 0:
                s = base_surf
        else:
            print('invalid feature on {}'.format(f))
            s = base_surf
            
        if feature_type == 'surf':
            video_vec = None
            for i in range(len(s)):
                frame_vec = np.zeros((1, dim))
                if s[i][0] is None:
                    continue
                g, c = np.unique(kmeans_model.predict(s[i].reshape(-1, feat_dim)), return_counts=True)
                frame_vec[0, g] = c / c.sum()
                try:
                    video_vec = np.concatenate((video_vec, frame_vec), axis=0)
                except:
                    video_vec = frame_vec
            video_feature = np.concatenate((np.average(video_vec, axis=0), np.max(video_vec, axis=0)))
        else:
            video_feature = np.zeros((1, dim))
            g, c = np.unique(kmeans_model.predict(s.reshape(-1, feat_dim)), return_counts=True)
            video_feature[0][g] = c / c.sum()
        np.savetxt('kmeans/{}/{}'.format(feature_type, f), video_feature, fmt='%1.8e', delimiter=',')
                
        cnt += 1


def make_features(feature_type='surf'):
    kmeans_model = pickle.load(open('kmeans_{}'.format(feature_type), 'rb'))
    train_list = [l.strip() for l in open('list/train.video', 'r').readlines()]
    val_list = [l.strip() for l in open('list/val.video', 'r').readlines()]
    test_list = [l.strip() for l in open('list/test.video', 'r').readlines()]

    for data_list in [train_list, val_list, test_list]:
        kmeans_feature(data_list, feature_type, kmeans_model)
    
    print('complete create kmeans representation for {}!'.format(feature_type))


if __name__ == '__main__':
    feat_param = sys.argv[1]
    os.makedirs('./kmeans/{}'.format(feat_param), exist_ok=True)
    make_features(feat_param)    

