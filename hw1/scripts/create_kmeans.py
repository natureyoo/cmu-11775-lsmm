#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
# Generate k-means features for videos; each video is represented by a single vector

def read_video_feat(video_id):
    file_path = os.path.join('mfcc', '{}.mfcc.csv'.format(video_id))
    feature = numpy.loadtxt(file_path, delimiter=';')
    return feature

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)

    kmeans_model = sys.argv[1]; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model,"rb"))
    video_list = [line.strip() for line in open(file_list, 'r')]  
    
    video_vec = numpy.zeros((len(video_list), cluster_num))

    for idx, video in enumerate(video_list):
        if idx % 100 == 0:
            print "{}th video processing...".format(idx)
        try:
            feature = read_video_feat(video)
            feature_map = kmeans.predict(feature)
            group, count = numpy.unique(feature_map, return_counts=True)
            for (g, c) in zip(group, count):
                video_vec[idx, g] = float(c) / count.sum()       
        except IOError:
            pass
    numpy.savetxt(os.path.join('kmeans', 'result_{}.csv'.format(cluster_num)), video_vec, delimiter=',')
 
    print "K-means features generated successfully!"
