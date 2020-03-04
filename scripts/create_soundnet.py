#!/bin/python
import numpy as np
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: {0} soundnet_directory, output_directory'.format(sys.argv[0]))
        exit(1)

    soundnet_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    video_list = [line.strip() for line in open('list/all.video')]
    video_idx = {}
    for idx, v in enumerate(video_list):
        video_idx[v] = idx

    conv_dir_list = ['conv_03', 'conv_04', 'conv_06', 'conv_08']
    for conv in conv_dir_list:
        print('Start {}!'.format(conv))
        feat_num = 0
        conv_dim = 0    
        for f in os.listdir(os.path.join(soundnet_dir, conv)):
            video_name = f.split('.')[0]
            conv_feat = np.load(os.path.join(soundnet_dir, conv, f))  

            if conv_feat.shape[0] > 32:
                continue

            if conv_dim == 0:
                conv_dim = conv_feat.shape[0]
                video_feature = np.zeros((len(video_list), conv_dim))

            video_feature[video_idx[video_name]] = conv_feat
            feat_num += 1
        print('Finish processing {} video features in {}'.format(feat_num, conv))

        np.savetxt(os.path.join(output_dir, 'result_{}.csv'.format(conv.split('_')[1])), video_feature, delimiter=',')

    print('SoundNet feature generated successfully!')
                  
