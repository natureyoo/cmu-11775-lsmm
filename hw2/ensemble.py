import numpy as np
import os
import sys


if __name__ == '__main__':
    feat_list = sys.argv[1].split(',')
    result_dir = sys.argv[2]

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    for event in ['P001', 'P002', 'P003']:
        for idx, feat_type in enumerate(feat_list):
            if idx == 0:
                val_conf = np.loadtxt(os.path.join(result_dir, '{}_{}_val.lst'.format(event, feat_type)))
                test_conf = np.loadtxt(os.path.join(result_dir, '{}_{}.lst'.format(event, feat_type)))
            else:
                val_conf += np.loadtxt(os.path.join(result_dir, '{}_{}_val.lst'.format(event, feat_type)))
                test_conf += np.loadtxt(os.path.join(result_dir, '{}_{}.lst'.format(event, feat_type)))
        
        np.savetxt(os.path.join(result_dir, '{}_best_val.lst'.format(event)), val_conf / len(feat_list))
        np.savetxt(os.path.join(result_dir, '{}_best.lst'.format(event)), test_conf / len(feat_list))

