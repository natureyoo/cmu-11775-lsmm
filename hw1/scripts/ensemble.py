import numpy as np
import os
import sys


if __name__ == '__main__':
    result_dir = sys.argv[1].split(',')
    output_dir = sys.argv[2]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for event in ['P001', 'P002', 'P003']:
        for idx, result in enumerate(result_dir):
            feat_type = result.split('_')[0]
            if idx == 0:
                val_conf = np.loadtxt(os.path.join(result, '{}_{}_val.lst'.format(event, feat_type)))
                test_conf = np.loadtxt(os.path.join(result, '{}_{}.lst'.format(event, feat_type.upper())))
            else:
                val_conf += np.loadtxt(os.path.join(result, '{}_{}_val.lst'.format(event, feat_type)))
                test_conf += np.loadtxt(os.path.join(result, '{}_{}.lst'.format(event, feat_type.upper())))
        
        np.savetxt(os.path.join(output_dir, '{}_best_val.lst'.format(event)), val_conf / len(result_dir))
        np.savetxt(os.path.join(output_dir, '{}_best.lst'.format(event)), test_conf / len(result_dir))
