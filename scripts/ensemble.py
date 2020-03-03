import numpy as np
import os
import sys


if __name__ == '__main__':
    result_dir = sys.argv[1].split(',')
    output_dir = sys.argv[2]

    for event in ['P001', 'P002', 'P003']:
        for idx, result in enumerate(result_dir):
            feat_type = result.split('_')[0]
            if idx == 0:
                conf = np.loadtxt(os.path.join(result, '{}_{}.lst'.format(event, feat_type)))
            else:
                conf += np.loadtxt(os.path.join(result, '{}_{}.lst'.format(event, feat_type)))
        output_type = output_dir.split('_')[0]
        np.savetxt(os.path.join(output_dir, '{}_{}.lst'.format(event, output_type)), conf / len(result_dir))    
