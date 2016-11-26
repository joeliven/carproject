import sys, os

proj_roots = [
    '/Users/joeliven/repos/carproject',
    '/scratch/cluster/joeliven/carproject',
    '/u/joeliven/repos/carproject',
    ]

for proj_root in proj_roots:
    if proj_root not in sys.path:
        if os.path.exists(proj_root):
            sys.path.append(proj_root)

import numpy as np
from lib.data_utils import save_data, split_data, combine_data

def process_labels_file(raw_dir):
    i = 0
    with open(raw_dir, 'r') as raw:
        laps = {}
        line = skip(raw)
        while ('EOF' not in line):
            lap_id = 'ERROR'
            if line.startswith('lap_id'):
                lap_id = line.split(':')[1].strip()
                line = skip(raw)
            rng_l = []
            while not line.startswith('end_lap'):
                startend_lab = line.split(':')
                start_end = startend_lab[0].strip()
                lab = startend_lab[1].strip()
                start_end = [int(x.strip()) for x in start_end.split('-')]
                start = start_end[0]
                end = (start_end[1] + 1) if len(start_end) > 1 else (start + 1)
                d = {'start':start_end[0], 'end':end, 'label':lab}
                rng_l.append(d)
                line = skip(raw)

            laps[lap_id] = rng_l
            line = skip(raw)
    return laps

def skip(raw):
    line = raw.readline()
    while line in {'', ' ', '\t', '\n'} or line.startswith('#'):
        line = raw.readline()
    return line

label_map = {
    'S': [1,0,0],
    'T': [0,1,0],
    'Bad': [0,0,1],
}
idx2label = ['S','T','Bad']

# def process_labels(raw_dir, save_dir, proc_shape, target_H, train_ratio, val_ratio, test_ratio):
def process_labels(raw_dir):
    laps_d = process_labels_file(raw_dir)
    y_train = list()
    srtd = sorted(laps_d.keys())
    print('srtd')
    print(srtd)
    for lap_id in sorted(laps_d.keys()):
        l = laps_d[lap_id]
        y_train_i = np.zeros(shape=(l[-1].get('end'),3), dtype=np.int)
        for d in l:
            start = d['start']
            end = d['end']
            label = label_map.get((d['label']))
            for j in range(start,end):
                y_train_i[j] = np.asarray(label)
        print(y_train_i)
        print(y_train_i.shape)
        y_train.append(y_train_i)
    y_train = np.vstack(y_train)
    return y_train

def main():

    if len(sys.argv) < 3:
      print ("label_utils.py: <raw_directory> <save_directory>")
      sys.exit()

    raw_dir = sys.argv[1] #'data/labels'
    save_dir = sys.argv[2] #'data/labels'
    y_all = process_labels(raw_dir)

    print(y_all.shape)
    print(y_all)
    print('y_all')

    # y_train, y_val, y_test = split_data(y_all)
    # print(y_train.shape)
    # print('y_train')
    # print(y_val.shape)
    # print('y_val')
    # print(y_test.shape)
    # print('y_test')

    #save_dir = 'data/preprocessed/gdc_3s'
    save_name = 'y'
    save_data(y_all, save_dir, '%s_train' % save_name, save_format='npy')

    # fs_train = [
    #  'data/preprocessed/gdc_3s/X_train_ccw.npy',
    #     'data/preprocessed/gdc_3s/X_train_cw.npy',
    # ]
    #
    # X_train = combine_data(fs_train)
    # print(X_train.shape)
    # print('X_train.shape')
    #
    # save_dir = 'data/preprocessed/gdc_3s'
    # save_name = 'X'
    # save_data(X_train, save_dir, '%s_train' % save_name, save_format='npy')

if __name__ == "__main__":
  main()
