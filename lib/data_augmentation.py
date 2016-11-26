"""
Script for finalizing the data preprocessing. Does several things:
    (1) removes'Bad' labelled images from both the label set (y) and the data set (X)
    (2) adjusts the label set from 3 class one-hot labels: [1,0,0] representing ['Straight', 'Turn', 'Bad']
        to 2 class one-hot labels [1,0] representing ['Straight', 'Turn']
        Note: we could switch things over to binary cross-entropy labels/loss/training, but I'm keeping it
        as categorical cross-entropy training with a class size of 2 for now bc less code changes required
        and also in case we decide we want to add additional labels in the future
    (3) counts the number of training samples for each label and duplicates the under-represented class
        to ensure that the data set is approximately balances
    (4) shuffles the data set randomly to reduce correlation during training and help ensure coherent gradient updates
    (5) splits the data into train, val, and test sets
    (6) saves the newly formed X_train, y_train, X_val, y_val, X_test, y_test data sets as .npy files in specified dir
"""
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


label_map = {
    'S': [1,0,0],
    'T': [0,1,0],
    'Bad': [0,0,1],
}
idx2label = ['S','T','Bad']

def remove_bads(X_orig, y_orig):
    X = list()
    y = list()
    T_ct = 0
    for i,onehot in enumerate(y_orig):
        label = idx2label[np.argmax(onehot)]
        # print('i:%d\tlabel:%s' % (i,label))
        if 'Bad' not in label:
            # keep the sample and label:
            X.append(X_orig[i])
            y.append(onehot[0:2])
        # else: # label is 'Bad' so remove (don't add to new list) both sample and label:
        if 'T' in label:
            if T_ct % 6 == 0:
                # duplicate the underrepresented sample and label to help balance the dataset:
                X.append(X_orig[i])
                y.append(onehot[0:2])
            T_ct += 1
    X = np.asarray(X)
    y = np.asarray(y)
    print('X.shape')
    print(X.shape)
    print('y.shape')
    print(y.shape)
    assert X.shape[0] == y.shape[0]
    return X,y

def check_balance(X, y):
    S_T_ct = np.sum(y, axis=0)
    print('S_T_ct')
    print(S_T_ct)
    # input('...assess...')

def shuffle(X, y, seed=13):
    # print('not shuffled:')
    # for i,onehot in enumerate(y):
    #     label = idx2label[np.argmax(onehot)]
    #     print('i:%d\tlabel:%s' % (i,label))

    # seed random shuffler so that it's the same for both X and y
    np.random.seed(seed)
    # shuffle the X samples (shuffling done in place, so no need to return)
    np.random.shuffle(X)
    # seed random shuffler so that it's the same for both X and y
    np.random.seed(seed)
    # shuffle the y labels (shuffling done in place, so no need to return)
    np.random.shuffle(y)

    # input('now shuffled:')
    # for i,onehot in enumerate(y):
    #     label = idx2label[np.argmax(onehot)]
    #     print('i:%d\tlabel:%s' % (i,label))



def main():
    # process commandline args:
    if len(sys.argv) < 3:
      print ("data_augmentation.py: <X_path> <y_path> <save_dir>")
      sys.exit()
    X_path = sys.argv[1]
    y_path = sys.argv[2]
    save_dir = sys.argv[3]
    print('X_path')
    print(X_path)
    print('y_path')
    print(y_path)
    print('save_dir')
    print(save_dir)

    X_orig = np.load(X_path)
    y_orig = np.load(y_path)
    print('X_orig.shape')
    print(X_orig.shape)
    print('y_orig.shape')
    print(y_orig.shape)

    # (1) remove all samples from X whose corresponding label is 'Bad'
    # (2) adjust labels from 3 classes down to 2
    # (3) counts the number of training samples for each label and duplicates the under-represented class
    # to ensure that the data set is approximately balances
    X_bal, y_bal = remove_bads(X_orig,y_orig)
    check_balance(X_bal, y_bal)
    # (4) shuffles the data set randomly to reduce correlation during training
    # and help ensure coherent gradient updates
    shuffle(X_bal, y_bal) # shuffling done in place so nothing gets returned

    # (5) splits the data into train, val, and test sets
    X_train, X_val, X_test = split_data(X_bal, train_ratio=.85, val_ratio=.1, test_ratio=.05, random=False, random_seed=None)
    y_train, y_val, y_test = split_data(y_bal, train_ratio=.85, val_ratio=.1, test_ratio=.05, random=False, random_seed=None)
    print('X_train.shape')
    print(X_train.shape)
    print('y_train.shape')
    print(y_train.shape)
    assert X_train.shape[0] == y_train.shape[0]

    print('X_val.shape')
    print(X_val.shape)
    print('y_val.shape')
    print(y_val.shape)
    assert X_val.shape[0] == y_val.shape[0]

    print('X_test.shape')
    print(X_test.shape)
    print('y_test.shape')
    print(y_test.shape)
    assert X_test.shape[0] == y_test.shape[0]

    print('now saving files...')
    save_data(X_train, save_dir, 'X_train', save_format='npy')
    save_data(X_val, save_dir, 'X_val', save_format='npy')
    save_data(X_test, save_dir, 'X_test', save_format='npy')
    save_data(y_train, save_dir, 'y_train', save_format='npy')
    save_data(y_val, save_dir, 'y_val', save_format='npy')
    save_data(y_test, save_dir, 'y_test', save_format='npy')
    print('done saving files.')

if __name__ == "__main__":
  main()
