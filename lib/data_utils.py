import numpy as np
import os


def split_data(X,train_ratio=.75,val_ratio=.15,test_ratio=.1, random=False,random_seed=None):
    assert isinstance(X,np.ndarray)
    nb_train = round(X.shape[0] * train_ratio)
    if test_ratio <= 0. or test_ratio is None:
        nb_test = 0
    else:
        nb_test = round(X.shape[0] * test_ratio)
    if val_ratio <= 0. or val_ratio is None:
        nb_val = 0
    else:
        nb_val = X.shape[0] - nb_train - nb_test
    assert ((nb_val == int(X.shape[0] * val_ratio)) or
            (nb_val == int(X.shape[0] * val_ratio) - 1) or
            (nb_val == int(X.shape[0] * val_ratio) + 1)), \
        'nb_val calculation is off'
    mask = np.arange(X.shape[0])
    if random:
        if random_seed is not None:
            assert isinstance(random_seed,int),'random_seed must be an int'
            np.random.seed(random_seed)
        np.random.shuffle(mask)
    train_mask = mask[0:nb_train]
    val_mask = mask[nb_train:nb_train+nb_val]
    test_mask = mask[nb_train+nb_val:]
    X_train = X[train_mask]
    X_val = X[val_mask] if nb_val else None
    X_test = X[test_mask] if nb_test else None
    return X_train, X_val, X_test


def save_data(data, save_dir, save_name, save_format='npy'):
    if os.path.isfile(save_dir):
        raise FileExistsError('Specified save-dir %s is already an existing file. Please choose a different name for save-dir' % save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print('making dir: %s' % save_dir)

    if save_format == 'npy':
        assert isinstance(data, np.ndarray), 'npy save format requires that the data being saved is a numpy.ndarray, but it is %s' % str(type(data))
        save_path = os.path.join(save_dir,save_name)
        np.save(save_path, data)
    elif save_format == 'nptext':
        raise NotImplementedError('%s save_format has not been implemented yet' % str(save_format))
    else:
        raise RuntimeError('unrecognized save_format: %s' % str(save_format))


def load_data(train_val_test='train', data_dir=None, lim=-1):
    rel_dir = 'data/preprocessed/coco_all'
    if data_dir:
        data_dir = os.path.join(data_dir,rel_dir)
    else:
        data_dir = rel_dir
    if train_val_test=='train':
        X = np.load(os.path.join(data_dir, 'X_train_0.npy'))
        y = np.load(os.path.join(data_dir, 'y_train_0.npy'))
    elif train_val_test=='val':
        X = np.load(os.path.join(data_dir, 'X_val_0.npy'))
        y = np.load(os.path.join(data_dir, 'y_val_0.npy'))
    elif train_val_test=='test':
        X = np.load(os.path.join(data_dir, 'X_test_0.npy'))
        y = np.load(os.path.join(data_dir, 'y_test_0.npy'))
    else:
        raise RuntimeError('unrecognized value for train_val_test: %s' % str(train_val_test))
    if lim > 0:
        X = X[0:lim]
        y = y[0:lim]
    return X,y


def load_data_gtor(train_val_test, data_dir=None, batch_size=32, lim=-1):
    """
    creates a generator that yields a tuple of (X_batch,y_batch) data samples where X_batch and y_batch
    are batch_size slices of data samples. This generator will loop indefinitely over the data,
    which can be training, val, or test data. The data arrays should all be stored in .npy (numpy array)
    format in a single directory.
    Requirements:
    - the path of the directory can optionally be prefixed by 'data_dir' but must end with: 'data/preprocessed/coco_all'
    - data files must follow this naming convention:
        TRAINING data:
            inputs:     'X_train_i' where i = 0,1,2,...
            targets:    'y_train_i' where i = 0,1,2,...
            Note:       'y_train_i' must be the targets corresponding to inputs 'X_train_i' and
                        X_train_i.shape[0] == y_train_i.shape[0] == nb_samples
        VALIDATION data:
            inputs:     'X_val_i' where i = 0,1,2,...
            targets:    'y_val_i' where i = 0,1,2,...
            Note:       'y_val_i' must be the targets corresponding to inputs 'X_val_i' and
                        X_val_i.shape[0] == y_train_i.shape[0] == nb_samples
        TEST data:
            inputs:     'X_test_i' where i = 0,1,2,...
            targets:    'y_test_i' where i = 0,1,2,...
            Note:       'y_test_i' must be the targets corresponding to inputs 'X_test_i' and
                        X_test_i.shape[0] == y_test_i.shape[0] == nb_samples
    :param train_val_test: str
        string indicating whether the generator is for training, validation, or test data
        must be in {'train','val','test'} or else a ValueError is raised
    :param data_dir: str (optional)
        string indicating the prefix of the path for the directory where the data is stored
    :param batch_size: int
        integer indicating the batch_size (i.e. number of data samples to yield each time the generator is called)
    :param lim: int
        integer indicating the amount of data within each data file's array to use
    :return:
    """
    rel_dir = 'data/preprocessed/coco_all'
    if data_dir:
        data_dir = os.path.join(data_dir, rel_dir)
    else:
        data_dir = rel_dir

    if train_val_test not in {'train', 'val', 'test'}:
        raise ValueError('value of train_val_test is not recognized: %s. Should be train, val, or test.' % str(train_val_test))

    # outermost while loop loops indefinitely since our training/val
    # data generator needs to be able to loop continuously over the data:
    while True:
        X_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if
                         os.path.isfile(os.path.join(data_dir, f)) and 'X_%s' % train_val_test in f]
        y_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if
                         os.path.isfile(os.path.join(data_dir, f)) and 'y_%s' % train_val_test in f]

        X_files.sort(reverse=True)
        y_files.sort(reverse=True)
        assert len(X_files) == len(y_files)
        assert len(X_files) > 0  # we already know lengths of the lists are equal from previous two lines

        # set up 'leftover' arrays to handle the case when the number of training sample in
        # the array is not an even multiple of the batch_size (which will happen very frequently)
        # and initialize them to None
        leftover = 0
        X_leftover = None
        y_leftover = None

        # this intermediate while loop loops as long as our list of data files isn't empty:
        while len(X_files) > 0:
            X_f = X_files.pop()
            y_f = y_files.pop()

            try:
                # print('loading file: %s' % str(X_f))
                X = np.load(X_f)
                y = np.load(y_f)
                assert (X.shape[0] == y.shape[0])
                if lim > 0:
                    X = X[0:lim]
                    y = y[0:lim]
            except (IOError, ValueError) as e:
                print('Error while trying to load file to numpy array...')
                repr(e)
                print('Skipping this set of training/val data.')
                continue

            if leftover > 0:
                X = np.vstack((X_leftover,X))
                y = np.vstack((y_leftover,y))
                leftover = 0

            # initialize the sample_num to zero
            sample_num = 0

            # this innermost while loop loops as long as there are batch_size
            # data samples left in the current data array:
            while (sample_num + batch_size) <= X.shape[0]:
                yield (X[sample_num:sample_num + batch_size], y[sample_num:sample_num + batch_size])
                sample_num += batch_size

            leftover = X.shape[0] % batch_size
            if leftover > 0:
                X_leftover = X[-leftover:]
                y_leftover = y[-leftover:]

def combine_data(fs):
    all = list()
    for fname in fs:
        one = np.load(fname)
        all.append(one)
    combined = np.vstack(all)
    # print(combined.shape)
    # input('combined.shape')
    return combined