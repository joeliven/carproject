"""Script for executing varoius preprocessing tasks,
such as preprocessing and saving training, validation, and test data"""
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

import time, math
import numpy as np
import argparse
from skimage.io import imread
from skimage.io import imread_collection
from skimage import transform
import matplotlib.pyplot as plt
from lib.image_utils import preprocess_image, deprocess_image, channel_first2last, channel_last2first, bgr2rgb, rgb2bgr, downsample_image
from lib.data_utils import split_data, save_data


def preprocess_coco(raw_dir, save_dir, proc_shape, target_H, train_ratio, val_ratio, test_ratio, random_split=False, random_seed=13, lim=-1, chunk_size=5000, targets3d=False):
    start_time = time.time()
    raw_files = [os.path.join(raw_dir,f) for f in os.listdir(raw_dir) if (os.path.isfile(os.path.join(raw_dir, f)) and f.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff','.gif','.bmp')))]
    if lim is None:
        lim = -1
    if lim > 0:
        raw_files = raw_files[0:lim]
    # if there are more than 'chunk_size' raw files, we split the data into chunks of chunk_size to reduce the memory footprint
    # and storage space needed to save each training/val/test file on disk
    raw_files_l = list()
    for start in range(0,len(raw_files),chunk_size):
        end = start + chunk_size
        raw_files_l.append(raw_files[start:end])
    for chunk_num,raw_files in enumerate(raw_files_l):
        print('raw_files')
        print(raw_files)
        processed = list()
        targets = list()
        for r,raw_file in enumerate(raw_files):
            if r % 10 == 0:
                print('chunk:%d\tpreprocessing file %d' % (chunk_num, r))
            raw_img = imread(raw_file)
            img = preprocess_image(raw_img, proc_shape=proc_shape)
            if img is None:
                continue
            deproc = deprocess_image(img)
            deproc_ratio = deproc.shape[1] / deproc.shape[0] # ratio of W:H
            if targets3d:
                targ = transform.resize(deproc,output_shape=(3,112,112))
                targets.append(targ)
            else:
                small_shape = (target_H, int(target_H * deproc_ratio))
                small_gray = downsample_image(deproc, new_size=small_shape, grey=True)
                targets.append(small_gray.flatten())
            # print('(caller) small_gray.shape:%s' % str(small_gray.shape))
            # print('(caller) img.shape:%s' % str(img.shape))
            processed.append(img)
        processed = np.asarray(processed)
        targets = np.asarray(targets)
        print('chunk:%d\tprocessed.shape:%s' % (chunk_num,str(processed.shape)))
        print('chunk:%d\ttargets.shape:%s' % (chunk_num,str(targets.shape)))
        X_train, X_val, X_test = split_data(processed, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, random=random_split, random_seed=random_seed)
        y_train, y_val, y_test = split_data(targets, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, random=random_split, random_seed=random_seed)
        print('chunk:%d\tX_train_%d.shape%s' % (chunk_num,chunk_num,str(X_train.shape)))
        print('chunk:%d\ty_train_%d.shape%s' % (chunk_num,chunk_num,str(y_train.shape)))
        if X_val is not None:
            print('chunk:%d\tX_val_%d.shape%s' % (chunk_num,chunk_num,str(X_val.shape)))
        if y_val is not None:
            print('chunk:%d\ty_val_%d.shape%s' % (chunk_num,chunk_num,str(y_val.shape)))
        if X_test is not None:
            print('chunk:%d\tX_test_%d.shape%s' % (chunk_num,chunk_num,str(X_test.shape)))
        if y_test is not None:
            print('chunk:%d\ty_test_%d.shape%s' % (chunk_num,chunk_num,str(y_test.shape)))

        save_data(X_train,save_dir,'X_train_%d' % chunk_num,save_format='npy')
        if X_val is not None:
            save_data(X_val,save_dir,'X_val_%d' % chunk_num,save_format='npy')
        if X_test is not None:
            save_data(X_test,save_dir,'X_test_%d' % chunk_num,save_format='npy')
        save_data(y_train,save_dir,'y_train_%d' % chunk_num,save_format='npy')
        if y_val is not None:
            save_data(y_val,save_dir,'y_val_%d' % chunk_num,save_format='npy')
        if y_test is not None:
            save_data(y_test,save_dir,'y_test_%d' % chunk_num,save_format='npy')
        del raw_files
        del processed
        del targets
    end_time = time.time()
    duration = (end_time - start_time) / 60.
    print('total duration: %f mins' % (duration))

def deprocess_coco(processed_dir, processed_name, targets_name, lim):
    Xf = os.path.join(processed_dir,processed_name)
    X = np.load(Xf)
    yf = os.path.join(processed_dir,targets_name)
    y = np.load(yf)
    print('X.shape:%s' % str(X.shape))
    print('y.shape:%s' % str(y.shape))
    H_W = int(math.sqrt(y.shape[1]))
    y = y.reshape((-1,H_W,H_W)) # note this assumes that targets (small, greyscaled images) are square in shape
    print('y.shape:%s' % str(y.shape))
    for i,proc_img in enumerate(X):
        # print('processed_img.shape:%s' % str(proc_img.shape))
        fig, ax = plt.subplots(1, 3)
        # ax[0].imshow((channel_first2last(bgr2rgb(proc_img))))
        ax[0].imshow((channel_first2last(proc_img)))
        deproc_img = deprocess_image(proc_img)
        # nqe = deproc_img[deproc_img != channel_first2last(bgr2rgb(proc_img))]
        # print('nqe:')
        # print(nqe.shape)
        ax[1].imshow(deproc_img)
        target = y[i]
        ax[2].imshow(target,cmap='Greys_r')
        plt.show()


###################################################################################################
# SETUP
###################################################################################################
"""
DESCRIPTION:
    prepare the commandline parser object
PARAMS:
RETURNS:
    cmdln: an argparse.ArgumentParser object
        the returned object can be used to easily parse commandline args
"""
def prep_cmdln_parser():
    usage = "usage: %prog [options]"
    cmdln = argparse.ArgumentParser(usage)
    cmdln.add_argument("--task", action="store", dest="TASK", default='preprocess-coco',
                                help="specify the preprocessing task to perform [default: %default].")
    cmdln.add_argument("--raw-dir", action="store", dest="RAW_DIR", default='data/raw/coco_all',
                                help="specify the directory in which to the raw images are saved [default: %default].")
    cmdln.add_argument("--save-dir", action="store", dest="SAVE_DIR", default='data/preprocessed/coco_all',
                                help="specify the directory in which to save processed images as training/val/test data [default: %default].")
    cmdln.add_argument("--processed-dir", action="store", dest="PROCESSED_DIR", default='data/preprocessed/coco_all',
                                help="specify the directory in which the processed training/val/test data is saved [default: %default].")
    cmdln.add_argument("--processed-name", action="store", dest="PROCESSED_NAME", default='X_train.npy',
                                help="specify the name of the processed training/val/test file to open for deprocessing [default: %default].")
    cmdln.add_argument("--targets-name", action="store", dest="TARGETS_NAME", default='y_train.npy',
                                help="specify the name of the processed training/val/test file to open for deprocessing [default: %default].")
    cmdln.add_argument("--proc-shape", action="store", dest="PROC_SHAPE", default='3,224,224',
                                help="specify the desired processed shape for the preprocessed images [default: %default].")
    cmdln.add_argument("--target-h", action="store", dest="TARGET_H", default=60, type=int,
                                help="specify the desired target height for the downsampled (and grayscale) images that the autoencoder is trained to reconstruct [default: %default].")
    cmdln.add_argument("--lim", action="store", dest="LIM", default=1000, type=int,
                                help="specify how many images to preprocess (or deprocess) in total (including training, val, and test data if preprocessing) [default: %default].")
    cmdln.add_argument("--train-ratio", action="store", dest="TRAIN_RATIO", default=.8, type=float,
                                help="specify the percentage of the total data to use for training [default: %default].")
    cmdln.add_argument("--val-ratio", action="store", dest="VAL_RATIO", default=.1, type=float,
                                help="specify the percentage of the total data to use for validation [default: %default].")
    cmdln.add_argument("--test-ratio", action="store", dest="TEST_RATIO", default=.1, type=float,
                                help="specify the percentage of the total data to use for testing [default: %default].")
    cmdln.add_argument("--random-split", action="store_true", dest="RANDOM_SPLIT", default=False,
                                help="specify whether to split the data into train/val/test splits randomly or not [default: %default].")
    cmdln.add_argument("--random-seed", action="store", dest="RANDOM_SEED", default=13, type=int,
                                help="specify a random seed to use for splitting the data into train/val/test splits [default: %default].")
    cmdln.add_argument("--chunk-size", action="store", dest="CHUNK_SIZE", default=5000, type=int,
                                help="specify the chunk size to use when splitting the training/val/test data into smaller chunks"
                                     "for the purpose of reducing storage size and memeory footprint [default: %default].")
    cmdln.add_argument("--targets3d", action="store_true", dest="TARGETS3D", default=False,
                                help="this flag indicates that you want the target reconstruction images to be 3d (RGB) images"
                                     "as opposed to grayscale images [default: %default].")
    return cmdln


def main():
    cmdln = prep_cmdln_parser()
    args = cmdln.parse_args()
    if args.TASK == 'preprocess-coco':
        try:
            args.PROC_SHAPE = tuple([int(d) for d in args.PROC_SHAPE.split(',')])
        except Exception:
            raise SyntaxError('proc-shape command line option: %s is not a valid option' % str(args.PROC_SHAPE))
        assert len(args.PROC_SHAPE) == 3, 'target-shape must specify exactly 3 dimensions (channels,height,width)'

        print(args.PROC_SHAPE)

        preprocess_coco(raw_dir=args.RAW_DIR, save_dir=args.SAVE_DIR, proc_shape=args.PROC_SHAPE, target_H=args.TARGET_H, train_ratio=args.TRAIN_RATIO, val_ratio=args.VAL_RATIO,
                        test_ratio=args.TEST_RATIO, random_split=args.RANDOM_SPLIT, random_seed=args.RANDOM_SEED, lim=args.LIM, chunk_size=args.CHUNK_SIZE, targets3d=args.TARGETS3D)

    elif args.TASK == 'deprocess-coco':
        deprocess_coco(args.PROCESSED_DIR, args.PROCESSED_NAME, args.TARGETS_NAME, args.LIM)

if __name__ == "__main__":
    main()
