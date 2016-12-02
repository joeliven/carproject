# import numpy as np
import skimage.transform
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import transform as tf
#from skimage.util import crop
#from skimage.feature import (corner_harris, corner_fast, CENSURE,
#                             corner_peaks, ORB, blob_dog, daisy)
from scipy import signal
import matplotlib.pyplot as plt
#import urllib.request, urllib.error, urllib.parse, os, tempfile
import os, json
import numpy as np
#import h5py
import math, time, sys
from numpy.lib.arraypad import _validate_lengths

"""
Utility functions used for viewing and processing images.
"""

def load_idx2class(txtpath='data/raw/synset_words.txt'):
    classes = list()
    with open(txtpath, 'r') as f:
        for line in f:
            syns = line.split(',')
            first = ' '.join(syns[0].split()[1:])
            syns[0] = first
            syns = [syn.strip() for syn in syns]
            classes.append(syns)
    return classes

def rgb2bgr(img_sk):
    # convert skimage color channel format (RGB) to opencv format (BGR)
    if img_sk.shape[0] == 3:
        img_cv2 = img_sk.copy()
        img_cv2[0,:,:] = img_sk[2,:,:]
        img_cv2[2,:,:] = img_sk[0,:,:]
    else:
        img_cv2 = img_sk.copy()
        img_cv2[:,:,0] = img_sk[:,:,2]
        img_cv2[:,:,2] = img_sk[:,:,0]
    return img_cv2

def bgr2rgb(img_cv2):
    # convert opencv color channel format (BGR) to skimage format (RGB)
    if img_cv2.shape[0] == 3:
        img_sk = img_cv2.copy()
        img_sk[0,:,:] = img_cv2[2,:,:]
        img_sk[2,:,:] = img_cv2[0,:,:]
    else:
        img_sk = img_cv2.copy()
        img_sk[:,:,0] = img_cv2[:,:,2]
        img_sk[:,:,2] = img_cv2[:,:,0]
    return img_sk

def channel_last2first(img, nb_channels=3):
    if len(img.shape) == 3:
        # convert image from shape (H,W,nb_channels) to (nb_channels,H,W) if needed
        if img.shape[0] == nb_channels:
            # no transpose needed:
            return img
        else:
            return img.transpose(2,0,1)
    elif len(img.shape) == 4:
        # convert image(s) from shape (nb_samples,H,W,nb_channels) to (nb_samples,nb_channels,H,W) if needed
        if img.shape[1] == nb_channels:
            # no transpose needed:
            return img
        else:
            return img.transpose(0,3,1,2)

def channel_first2last(img, nb_channels=3):
    if len(img.shape) == 3:
        # convert image from shape (nb_channels,H,W) to (H,W,nb_channels) if needed
        if img.shape[2] == nb_channels:
            # no transpose needed:
            return img
        else:
            return img.transpose(1,2,0)
    elif len(img.shape) == 4:
        # convert image(s) from shape (nb_samples,nb_channels,H,W) to (nb_samples,H,W,nb_channels) if needed
        if img.shape[3] == nb_channels:
            # no transpose needed:
            return img
        else:
            return img.transpose(0,2,3,1)

def crop(ar, crop_width, copy=False, order='K'):
    ar = np.array(ar, copy=False)
    crops = _validate_lengths(ar, crop_width)
    slices = [slice(a, ar.shape[i] - b) for i, (a, b) in enumerate(crops)]
    if copy:
        cropped = np.array(ar[slices], order=order, copy=True)
    else:
        cropped = ar[slices]
    return cropped


def preprocess_image(img, proc_shape=(3, 224, 224), mean_pixel=[0.,0.,0.], dbg=False):
# def preprocess_image(img, proc_shape=(3, 224, 224), mean_pixel=[103.939, 116.779, 123.68], dbg=False):
    """
    Convert to float, transpose, and subtract mean pixel

    Input:
    - img: (H, W, nb_channels)

    Returns:
    - (H, W, nb_channels)
    with the mean pixel subtracted from each color channel
    """
    if len(img.shape) != 3:
        print('skipping this image due to unknown image dims: %s' % str(img.shape))
        return None
    if img.shape[2] != 3:
        if img.shape[0] == 3 and len(img.shape) == 3:
            img = channel_first2last(img)
        else:
            print('skipping this image due to unknown image shape: %s' % str(img.shape))
            return None
    if dbg:
        assert (img.shape[2] == 3), 'Dimensions of the input image are off...img.shape[2] != 3'
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(img)
        print('img.shape:%s' % str(img.shape))
    H = img.shape[0]
    W = img.shape[1]
    target_ratio = proc_shape[1] / proc_shape[2]
    img_ratio = H / W
    if dbg:
        print('target_ratio:%f' % target_ratio)
        print('img_ratio:%f' % img_ratio)
    crop_dims = None
    if img_ratio < target_ratio:
        # img is wider than target (by ratio) so we rescale to target's H, and then crop to target's W
        if dbg:
            print('wider')
        scale = proc_shape[1] / float(H)

        crop_ = (round(W * scale) - proc_shape[2]) // 2
        if dbg:
            assert ((crop_ * 2) + proc_shape[2] == round(W * scale)) or ((crop_ * 2) + proc_shape[2] == round(W * scale) - 1), 'crop dims are wrong:%f != %d' % (((crop_ * 2) + proc_shape[2]), round(W * scale))
        if (crop_ * 2) + proc_shape[2] == round(W * scale):
            crop_dims = ((0,0),(crop_,crop_),(0,0))
        else: # (crop_ * 2) + target_shape[2] == round(W * scale) - 1:
            crop_dims = ((0,0),(crop_ + 1,crop_),(0,0))
    elif img_ratio > target_ratio:
        # img is taller than target (by ratio) so we rescale to target's W, and then crop to target's H
        if dbg:
            print('taller')
        scale = proc_shape[2] / W
        crop_ = (round(H * scale) - proc_shape[1]) // 2
        if dbg:
            assert ((crop_ * 2) + proc_shape[1] == round(H * scale)) or ((crop_ * 2) + proc_shape[1] == round(H * scale) - 1), 'crop dims are wrong:%f != %d' % (((crop_ * 2) + proc_shape[1]), round(H * scale))
        if (crop_ * 2) + proc_shape[1] == round(H * scale):
            crop_dims = ((crop_,crop_),(0,0),(0,0))
        else: # (crop_ * 2) + target_shape[2] == round(H * scale) - 1:
            crop_dims = ((crop_ + 1,crop_),(0,0),(0,0))
    else:
        # img is has exact same H:W ratio as target so we rescale to target's exact W and H without having to crop at all
        if dbg:
            print('same ratio')
        scale = proc_shape[2] / W
        scale2 = proc_shape[1] / H
        if dbg:
            assert scale == scale2, 'scale ratios should be the same but are not'
    if dbg:
        print('scale:%f' % scale)
        print('crop_dims:%s' % str(crop_dims))

    img_sk = skimage.transform.rescale(img, scale)
    if dbg:
        print('(after scaling) img_sk.shape:%s' % str(img_sk.shape))
    if dbg:
        assert ((img_sk.shape[0] == proc_shape[1]) or (img_sk.shape[1] == proc_shape[2])), 'img was not rescaled to the correct dims'
    if crop_dims is not None:
        img_sk = crop(img_sk, crop_width=crop_dims)
        if dbg:
            print('(after cropping) img_sk.shape:%s' % str(img_sk.shape))
    assert ((img_sk.shape[0] == proc_shape[1]) and (img_sk.shape[1] == proc_shape[2])), 'img was not rescaled or cropped to the corrects dims'
    if dbg:
        ax[1].imshow(img_sk)
        #plt.show()
    img = rgb2bgr(img_sk)
    mean_pixel = np.asarray(mean_pixel) / 255.
    for c in range(3): # c represents 'channel'
        img[:, :, c] -= mean_pixel[c]
    # img = channel_last2first(img)
    return img


def deprocess_image(img, mean_pixel=[0., 0., 0.]):
    """
    Add mean pixel, transpose, and convert to uint8

    Input:
    - (nb_channels, H, W) or
    - (H, W, nb_channels)

    Returns:
    - (H, W, nb_channels) with the mean pixel value added back on to each channel
    """
    deproc_img = img.copy()
    if deproc_img.shape[0] == 3:
        deproc_img = channel_first2last(deproc_img)
    assert deproc_img.shape[2] == 3
    mean_pixel = np.asarray(mean_pixel) / 255.
    for c in range(3): # c represents 'channel'
        deproc_img[:, :, c] += mean_pixel[c]
    deproc_img = bgr2rgb(deproc_img)
    return deproc_img


def downsample_image(img, new_size=(60,60), grey=True):
    """
    downsample the image, reszinging it to new_size and optionally convert to greyscale

    Input:
    - img
        image to be downsampled
    - new_size
        tuple of shape (W,H) indicating the desired new width and height
    - grey
        boolean: whether or not to convert the image to greyscale

    Returns:
    - downsampled and optionally greyscaled image (Matrix if greyscale=True, 3d tensor if greyscale=False)
    """
    if grey:
        img = rgb2gray(img)
        img = skimage.transform.resize(img, (new_size[0],new_size[1]), preserve_range=False)
        return img
    else:
        raise NotImplementedError('downsample_image() function not implemented yet for grey=False')


def inc_row_col(row, col, nrows, ncols):
    new_row = row + 1 if (col + 1) == ncols else row
    new_col = (col + 1) % ncols
    return  new_row, new_col

def display_features(orig_img, dsae_feats=None, dsae_duration=None, other_feats=['corner_harris','corner_fast','CENSURE_STAR','CENSURE_Octagon','CENSURE_DoB','ORB','daisy'] ):
    """
    display the original image, optionally displaying the identified features over it

    Input:
    - orig_img
        original image
    - dsae_feats
        2d array of spatial feature coordinates of shape (nb_feats,2)
        Note, the dimension length 2 has its first element representing the i coordinate and
        its second element representing the j coordinate
    - dsea_duration
        float: the amount of time it took to preprocess the image and produce the predicted encoding (features)
    - other_feats
        list of strings: a list of the other types of feature detectors to visually compare with ours
    Returns:
    """
    nimgs = len(other_feats) + 1
    nrows = 2
    ncols = nimgs // 2
    if nimgs % 2 != 0:
        ncols += 1
    if ncols == 1:
        ncols += 1
    assert (nrows * ncols >= nimgs)
    print('nrows:%d\tncols:%d' % (nrows,ncols))
    fig, ax = plt.subplots(figsize=(20,10),nrows=nrows, ncols=ncols)
    row, col = 0, 0
    gray_img = rgb2gray(orig_img)
    dsae_feats_orig = dsae_feats.copy()
    if orig_img.shape[0] == 3:
        H = orig_img.shape[1]
        W = orig_img.shape[2]
    else:
        H = orig_img.shape[0]
        W = orig_img.shape[1]

    # dsae_feats_orig[:,0] *= H
    # dsae_feats_orig[:,1] *= W

    # plt.imshow(orig_img)
    ax[row,col].imshow(orig_img)
    if dsae_feats is not None:
        extractor = 'DSAE_VGG (ours)'
        # plt.scatter(x=dsae_feats_orig[:,1], y=dsae_feats_orig[:,0], c='r', s=15)
        ax[row,col].scatter(x=dsae_feats_orig[:,1], y=dsae_feats_orig[:,0], c='r', s=15)
        ax[row, col].text(0, 0, extractor, color='r', fontsize=25, fontweight='bold')
        print('dsae_duration:')
        print(dsae_duration)
        # sys.exit(1)
        if dsae_duration:
            ax[row, col].text(0, 15, '%s ms' % dsae_duration, color='b', fontsize=15)
            print('%s: %f ms' % (extractor,dsae_duration))
    # plt.show()
    row, col = inc_row_col(row,col,nrows,ncols)

    if 'ORB' in other_feats:
        extractor = 'ORB'
        descriptor_extractor = ORB(n_keypoints=256)
        start = time.time()
        descriptor_extractor.detect_and_extract(gray_img)
        orb_feats = descriptor_extractor.keypoints
        end = time.time()
        duration = (end - start) * 1000. # milliseconds now
        # descriptors1 = descriptor_extractor.descriptors
        ax[row,col].imshow(orig_img)
        ax[row,col].scatter(x=orb_feats[:, 1], y=orb_feats[:, 0], c='b', s=15)
        ax[row,col].text(0, 0, extractor, color='b', fontsize=25, fontweight='bold')
        print('H',H)
        ax[row,col].text(0, H+80, '%s ms' % duration, color='b', fontsize=15)
        print('%s: %f ms' % (extractor, duration))
        row, col = inc_row_col(row, col, nrows, ncols)
    if 'blob_dog' in other_feats:
        extractor = 'blob_dog'
        start = time.time()
        blob_dog_feats = blob_dog(gray_img)
        end = time.time()
        duration = (end - start) * 1000. # milliseconds now
        ax[row,col].imshow(orig_img)
        ax[row,col].scatter(x=blob_dog_feats[:, 1], y=blob_dog_feats[:, 0], c='g', s=15)
        ax[row,col].text(0, H+80, '%s ms' % duration, color='g', fontsize=15)
        print('%s: %f ms' % (extractor, duration))
        row, col = inc_row_col(row,col,nrows,ncols)
    if 'corner_fast' in other_feats:
        extractor = 'corner_fast'
        start = time.time()
        corner_fast_feats = corner_peaks(corner_fast(gray_img))
        end = time.time()
        duration = (end - start) * 1000. # milliseconds now
        ax[row,col].imshow(orig_img)
        ax[row,col].scatter(x=corner_fast_feats[:, 1], y=corner_fast_feats[:, 0], c='m', s=15)
        ax[row,col].text(0, 0, 'corner_fast', color='m', fontsize=25, fontweight='bold')
        ax[row,col].text(0, H+80, '%s ms' % duration, color='m', fontsize=15)
        print('%s: %f ms' % (extractor, duration))
        row, col = inc_row_col(row,col,nrows,ncols)
    if 'corner_harris' in other_feats:
        extractor = 'corner_harris'
        start = time.time()
        corner_harris_feats = corner_peaks(corner_harris(gray_img))
        end = time.time()
        duration = (end - start) * 1000. # milliseconds now
        ax[row,col].imshow(orig_img)
        ax[row,col].scatter(x=corner_harris_feats[:, 1], y=corner_harris_feats[:, 0], c='m', s=15)
        ax[row,col].text(0, 0, 'corner_harris', color='m', fontsize=25, fontweight='bold')
        ax[row,col].text(0, H+80, '%s ms' % duration, color='m', fontsize=15)
        print('%s: %f ms' % (extractor, duration))
        row, col = inc_row_col(row,col,nrows,ncols)
    if 'CENSURE_STAR' in other_feats:
        extractor = 'CENSURE_STAR'
        censure = CENSURE(mode='STAR')
        start = time.time()
        censure.detect(gray_img)
        censure_star_keypoints = censure.keypoints
        end = time.time()
        duration = (end - start) * 1000. # milliseconds now
        ax[row,col].imshow(orig_img)
        ax[row,col].scatter(x=censure_star_keypoints[:, 1], y=censure_star_keypoints[:, 0], c='k', s=15)
        ax[row,col].text(0, 0, 'CENSURE_STAR', color='k', fontsize=25, fontweight='bold')
        ax[row,col].text(0, H+80, '%s ms' % duration, color='k', fontsize=15)
        print('%s: %f ms' % (extractor, duration))
        row, col = inc_row_col(row,col,nrows,ncols)
    if 'CENSURE_Octagon' in other_feats:
        extractor = 'CENSURE_Octagon'
        censure = CENSURE(mode='Octagon')
        start = time.time()
        censure.detect(gray_img)
        censure_oct_keypoints = censure.keypoints
        end = time.time()
        duration = (end - start) * 1000. # milliseconds now
        ax[row,col].imshow(orig_img)
        ax[row,col].scatter(x=censure_oct_keypoints[:, 1], y=censure_oct_keypoints[:, 0], c='k', s=15)
        ax[row,col].text(0, 0, 'CENSURE_Octagon', color='k', fontsize=25, fontweight='bold')
        ax[row,col].text(0, H+80, '%s ms' % duration, color='k', fontsize=15)
        print('%s: %f ms' % (extractor, duration))
        row, col = inc_row_col(row,col,nrows,ncols)
    if 'CENSURE_DoB' in other_feats:
        extractor = 'CENSURE_DoB'
        censure = CENSURE(mode='DoB')
        start = time.time()
        censure.detect(gray_img)
        censure_dob_keypoints = censure.keypoints
        end = time.time()
        duration = (end - start) * 1000. # milliseconds now
        ax[row,col].imshow(orig_img)
        ax[row,col].scatter(x=censure_dob_keypoints[:, 1], y=censure_dob_keypoints[:, 0], c='k', s=15)
        ax[row,col].text(0, 0, 'CENSURE_DoB', color='k', fontsize=25, fontweight='bold')
        ax[row,col].text(0, H+80, '%s ms' % duration, color='k', fontsize=15)
        print('%s: %f ms' % (extractor, duration))
        row, col = inc_row_col(row,col,nrows,ncols)
    if 'daisy' in other_feats: # SIFT-like feature descriptor
        extractor = 'daisy'
        start = time.time()
        # descs, descs_img = daisy(gray_img, visualize=True)

        descs, descs_img = daisy(gray_img, step=50, radius=25, rings=2, histograms=6,
                                 orientations=8, visualize=True)
        end = time.time()
        duration = (end - start) * 1000. # milliseconds now
        ax[row,col].imshow(orig_img)
        ax[row,col].imshow(descs_img)
        ax[row,col].text(0, 0, 'Daisy', color='w', fontsize=25, fontweight='bold')
        ax[row,col].text(0, H+80, '%s ms' % duration, color='w', fontsize=15)
        print('%s: %f ms' % (extractor, duration))
        row, col = inc_row_col(row,col,nrows,ncols)
    plt.show()


def display_reconstruction(orig_img, ground_truth, recon_img, recon_duration=None, targets3d=False):
    """
    display the original image, with standard downsampled/greyscale image next to it,
    and our reconstructed downsampled/greyscale image next to that

    Input:
    - orig_img
        original image
    - small_gray_img
        the downsampled/greyscale version of the original img with dims matching our reconstructed version
    - recon_img
        our reconstructed downsampled/greyscale version of the original img
    - recon_duration
        float: the amount of time it took to preprocess the image and produce the predicted reconstruction
    Returns:
    """
    fig, ax = plt.subplots(figsize=(20,10),nrows=1, ncols=3)
    title = 'orig_img'
    ax[0].imshow(orig_img)
    ax[0].text(0, 0, title, color='r', fontsize=25, fontweight='bold')

    title = 'ground_truth (orig)'
    if targets3d:
        print(ground_truth.shape)
        input('ground_truth.shape')
        ax[1].imshow(ground_truth)
    else:
        ax[1].imshow(ground_truth, cmap='Greys_r')
    ax[1].text(0, 0, title, color='b', fontsize=25, fontweight='bold')

    title = 'reconstruction'
    if targets3d:
        ax[2].imshow(recon_img)
    else:
        ax[2].imshow(recon_img, cmap='Greys_r')
    ax[2].text(0, 0, title, color='m', fontsize=25, fontweight='bold')

    if recon_duration:
        # ax[2].text(0, 0, '%s ms' % recon_duration, color='m', fontsize=15)
        ax[2].text(0, 5, '%s ms' % recon_duration, color='m', fontsize=15)
        print('recon_duration %s: %f ms' % (title, recon_duration))
    plt.show()


def display_activation(orig_img, activation_imgs, layer_name, map_nums, activation_duration=None, dim_ordering='tf'):
    """
    display the original image, with the images activation from layer 'layer_name' next to it,

    Input:
    - orig_img
        original image
    - activation_img
        the image produced from the activation of a certain intermediate layer of the network
    - layer_name
        the name of the layer in the network from which the activation is being produced (e.g. 'conv1_1' or 'conv3_2', or 'dense2')
    - activation_duration
        float: the amount of time it took to preprocess the image and produce the activation image
    Returns:
    """
    if dim_ordering == 'tf':
        activation_imgs = activation_imgs.transpose(2,0,1)

    ntiles = activation_imgs.shape[0] + 1
    nrows = int(math.sqrt(ntiles))
    ncols = math.ceil(ntiles / nrows)
    print('ntiles:%d' % ntiles)
    print('nrows:%d' % nrows)
    print('ncols:%d' % ncols)
    assert(ncols * nrows >= ntiles)
    fig, ax = plt.subplots(figsize=(20,10),nrows=nrows, ncols=ncols)
    title = 'orig_img'
    col = 0
    row = 0
    if nrows > 1:
        ax[row,col].imshow(orig_img)
        ax[row,col].text(0, 0, title, color='r', fontsize=25, fontweight='bold')
    else:
        ax[col].imshow(orig_img)
        ax[col].text(0, 0, title, color='r', fontsize=25, fontweight='bold')

    for feature_map_num in range(activation_imgs.shape[0]):
        print('***********************************')
        print('\nfeature_map[%d]:' % map_nums[feature_map_num])
        feat_map = activation_imgs[feature_map_num]

        feat_map_flat = feat_map.reshape(feat_map.shape[0] * feat_map.shape[1])
        e_x = np.exp(feat_map_flat - np.amax(feat_map_flat, axis=-1, keepdims=True))
        s = e_x.sum(axis=-1, keepdims=True)
        soft1d = e_x / s
        soft2d = soft1d.reshape(feat_map.shape[0], feat_map.shape[1])
        idxs_i = np.arange(feat_map.shape[0])
        idxs_j = np.arange(feat_map.shape[1])
        i = (soft2d.transpose(1, 0) * idxs_i).transpose(1, 0).sum(axis=(0, 1))
        # i /= (feat_map.shape[0] - 1.)
        j = (soft2d * idxs_j).sum(axis=(0, 1))
        # j /= (feat_map.shape[1] - 1.)
        ij = np.stack([i, j], axis=-1)
        print('softargmax:' + str(ij))

        # print(activation_imgs[feature_map_num])
        print(activation_imgs[feature_map_num].shape)
        avg = np.mean(activation_imgs[feature_map_num])
        print('avg:%f' % avg)
        top10 = np.dstack(np.unravel_index(np.argsort(activation_imgs[feature_map_num].ravel())[::-1][:50], activation_imgs[feature_map_num].shape))[0]
        # RIGHT HERE
        # np.savetxt('Map_batchnorm3_1.nptxt',activation_imgs[feature_map_num], fmt='%.8f')

        # top10 = np.dstack(np.unravel_index(np.argsort(activation_imgs[feature_map_num].ravel())[::-1][:], activation_imgs[feature_map_num].shape))[0]
        top10vals = np.asarray([activation_imgs[feature_map_num][i,j] for [i,j] in top10])
        print('top6:')
        for x in range(len(top10)):
            print(str(top10[x]) + ':\t' + str(top10vals[x]))

        midpt = np.prod(activation_imgs[feature_map_num].shape) // 2
        print('midpt:%d' % midpt)

        mid10 = np.dstack(np.unravel_index(np.argsort(activation_imgs[feature_map_num].ravel())[::-1][midpt - 3: midpt + 3], activation_imgs[feature_map_num].shape))[0]
        mid10vals = np.asarray([activation_imgs[feature_map_num][i,j] for [i,j] in mid10])
        print('mid6 :')
        for x in range(len(mid10)):
            print(str(mid10[x]) + ':\t' + str(mid10vals[x]))

        bottom10 = np.dstack(np.unravel_index(np.argsort(activation_imgs[feature_map_num].ravel())[::-1][-6:], activation_imgs[feature_map_num].shape))[0]
        bottom10vals = np.asarray([activation_imgs[feature_map_num][i,j] for [i,j] in bottom10])
        print('bottom6 :')
        for x in range(len(bottom10)):
            print(str(bottom10[x]) + ':\t' + str(bottom10vals[x]))

        # RIGHT HERE
        col = ((feature_map_num + 1) % ncols)
        print('featur_map_num:%d' % feature_map_num)
        print('ncols:%d' % ncols)
        if (feature_map_num + 1) % ncols == 0:
            row += 1
        print('row:%d' % row)
        print('col:%d' % col)
        title = 'layer: %s, \tmap_num: %d' % (layer_name,map_nums[feature_map_num])
        # ax[1].imshow(activation_img)
        if nrows > 1:
            ax[row,col].imshow(activation_imgs[feature_map_num], cmap='Greys_r')
            ax[row,col].text(0, -10, title, color='m', fontsize=15, fontweight='bold')
            if activation_duration:
                # ax[row,col].text(0, 0, '%s ms' % recon_duration, color='m', fontsize=15)
                ax[row,col].text(0, 10, '%s ms' % activation_duration, color='m', fontsize=10)
                print('activation_duration %s: %f ms' % (title, activation_duration))
        else:
            ax[col].imshow(activation_imgs[feature_map_num], cmap='Greys_r')
            ax[col].text(0, -10, title, color='m', fontsize=15, fontweight='bold')
            if activation_duration:
                # ax[row,col].text(0, 0, '%s ms' % recon_duration, color='m', fontsize=15)
                ax[col].text(0, 10, '%s ms' % activation_duration, color='m', fontsize=10)
                print('activation_duration %s: %f ms' % (title, activation_duration))

    plt.show()


def display_saliency_maps(orig_img, saliency_maps, cls=0, duration=None):
    """
    display the original image, with the images saliency map(s) from certain classes next to it,

    Input:
    - orig_img
        original image
    - saliency_maps

    - cls

    - duration
        float: the amount of time it took to preprocess the image and produce the saliency map(s)
    Returns:
    """
    nrows = saliency_maps.shape[0] + 1
    ncols = 3
    print('nrows:%d' % nrows)
    print('ncols:%d' % ncols)
    fig, ax = plt.subplots(figsize=(20,10),nrows=nrows, ncols=ncols)
    title = 'orig_img'
    col = 0
    row = 0
    if nrows > 1:
        ax[row,col].imshow(orig_img)
        ax[row,col].text(0, 0, title, color='r', fontsize=25, fontweight='bold')
    else:
        ax[col].imshow(orig_img)
        ax[col].text(0, 0, title, color='r', fontsize=25, fontweight='bold')

    for map_num in range(saliency_maps.shape[0]):
        print('***********************************')
        map_ = saliency_maps[map_num]
        row += 1
        col = 0
        print('featur_map_num:%d' % map_num)
        print('ncols:%d' % ncols)
        print('row:%d' % row)
        print('col:%d' % col)
        title = 'cls: %s' % str(cls)
        # ax[1].imshow(activation_img)
        if nrows > 1:
            smap = bgr2rgb(channel_first2last(saliency_maps[map_num]))
            smap_abs = np.abs(smap).sum(axis=-1)
            # smap_abs = np.abs(smap).sum(axis=-1)
            smap_pos = (np.maximum(0, smap) / smap.max())[:, :, :]
            smap_neg = (np.maximum(0, -smap) / -smap.min())[:,:,:]

            ax[row,col].imshow(smap_abs, cmap='gray')
            ax[row,col].text(0, -10, title + ' saliency_abs', color='m', fontsize=15, fontweight='bold')
            if duration:
                # ax[row,col].text(0, 0, '%s ms' % recon_duration, color='m', fontsize=15)
                ax[row,col].text(0, 10, '%s ms' % duration, color='m', fontsize=10)
                print('duration %s: %f ms' % (title, duration))
            col +=  1
            ax[row,col].imshow(smap_pos)
            ax[row,col].text(0, -10, title + ' saliency_pos', color='m', fontsize=15, fontweight='bold')
            col +=  1
            ax[row,col].imshow(smap_neg)
            ax[row,col].text(0, -10, title + ' saliency_neg', color='m', fontsize=15, fontweight='bold')

        else:
            smap = bgr2rgb(channel_first2last(saliency_maps[map_num]))
            smap_abs = np.abs(smap).max(axis=-1)
            smap_pos = (np.maximum(0, smap) / smap.max())[:, :, 0]
            smap_neg = (np.maximum(0, -smap) / -smap.min())[:,:,0]

            ax[col].imshow(smap_abs, cmap='gray')
            ax[col].text(0, -10, title + ' saliency_abs', color='m', fontsize=15, fontweight='bold')
            if duration:
                # ax[row,col].text(0, 0, '%s ms' % recon_duration, color='m', fontsize=15)
                ax[col].text(0, 10, '%s ms' % duration, color='m', fontsize=10)
                print('duration %s: %f ms' % (title, duration))
            col +=  1
            ax[col].imshow(smap_pos)
            ax[col].text(0, -10, title + ' saliency_pos', color='m', fontsize=15, fontweight='bold')
            col +=  1
            ax[col].imshow(smap_neg)
            ax[col].text(0, -10, title + ' saliency_neg', color='m', fontsize=15, fontweight='bold')

    plt.show()


def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    We write the image to a temporary file then read it back. Kinda gross.
    """
    try:
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()
        with open(fname, 'wb') as ff:
            ff.write(f.read())
        img = imread(fname)
        os.remove(fname)
        return img
    except urllib.error.URLError as e:
        print( 'URL Error: ', e.reason, url)
    except urllib.error.HTTPError as e:
        print('HTTP Error: ', e.code, url)

def download_and_save_image_from_url(url, savepath):
    """
    Download an image from a URL.
    """
    try:
        f = urllib.request.urlopen(url)
        with open(savepath, 'wb') as ff:
            ff.write(f.read())
    except urllib.error.URLError as e:
        print( 'URL Error: ', e.reason, url)
    except urllib.error.HTTPError as e:
        print('HTTP Error: ', e.code, url)


# def sample_coco_minibatch(data, batch_size=100, split='train'):
#     split_size = data['%s_captions' % split].shape[0]
#     mask = np.random.choice(split_size, batch_size)
#     captions = data['%s_captions' % split][mask]
#     image_idxs = data['%s_image_idxs' % split][mask]
#     image_features = data['%s_features' % split][image_idxs]
#     urls = data['%s_urls' % split][image_idxs]
#     return captions, image_features, urls

if __name__ == "__main__":
    pass
    # base_dir = '../data/raw/coco_captioning'
    # impath = os.path.join(base_dir,'coco_img_33.jpg')
    # img = imread(impath)
    # # print(img.shape)
    # # print(img[0:20,0,0])
    # # print(type(img))
    # # plt.imshow(img)
    # # plt.show()
    # # img = img.astype(np.float32, copy=False)
    # # print(img.shape)
    # # print(img[0:20,0,0])
    # # print(type(img))
    # # plt.imshow(img)
    # # plt.show()
    # # img = skimage.transform.resize(img, (224,224), preserve_range=True)
    # # print(img.shape)
    # # print(img[0:20,0,0])
    # # print(type(img))
    # # plt.imshow(img.astype(np.uint8))
    # # plt.show()
    #
    # img_pre = preprocess_image(img)
    #
    # print(img_pre.shape)
    # print(img_pre[0:20,0,0])
    # print(type(img_pre))
    # plt.imshow(img_pre.astype(np.uint8).transpose(1,2,0))
    # plt.show()
    # small = downsample_image(img)
    # print(small.shape)
    # plt.imshow(small,cmap='gray')
    # plt.show()
    #
    #
    # # resized = skimage.transform.resize(img, (224,224), preserve_range=False)
    # # print(resized.shape)
    # # print(resized[0:20,0,0])
    # # print(type(resized))
    # # plt.imshow(resized)
    # # plt.show()
    # # processed = preprocess_image(resized)
    # # print(processed.shape)
    # # print(processed[0:10,0,0])
    # # print(type(processed))
    # # plt.imshow(processed)
    # # plt.show()
    #
