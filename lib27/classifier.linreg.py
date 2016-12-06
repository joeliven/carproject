import os,sys
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

idx2label = ['S', 'T']

class _Classifier(object):
    def __init__(self, **kwargs):
        self.dim_img = kwargs.get('dim_img', 224)
        self.nb_channels = kwargs.get('nb_channels', 3)

    def hacky_predict(self, X, magic_i1=50, magic_i2=100, magic_j1=50, magic_j2=224-50, magic_thresh_light=.985, magic_thresh_sum=20., verbose=False):
        X = self.threshold(X, magic_thresh_light)
        magic_j1, magic_j2 = self.get_window(X, width_orig=40)
        X_below = X[magic_i1:magic_i2,magic_j1:magic_j2]
        X_below_sum = np.sum(X_below)
        print('Sum: %.2f' % X_below_sum)
        # plt.imshow(X)
        txt = 'j1: %d \tj2:%d' % (magic_j1,magic_j2)
        # plt.text(0, 0, txt, color='r', fontsize=15, fontweight='bold')
        # plt.show()
        # plt.imshow(X_below)
        # plt.show()
        if X_below_sum > magic_thresh_sum:
            return False
        else:
            return True

    def predict(self, X, verbose=False):
        """make a prediction on a single incoming image"""
        start_time = time.time()
        pred =  self.hacky_predict(X=X, verbose=verbose)
        if pred:
            pred_str = 'TURN'
        else:
            pred_str = 'STRAIGHT'
        duration = time.time() - start_time
        if verbose:
            print('prediction: %s (duration: %.3f)' % (pred_str,duration))
        return pred, duration


    def threshold(self, X, thresh):
        X = np.where(X >= thresh, X, 0)
        X_mask = X[:, :, 0] * X[:, :, 1] * X[:, :, 2]
        X_mask = np.asarray([X_mask, X_mask, X_mask]).transpose(1, 2, 0)
        X = X * X_mask
        return X

    def get_window(self, X, width_orig=40, patch_sum_thresh=50):
        # plt.imshow(X)
        # plt.show()
        nb_patches = 224 // width_orig
        # print 'nb_patches',nb_patches
        start = 0
        end = width_orig-1
        width = width_orig
        slopes = np.zeros(shape=(nb_patches+1,))
        best_err = np.inf
        best_start = 0
        for patch_num in range(nb_patches+1):
            if patch_num == nb_patches:
                width = 224 - start
            patch = X[:,start:end+1]
            patch = np.sum(patch,axis=-1)
            # print 'patch_num', patch_num,'patch.shape', patch.shape, 'width', width
            assert patch.shape == (224,width)
            patch_sum = np.sum(patch)
            # print 'patch_sum', patch_sum
            if patch_sum < patch_sum_thresh:
                start += width
                end += width
                continue
            m_b, X1, y = self.fit_line(patch)
            slopes[patch_num] = m_b[0]
            err = self.get_error(m_b, X1, y, start)
            # print 'patch_num',patch_num, 'err', err
            if err < best_err:
                best_err = err
                best_start = start
            start += width
            end += width
        best_start_idx = np.argmax(np.abs(slopes))
        best_start = width_orig * best_start_idx
        magicj1 = best_start
        if magicj1 > 224 - width_orig:
            magicj2 = best_start + width
        else: magicj2 = best_start + width_orig
        # print('magicj1: %d \tmagicj2: %d' % (magicj1,magicj2))

        return magicj1,magicj2

    def fit_line(self, patch):
        # patch = np.where(patch > 0, 1., 0.)
        # print 'patch', patch.shape
        # print patch
        y,X = np.where(patch > 0.)
        # X,y = np.where(patch > 0.)
        y = 224. - y
        # print('y.shape')
        # print(y.shape)
        # print('X.shape')
        # print(X.shape)
        biases = np.ones(shape=X.shape[0])
        # print 'biases.shape', biases.shape
        X1_T = np.asarray([X,biases])
        # print 'X1_T.shape', X1_T.shape
        X1 = X1_T.transpose()
        # print('X1.shape')
        # print(X1.shape)
        X1_T_X1 = np.matmul(X1_T, X1)
        # print 'X1_T_X1.shape', X1_T_X1.shape
        # raw_input('...')
        try:
            X1_T_X1_inv = np.linalg.inv(X1_T_X1)
        except np.linalg.linalg.LinAlgError:
            X1_T_X1_inv = np.linalg.pinv(X1_T_X1)
        m_b = np.matmul(np.matmul(X1_T_X1_inv, X1_T),y)
        # print('m_b.shape')
        # print(m_b.shape)
        # print(m_b)
        return m_b, X1, y

    def get_error(self, m_b, X1, y, j1):
        Xs = X1[:,0]
        # print 'Xs', Xs.shape
        # print Xs

        yhat = np.matmul(X1, m_b)
        err = yhat - y
        err = np.abs(err)
        # err = err**2
        # print 'yhat', yhat.shape
        # print yhat
        # print 'y', y.shape
        # print y

        plt.plot(Xs+j1,y,'bx', Xs+j1,yhat, 'r')
        # plt.plot(y,Xs+j1,'bx',yhat,Xs+j1, 'r')
        plt.axis((0,224,0,224))
        plt.show()
        # print 'err', err.shape
        # print err
        # print('X1.shape')
        # print(X1.shape)
        # raw_input('here;lajrlej;')
        err = np.sum(err)
        err = float(err) / float(X1.shape[0])
        # print('err: %.4f' % err)
        return err

if __name__ == '__main__':
    if len(sys.argv) < 4:
      print ("classifier.py: <meta_path> <weights_path> <image_path> <verbose> (verbose is optional)")
      sys.exit()

    meta_path = sys.argv[1]
    weights_path = sys.argv[2]
    image_path = sys.argv[3]
    labels_path = sys.argv[4]
    print('meta_path')
    print(meta_path)
    print('weights_path')
    print(weights_path)
    print('image_path')
    print(image_path)

    verbose = False
    if len(sys.argv) == 6:
        verbose = True
        print('verbose is True')
    else:
        print('verbose is False')

    model = _Classifier(meta_path=meta_path, weights_path=weights_path)
    dir = 'data/raw/4s-cclkwse'
    paths = [os.path.join(dir,x) for x in os.listdir(dir) if x.endswith('.jpg')]
    # paths = paths[0:200]
    paths = paths[70:210]
    print(len(paths))
    print('len(paths)')
    imgs = []
    for p in paths:
        imgs.append(preprocess_image(imread(p)))
    imgs = np.asarray(imgs)
    print('imgs.shape')
    print(imgs.shape)
    # imgs = np.load(image_path)
    # imgs = imread(image_path)
    labels = np.load(labels_path)

    # imgs = np.asarray([imgs]).astype(dtype=np.float64)
    # imgs = imgs[518]
    print('imgs.dtype')
    print(imgs.dtype)
    print('imgs.shape')
    print(imgs.shape)
    print 'imgs.sum', imgs.sum()/imgs.shape[0]
    # imgs = np.asarray([imgs])
    print('np.max(imgs[0])')
    print(np.max(imgs[0]))
    # noise = np.random.random_sample(size=(224,224,3))
    # noise = np.random.randint(0,250,size=(1,480,640,3)).astype(dtype=np.uint8)
    # imgs += noise
    labels = np.asarray([[1,0]])
    labels = np.zeros(shape=(imgs.shape[0],2))
    labels[:,0] = 1
    # plt.imshow(imgs[0])
    # plt.show()

    print('imgs.dtype')
    print(imgs.dtype)
    print('imgs.shape')
    print(imgs.shape)
    print('labels.shape')
    print(labels.shape)
    assert imgs.shape[0] == labels.shape[0]

    num_cor = 0
    tot_dur = 0.
    for i,img in enumerate(imgs):
        if i % 10 != 0:
        # if i % 1 != 0:
            continue
        # img = preprocess_image(deprocess_image(img))
        # img = preprocess_image(img)
        print 'img.sum', img.sum()
        print('img.dtype inside loop')
        print(img.dtype)
        plt.imshow(img)
        plt.show()

        # print(np.sum(img))
        # raw_input('...np.sum...')
        turn, dur = model.predict(img, verbose=verbose)
        tot_dur += dur
        label = labels[i]
        assert label.shape == (2,)
        if turn == True and np.argmax(label) == 1:
            num_cor += 1
            print('%d: correct' % i)
        elif turn == False and np.argmax(label) == 0:
            num_cor += 1
            print('%d: correct' % i)
        else:
            print('%d: wrong' % i)

        quit = raw_input('press q to quit, enter to continue:')
        if quit == 'q':
            sys.exit(1)
    print('num_cor')
    print(num_cor)
    print('imgs.shape[0]')
    print(imgs.shape[0])
    acc = float(num_cor) / float(imgs.shape[0])
    print('acc')
    print(acc)
    avg_dur = tot_dur / float(imgs.shape[0])
    print('avg_dur')
    print(avg_dur)
    print('acc: %.4f \tavg_dur: %.4f $' % (acc, avg_dur))
    print('DONE')


