import os,sys
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from vgg6t_encoder import get_encoder

idx2label = ['S', 'T']

class _Classifier(object):
    def __init__(self, **kwargs):
        self.dim_img = kwargs.get('dim_img', 224)
        self.nb_channels = kwargs.get('nb_channels', 3)
        self.META_PATH = kwargs.get('meta_path')
        assert '.meta' in self.META_PATH
        self.RESTORE_PATH = kwargs.get('weights_path')

        with tf.Graph().as_default():
            self.inputs_pl = tf.placeholder(tf.float32, [None, self.dim_img, self.dim_img, self.nb_channels])
            self.preds, _ = get_encoder(inputs_pl=self.inputs_pl)
            self.sess = tf.Session()
            # restore the trained model:
            saver = tf.train.Saver()
            saver.restore(self.sess,self.RESTORE_PATH)

    def hacky_predict(self, X, magic_i1=50, magic_i2=100, magic_j1=50, magic_j2=224-50, magic_thresh_light=.985, magic_thresh_sum=20., verbose=False):
        X = self.threshold(X, magic_thresh_light)
        X_below = X[magic_i1:magic_i2,magic_j1:magic_j2]
        X_below_sum = np.sum(X_below)
        print('Sum: %.2f' % X_below_sum)
        plt.imshow(X_below)
        plt.show()
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

    imgs = np.load(image_path)
    labels = np.load(labels_path)
    assert imgs.shape[0] == labels.shape[0]
    print('imgs.shape')
    print(imgs.shape)
    print('labels.shape')
    print(labels.shape)

    num_cor = 0
    tot_dur = 0.
    for i,img in enumerate(imgs):
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


