import os,sys
import time
import tensorflow as tf
import numpy as np
proj_roots = [
    '/Users/joeliven/repos/carproject',
    '/scratch/cluster/joeliven/carproject',
    '/u/joeliven/repos/carproject',
    ]
for proj_root in proj_roots:
    if proj_root not in sys.path:
        if os.path.exists(proj_root):
            sys.path.append(proj_root)

# from lib.vgg16_encoder import get_encoder
# from lib.vgg10_encoder import get_encoder
from lib.vgg6t_encoder import get_encoder
# from lib.vgg6f_encoder import get_encoder
# from lib.vgg5_encoder import get_encoder
# from lib27.image_utils import preprocess_image
# from lib.image_utils import preprocess_image, deprocess_image
from lib27.image_utils import preprocess_image, deprocess_image
from skimage.io import imread
import matplotlib.pyplot as plt

idx2label = ['S', 'T']

class Classifier(object):
    def __init__(self, **kwargs):
        self.dim_img = kwargs.get('dim_img', 224)
        self.nb_channels = kwargs.get('nb_channels', 3)
        self.META_PATH = kwargs.get('meta_path')
        assert '.meta' in self.META_PATH
        self.RESTORE_PATH = kwargs.get('weights_path')

        with tf.Graph().as_default():
            self.inputs_pl = tf.placeholder(tf.float32, [None, self.dim_img, self.dim_img, self.nb_channels])
            self.preds, _ = get_encoder(inputs_pl=self.inputs_pl)
            print('after self.preds = get_encoder()')
            self.sess = tf.Session()
            print('after self.sess = tf.Session()')
            # restore the trained model:
            saver = tf.train.Saver()
            saver.restore(self.sess,self.RESTORE_PATH)
            print('after saver.restore')

            weights = dict()
            outputs = dict()

            for k,v in _.items():
                print('key %s' % k)
                weights[k] = v.get('params')
                outputs[k] = v.get('outputs')

            for i,(k,v) in enumerate(weights.items()):
                if not v: continue
                w = v.get('w')
                b = v.get('b')
                w_,b_ = self.sess.run([w,b])
                print i,w_.shape, np.abs(w_).sum()/np.prod(w_.shape), b_.shape, np.abs(b_).sum()/np.prod(b_.shape)
            for i,(k,v) in enumerate(outputs.items()):
                if not v: continue
                print('v')
                print(v)
            self.outputs = outputs

    def hacky_predict(self, X, magic_i1=50, magic_i2=100, magic_j1=50, magic_j2=224-50, magic_thresh_light=.985, magic_thresh_sum=20., verbose=False):
        # X = self.threshold(X, .9)
        # X = self.threshold(X, .95)
        # X = self.threshold(X, .975)
        X = self.threshold(X, magic_thresh_light)
        # X = self.threshold(X, .99)
        # X = self.threshold(X, .999)
        X_below = X[magic_i1:magic_i2,magic_j1:magic_j2]
        X_below_sum = np.sum(X_below)
        # if X_below_sum > magic_thresh_sum:
        #     print('STRAIGHT')
        # else:
        #     print('TURN!!!')
        print('Sum: % %.2f' % X_below_sum)
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

        X = np.asarray([X])

        feed_dict = {self.inputs_pl: X}
        # scores = self.sess.run([self.preds], feed_dict=feed_dict)[0]  # [0] since sess.run([preds]) returns a list of len 1 in this case
        scores, conv1_1, pool1, conv2_1, pool2, conv3_1, pool3, fc1, fc2, fc3 = self.sess.run([self.preds,
                                                                                               self.outputs['encoder-conv1_1'],
                                                                                               self.outputs['encoder-pool1'],
                                                                                               self.outputs['encoder-conv2_1'],
                                                                                               self.outputs['encoder-pool2'],
                                                                                               self.outputs['encoder-conv3_1'],
                                                                                               self.outputs['encoder-pool3'],
                                                                                               self.outputs['encoder-fc1'],
                                                                                               self.outputs['encoder-fc2'],
                                                                                               self.outputs['encoder-fc3'],
                                                                                               ], feed_dict=feed_dict)  # [0] since sess.run([preds]) returns a list of len 1 in this case
        scores = scores[0] # since we are only classifying one image at a time
        print('scores.shape: %s' % str(scores.shape))

        conv1_1 = conv1_1[0] # since we are only classifying one image at a time
        print('conv1_1.shape: %s' % str(conv1_1.shape))
        self.viz(8, 8, 'conv1_1', conv1_1)

        pool1 = pool1[0] # since we are only classifying one image at a time
        print('pool1.shape: %s' % str(pool1.shape))
        self.viz(8, 8, 'pool1', pool1)

        conv2_1 = conv2_1[0] # since we are only classifying one image at a time
        print('conv2_1.shape: %s' % str(conv2_1.shape))
        self.viz(8, 8, 'conv2_1', conv2_1[:,:,0:64])

        pool2 = pool2[0] # since we are only classifying one image at a time
        print('pool2.shape: %s' % str(pool2.shape))
        self.viz(8, 8, 'pool2', pool2[:,:,0:64])

        conv3_1 = conv3_1[0] # since we are only classifying one image at a tim e
        print('conv3_1.shape: %s' % str(conv3_1.shape))
        self.viz(8, 8, 'conv3_1', conv3_1[:,:,0:64])

        pool3 = pool3[0] # since we are only classifying one image at a time
        print('pool3.shape: %s' % str(pool3.shape))
        self.viz(8, 8, 'pool3', pool3[:,:,0:64])

        fc1 = fc1[0] # since we are only classifying one image at a time
        print('fc1.shape: %s' % str(fc1.shape))

        fc2 = fc2[0] # since we are only classifying one image at a time
        print('fc2.shape: %s' % str(fc2.shape))

        fc3 = fc3[0] # since we are only classifying one image at a time
        print('fc3.shape: %s' % str(fc3.shape))

        print('fc1.shape: %s' % str(fc1.shape))
        print('fc2.shape: %s' % str(fc2.shape))
        print('fc3.shape: %s' % str(fc3.shape))
        # plt.show()

        duration = time.time() - start_time
        if verbose:
            print('scores: %s' % str(scores))
            pred_idx = np.argmax(scores)
            pred_class = idx2label[pred_idx]
            print('prediction: %s (duration: %.3f)' % (pred_class,duration))
        return scores[1] > scores[0], duration # scores[1] is TURN, scores[0] is STRAIGHT

    def viz(self,nb_rows,nb_cols,layer_name,tensor):
        # plt.imshow(tensor[:,:,0], cmap='Greys_r')
        # plt.show()
        return
        fig, ax = plt.subplots(figsize=(20, 10), nrows=nb_rows, ncols=nb_cols)
        for row in range(nb_rows):
            for col in range(nb_cols):
                map_num = row*nb_cols + col
                featmap2d = tensor[:,:,map_num]
                ax[row, col].imshow(featmap2d, cmap='Greys_r')
                ax[row, col].text(0, 0, '%s: %d' % (layer_name, map_num), color='r', fontsize=15, fontweight='bold')
        plt.show()

    def threshold(self,X,thresh):
        # print('in threshold')
        # print(X.shape)
        X = np.where(X >= thresh, X, 0)
        # X = np.where(X[:,:,0] >= thresh and X[:,:,1] >= thresh and X[:,:,2] >= thresh, X, 0)
        # print('X[:,:,0].shape')
        # print(X[:,:,0].shape)
        X_mask = X[:,:,0] * X[:,:,1] * X[:,:,2]
        X_mask = np.asarray([X_mask,X_mask,X_mask]).transpose(1,2,0)
        # print('X_mask.shape')
        # print(X_mask.shape)
        X = X*X_mask
        # print('threshold:')
        # plt.imshow(X)
        # plt.show()
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

    model = Classifier(meta_path=meta_path, weights_path=weights_path)
    dir = 'data/raw/4s-cclkwse'
    paths = [os.path.join(dir,x) for x in os.listdir(dir) if x.endswith('.jpg')]
    paths = paths[0:200]
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


