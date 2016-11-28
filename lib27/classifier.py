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

from lib.vgg16_encoder import get_encoder

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
            print('self.preds')
            print(self.preds)
            self.sess = tf.Session()
            raw_input('self.preds')
            # restore the trained model:
            saver = tf.train.Saver()
            saver.restore(self.sess,self.RESTORE_PATH)
            raw_input('can the saver restore...')



        # output_graph_path = 'models/vgg16a/vgg16a_graph.bin.froz'
        # with tf.Graph().as_default():
        #     output_graph_def = tf.GraphDef()
        #     with open(output_graph_path, "rb") as f:
        #         output_graph_def.ParseFromString(f.read())
        #         _ = tf.import_graph_def(output_graph_def, name="")
        #     self.sess = tf.Session()
        #     ops = tf.get_default_graph().get_operations()
        #     for i,op in enumerate(ops):
        #         print('%d:\t%s' % (i, op.name))
        #
        #     self.preds = tf.get_default_graph().get_operation_by_name("encoder-fc3").outputs[0]
        #     self.inputs_pl = tf.get_default_graph().get_operation_by_name("Placeholder").outputs[0]

            # print(self.preds)
            # raw_input('self.preds')
            # print(self.inputs_pl)
            # raw_input('self.inputs_pl')
            # output = self.sess.run(self.preds)
            # raw_input('just rant sess.run')

        # # saver = tf.train.import_meta_graph(self.META_PATH)
        # # saver = tf.import_graph_def(self.META_PATH)
        # graph_def = tf.get_default_graph().as_graph_def()
        # with open(self.META_PATH, 'rb') as f:
        #     graph_def.ParseFromString(f.read())
        #     graph_def_imp = tf.import_graph_def(graph_def)
        # print('graph_def')
        # print(graph_def)
        # print('graph_def_imp')
        # print(graph_def_imp)
        # input('...about to fail bc of saver ...')
        # saver.restore(self.sess, self.RESTORE_PATH)
        # self.preds = tf.get_default_graph().get_operation_by_name("encoder-fc3").outputs[0]
        # self.inputs_pl = tf.get_default_graph().get_operation_by_name("Placeholder").outputs[0]

    # def _preprocess(self, X):
    #     """preprocess a single incoming image"""
    #     # TODO: convert my python3.4 code from lib/image_utils.py-->preprocess_image() into python2.7 compatible code
    #     return X

    def predict(self, X, verbose=False):
        """make a prediction on a single incoming image"""
        print('in predict()')
        start_time = time.time()
        X = np.asarray([X])
        feed_dict = {self.inputs_pl: X}
        scores = self.sess.run([self.preds], feed_dict=feed_dict)[0]  # [0] since sess.run([preds]) returns a list of len 1 in this case
        scores = scores[0] # since we are only classifying one image at a time
        duration = time.time() - start_time
        if verbose:
            print('scores: %s' % str(scores))
            pred_idx = np.argmax(scores)
            pred_class = idx2label[pred_idx]
            print('prediction: %s (duration: %.3f)' % (pred_class,duration))
        return scores[1] > scores[0] # scores[1] is TURN, scores[0] is STRAIGHT


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

    imgs = np.load(image_path)
    labels = np.load(labels_path)
    assert imgs.shape[0] == labels.shape[0]
    print('imgs.shape')
    print(imgs.shape)
    print('labels.shape')
    print(labels.shape)

    num_cor = 0
    for i,img in enumerate(imgs):
        turn = model.predict(img, verbose=verbose)
        label = labels[i]
        assert label.shape == (2,)
        if turn == True and np.argmax(label) == 1:
            num_cor += 1
            print('correct')
        elif turn == False and np.argmax(label) == 0:
            num_cor += 1
            print('correct')
        else:
            print('wrong')
    acc = num_cor / imgs.shape[0]
    print('acc: %.f' % acc)
    print('DONE')


