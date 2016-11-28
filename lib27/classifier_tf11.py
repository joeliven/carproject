import sys, os
import time
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import numpy as np
# from skimage.io import imread
idx2label = ['S', 'T']

class Classifier(object):
    def __init__(self, **kwargs):
        self.META_PATH = kwargs.get('meta_path')
        assert '.meta' in self.META_PATH
        self.RESTORE_PATH = kwargs.get('weights_path')
        self.sess = tf.Session()
        # restore the trained model:
        saver = tf.train.import_meta_graph(self.META_PATH)
        saver.restore(self.sess, self.RESTORE_PATH)
        self.preds = tf.get_default_graph().get_operation_by_name("encoder-fc3").outputs[0]
        self.inputs_pl = tf.get_default_graph().get_operation_by_name("Placeholder").outputs[0]
        graphdef = tf.get_default_graph().as_graph_def()
        # tf.train.write_graph(graphdef, 'models/vgg16a', 'vgg16a_graph.bin', as_text=False)

        input_graph_path = 'models/vgg16a/vgg16a_graphdef'
        input_saver_def_path = ""
        input_binary = False
        output_node_names = "encoder-fc3"
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_graph_path = 'models/vgg16a/vgg16a_graph.froz'
        clear_devices = True
        print('about to call freeze_graph.freeze_graph...')
        freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                                  input_binary, self.RESTORE_PATH,
                                  output_node_names, restore_op_name,
                                  filename_tensor_name, output_graph_path,
                                  clear_devices, "")
        print('about to reload frozen graph...')

        # Now we make sure the variable is now a constant, and that the graph still
        # produces the expected result.
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            with open(output_graph_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")

        print('DONE...')
        sys.exit(1)

    # def _preprocess(self, X):
    #     """preprocess a single incoming image"""
    #     # TODO: convert my python3.4 code from lib/image_utils.py-->preprocess_image() into python2.7 compatible code
    #     return X

    def predict(self, X, verbose=False):
        """make a prediction on a single incoming image"""
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
    print('meta_path')
    print(meta_path)
    print('weights_path')
    print(weights_path)
    print('image_path')
    print(image_path)

    verbose = False
    if len(sys.argv) == 5:
        verbose = True
        print('verbose is True')
    else:
        print('verbose is False')

    img = np.load(image_path)[22]
    print('img.shape')
    print(img.shape)

    model = Classifier(meta_path=meta_path, weights_path=weights_path)
    model.predict(img, verbose=verbose)


