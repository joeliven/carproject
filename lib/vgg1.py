########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################
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

import time
import pdb
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

from collections import defaultdict
from lib.data_set import DataSet
from lib.image_utils import deprocess_image

I = 'inputs'
O = 'outputs'
P = 'params'

class VGG1(object):
    def __init__(self, **kwargs):
        self.name = 'dsae_st_1_testB'
        # unpack args:
        self.batch_size_int = kwargs.get('batch_size',32)
        self.dim_img_int = kwargs.get('dim_img', 224)
        self.nb_channels_int = kwargs.get('nb_channels', 3)
        # self.dim_fc1_int = kwargs.get('dim_fc1', 256)
        self.nb_classes = kwargs.get('nb_classes', 3) # number of target classes we are predicting
        self.encoder = defaultdict(dict) # each key maps to a dict with keys: inputs, outputs, params

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _encode(self, inputs_pl):
        # conv1_1
        prev_layer = inputs_pl
        layer_name = '%s-conv1_1' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            print(inputs_pl)
            print('tf.shape(inputs_pl)')
            print(tf.shape(inputs_pl))
            print('inputs_pl.get_shape().as_list()')
            print(inputs_pl.get_shape().as_list())
            self.encoder[layer_name][I] = prev_layer
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(inputs_pl, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 name='biases', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.encoder[layer_name][O] = tf.nn.relu(out, name=scope)
            self.encoder[layer_name][P] = {'w': kernel, 'b': biases}

        # conv1_2
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-conv1_2' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 name='biases', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.encoder[layer_name][O] = tf.nn.relu(out, name=scope)
            self.encoder[layer_name][P] = {'w': kernel, 'b': biases}

        # pool1
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-pool1' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            self.encoder[layer_name][O] = tf.nn.max_pool(self.encoder[layer_name][I],
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool1')
            self.encoder[layer_name][P] = None

        # conv2_1
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-conv2_1' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 name='biases', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.encoder[layer_name][O] = tf.nn.relu(out, name=scope)
            self.encoder[layer_name][P] = {'w': kernel, 'b': biases}

        prev_layer = self.encoder[layer_name]
        # conv2_2
        layer_name = '%s-conv2_2' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 name='biases', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.encoder[layer_name][O] = tf.nn.relu(out, name=scope)
            self.encoder[layer_name][P] = {'w': kernel, 'b': biases}

        # pool2
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-pool2' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            self.encoder[layer_name][O] = tf.nn.max_pool(self.encoder[layer_name][I],
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool2')
            self.encoder[layer_name][P] = None

        # conv3_1
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-conv3_1' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 name='biases', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.encoder[layer_name][O] = tf.nn.relu(out, name=scope)
            self.encoder[layer_name][P] = {'w': kernel, 'b': biases}

        # conv3_2
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-conv3_2' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 name='biases', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.encoder[layer_name][O] = tf.nn.relu(out, name=scope)
            self.encoder[layer_name][P] = {'w': kernel, 'b': biases}

        # conv3_3
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-conv3_3' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 name='biases', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.encoder[layer_name][O] = tf.nn.relu(out, name=scope)
            self.encoder[layer_name][P] = {'w': kernel, 'b': biases}
        # tf.shape should be: (None,56,56,256)

        # pool3
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-pool3' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            self.encoder[layer_name][O] = tf.nn.max_pool(self.encoder[layer_name][I],
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool3')
            self.encoder[layer_name][P] = None
        # tf.shape should be: (None,28,28,256)

        # conv4_1
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-conv4_1' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 name='biases', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.encoder[layer_name][O] = tf.nn.relu(out, name=scope)
            self.encoder[layer_name][P] = {'w': kernel, 'b': biases}
        # tf.shape should be: (None,28,28,512)

        # conv4_2
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-conv4_2' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 name='biases', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.encoder[layer_name][O] = tf.nn.relu(out, name=scope)
            self.encoder[layer_name][P] = {'w': kernel, 'b': biases}
        # tf.shape should be: (None,28,28,512)

        # conv4_3
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-conv4_3' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 name='biases', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.encoder[layer_name][O] = tf.nn.relu(out, name=scope)
            self.encoder[layer_name][P] = {'w': kernel, 'b': biases}
        # tf.shape should be: (None,28,28,512)

        # pool4
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-pool4' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            self.encoder[layer_name][O] = tf.nn.max_pool(self.encoder[layer_name][I],
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool4')
            self.encoder[layer_name][P] = None
        # tf.shape should be: (None,14,14,512)

        # conv5_1
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-conv5_1' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 name='biases', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.encoder[layer_name][O] = tf.nn.relu(out, name=scope)
            self.encoder[layer_name][P] = {'w': kernel, 'b': biases}
        # tf.shape should be: (None,14,14,512)

        # conv5_2
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-conv5_2' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 name='biases', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.encoder[layer_name][O] = tf.nn.relu(out, name=scope)
            self.encoder[layer_name][P] = {'w': kernel, 'b': biases}
        # tf.shape should be: (None,14,14,512)

        # conv5_3
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-conv5_3' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
            conv = tf.nn.conv2d(self.encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 name='biases', trainable=False)
            out = tf.nn.bias_add(conv, biases)
            self.encoder[layer_name][O] = tf.nn.relu(out, name=scope)
            self.encoder[layer_name][P] = {'w': kernel, 'b': biases}
        # tf.shape should be: (None,14,14,512)

        # pool5
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-pool5' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            self.encoder[layer_name][O] = tf.nn.max_pool(self.encoder[layer_name][I],
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool5')
            self.encoder[layer_name][P] = None
        # tf.shape should be: (None,7,7,512)

        # fc1
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-fc1' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            shape = int(np.prod(self.encoder[layer_name][I].get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable=False)
            fc1b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                                 name='biases', trainable=False)
            pool5_flat = tf.reshape(self.encoder[layer_name][I], [-1, shape])
            # tf.shape should be: (None,25088)
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.encoder[layer_name][O] = tf.nn.relu(fc1l, name=scope)
            self.encoder[layer_name][P] = {'w': fc1w, 'b': fc1b}
        # tf.shape should be: (None,4096)

        # fc2
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-fc2' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable=False)
            fc2b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                                 name='biases', trainable=False)
            fc2l = tf.nn.bias_add(tf.matmul(self.encoder[layer_name][I], fc2w), fc2b)
            self.encoder[layer_name][O] = tf.nn.relu(fc2l, name=scope)
            self.encoder[layer_name][P] = {'w': fc2w, 'b': fc2b}
        # tf.shape should be: (None,4096)

        # fc3
        prev_layer = self.encoder[layer_name]
        layer_name = '%s-fc3' % (ENCODER)
        with tf.name_scope(layer_name) as scope:
            self.encoder[layer_name][I] = prev_layer[O]
            fc3w = tf.Variable(tf.truncated_normal([4096, 3],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights', trainable=True)
            fc3b = tf.Variable(tf.constant(0.0, shape=[3], dtype=tf.float32),
                                 name='biases', trainable=True)
            fc3l = tf.nn.bias_add(tf.matmul(self.encoder[layer_name][I], fc3w), fc3b, name=scope)
            self.encoder[layer_name][O] = fc3l # no ReLU since this is the final (unormalized) prediction layer
            self.encoder[layer_name][P] = {'w': fc3w, 'b': fc3b}
        # tf.shape should be: (None,3)

        return self.encoder[layer_name][O] # --> self.encoder['encoder-fc3']['output']

    def _loss(self, predictions, targets):
        """Add L2Loss to all the trainable variables.
        Add summary for "Loss" and "Loss/avg".
        Args:
          predictions: Logits from inference().
          targets: Labels from distorted_inputs or inputs(). 1-D tensor
                  of shape [batch_size]
        Returns:
          Loss tensor of type float.
        """
        # Calculate the average cross entropy loss across the batch.
        print('tf.shape(targets)')
        print(tf.shape(targets))
        print(targets)
        print('tf.shape(predictions)')
        print(tf.shape(predictions))
        print(predictions)
        targets = tf.cast(targets, tf.int64)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            predictions, targets, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        tf.scalar_summary('xentropy_loss', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def _training(self, total_loss, lr, global_step): # keep this one
        """Train CIFAR-10 model.
        Create an optimizer and apply to all trainable variables. Add moving
        average for all trainable variables.
        Args:
          total_loss: Total loss from _loss().
          global_step: Integer Variable counting the number of training steps
            processed.
        Returns:
          train_op: op for training.
        """
        # get optimizer:
        optimizer = tf.train.AdamOptimizer(lr)
        # Compute gradients:
        grads = optimizer.compute_gradients(total_loss)
        # Apply gradients:
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)
        # Add histograms for gradients.
        print('len(grads)')
        print(len(grads))
        for grad, var in grads:
            print(var)
            print(grad)
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op

    def _evaluate(self, predictions, targets): # keep this one
        """Evaluate the quality of the logits at predicting the label.

        Args:
          predictions: predictions tensor, float - [batch_size, NUM_CLASSES].
          targets: targets tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).

        Returns:
          A scalar int32 tensor with the number of examples (out of batch_size)
          that were predicted correctly.
        """
        # For a classifier model, we can use the in_top_k Op.
        # It returns a bool tensor with shape [batch_size] that is true for
        # the examples where the label is in the top k (here k=1)
        # of all logits for that example.
        targets_sparse = tf.cast(tf.argmax(targets, dimension=1), dtype=tf.int32)
        correct = tf.nn.in_top_k(predictions, targets_sparse, 1)
        # Return the number of true entries.
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def train(self, **kwargs):
        """Train model for a number of steps."""
        # unpack args:
        data_train = kwargs.get('data_train', None)
        if data_train is None:
            raise ValueError(
                'data_train cannot be None. At least training data must be supplied in order to train the model.')
        data_val = kwargs.get('data_val', None)
        if data_val is None:
            print('Warning: no val data has been supplied.')
        batch_size_int = kwargs.get('batch_size', None)
        save_summaries_every = kwargs.get('save_summaries_every', 500)
        display_every = kwargs.get('display_every', 1)
        display = kwargs.get('display', False)
        nb_to_display = kwargs.get('nb_to_display', 5)
        nb_epochs = kwargs.get('nb_epochs', 100)
        save_best_only = kwargs.get('save_best_only', 'save_all')
        lr = kwargs.get('lr', 0.001)
        l2 = kwargs.get('l2', 0.0001)
        SAVE_PATH = kwargs.get('save_path', None)
        if SAVE_PATH is None:
            print('Warning: no SAVE_PATH has been specified.')
        weights = kwargs.get('weights', None)
        load_encoder = kwargs.get('load_encoder', True)

        # ensure batch_size is set appropriately:
        if batch_size_int is None:
            if self.batch_size_int is None:
                raise ValueError(
                    'batch_size must be specified either in model instantiation or passed into this training method,'
                    'but batch_size cannnot be None in both cases.')
            batch_size_int = self.batch_size_int

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            # Generate placeholders for the input images and the gold-standard labels
            inputs_pl = tf.placeholder(tf.float32, [None, self.dim_img_int, self.dim_img_int, self.nb_channels_int])
            targets_pl = tf.placeholder(tf.float32, [None, self.nb_classes])

            # Create a variable to track the global step.
            global_step = tf.Variable(0, name='global_step', trainable=False)

            # Build a Graph that computes predictions
            preds = self._encode(inputs_pl)

            # Add to the Graph the Ops for loss calculation.
            loss = self._loss(predictions=preds, targets=targets_pl)

            # Add to the Graph the Ops that calculate and apply gradients.
            train_op = self._training(total_loss=loss, lr=lr, global_step=global_step)

            # Add the Op to compute the avg cross-entropy loss for evaluation purposes.
            eval_correct = self._evaluate(predictions=preds, targets=targets_pl)

            # Build the summary Tensor based on the TF collection of Summaries.
            summary = tf.merge_all_summaries()

            # Add the variable initializer Op.
            init = tf.initialize_all_variables()

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # Instantiate a SummaryWriter to output summaries and the Graph.
            if SAVE_PATH is not None:
                summary_writer = tf.train.SummaryWriter(SAVE_PATH, sess.graph)
            else:
                print('WARNING: SAVE_PATH is not specified...cannot save model file')

            # And then after everything is built:
            # Run the Op to initialize the variables.
            sess.run(init)

            # load pretrained weights if desired:
            if weights is not None and sess is not None:
                self.load_weights(weights, sess, load_encoder=load_encoder)

            # Start the training loop: train for nb_epochs, where each epoch iterates over the entire training set once.
            history = [] # list for saving train_acc and val_acc upon evaluation after each epoch ends
            train_acc_best = 0.
            val_acc_best = 0.
            batch_tot = 0
            # START ALL EPOCHS
            for epoch_num in range(nb_epochs):
                # START ONE EPOCH
                print('starting epoch %d / %d' % (epoch_num+1, nb_epochs))
                nb_batches_per_epoch = (data_train.nb_samples // batch_size_int) + 1
                batch_num = 0
                end_of_epoch = False
                while not end_of_epoch:
                    # iterate over all the training data once, batch by batch:
                    batch_start_time = time.time()
                    batch_num += 1
                    batch_tot += 1
                    # Fill a feed dictionary with the actual set of images and labels
                    # for this particular training step.
                    next_batch = data_train.fill_feed_dict(inputs_pl, targets_pl, batch_size_int)

                    # Run one step of the model.  The return values are the activations
                    # from the `train_op` (which is discarded) and the `loss` Op.  To
                    # inspect the values of your Ops or variables, you may include them
                    # in the list passed to sess.run() and the value tensors will be
                    # returned in the tuple from the call.
                    _, loss_value, pred_vals = sess.run([train_op, loss, preds],
                                             feed_dict=next_batch)
                    batch_duration = time.time() - batch_start_time

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    # Print status to stdout.
                    print('\tbatch_num %d / %d : loss = %.2f (%.3f sec)' % (batch_num, nb_batches_per_epoch, loss_value, batch_duration))

                    # Write the summaries and print an overview fairly often.
                    # if step > 0 and step % 5 == 0: # 100
                    if batch_num % save_summaries_every == 0: # 100
                        # Print status to stdout.
                        # Update the events file.
                        summary_str = sess.run(summary, feed_dict=next_batch)
                        if SAVE_PATH is not None:
                            summary_writer.add_summary(summary_str, global_step=batch_tot)
                            summary_writer.flush()
                    end_of_epoch = data_train.end_of_epoch()
                    if end_of_epoch and epoch_num % display_every == 0:
                        print('pred_vals')
                        pred_vals = pred_vals[0:nb_to_display]
                        print(pred_vals.shape)
                        print(pred_vals)
                        cur_batch = data_train.get_cur_batch()
                        if cur_batch['X'].shape[0] < nb_to_display:
                            nb_to_display = cur_batch['X'].shape[0]
                        for samp_num in range(nb_to_display):
                            img = cur_batch['X'][samp_num]
                            class_true_idx = np.argmax(cur_batch['y'][samp_num])
                            class_true = idx2label[class_true_idx]
                            scores = pred_vals[samp_num]
                            probs = self.softmax(scores)
                            class_pred_idx = np.argmax(probs)
                            class_pred = idx2label[class_pred_idx]
                            txt = '\tpredicted class dist: %s\n' \
                                  '\tpredicted class: \t%s\n' \
                                  '\ttrue class: \t\t%s' % (str(probs), class_pred, class_true)
                            print(txt)
                            if display == True:
                                fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1)
                                ax.imshow(deprocess_image(img))
                                ax.text(0, 0, txt, color='r', fontsize=15, fontweight='bold')
                                plt.show()
                    # END ONE EPOCH

                # After epoch:
                #   (1) evaluate the model
                #   (2) save a checkpoint (possibly only if the val accuracy has improved)

                # (1a) Evaluate against the training set.
                print('Training Data Eval:')
                new_best_train = False
                train_acc = self.evaluate(sess, eval_correct, inputs_pl, targets_pl, data_train, batch_size_int, lim=100)
                if train_acc > train_acc_best:
                    train_acc_best = train_acc
                    new_best_train = True
                # (1b) Evaluate against the validation set if val data was provided.
                if data_val is not None:
                    print('Validation Data Eval:')
                    val_acc = self.evaluate(sess, eval_correct, inputs_pl, targets_pl, data_val, batch_size_int, lim=100)
                    new_best_val = False
                    if val_acc > val_acc_best:
                        val_acc_best = val_acc
                        new_best_val = True
                    history.append({'train_acc':train_acc, 'val_acc':val_acc})
                else:
                    history.append({'train_acc':train_acc})

                # (2) save checkpoint file
                if save_best_only == 'save_best_train':
                    if new_best_train:
                        checkpoint_file = os.path.join(SAVE_PATH, '%s_checkpoint' % self.name )
                        print('new_best_train_acc: %f \tsaving checkpoint to file: %s' % (train_acc_best, str(checkpoint_file)))
                        saver.save(sess, checkpoint_file, global_step=epoch_num)
                elif save_best_only == 'save_best_val' and data_val is not None:
                    if new_best_val:
                        checkpoint_file = os.path.join(SAVE_PATH, '%s_checkpoint' % self.name )
                        print('new_best_val_acc: %f \tsaving checkpoint to file: %s' % (val_acc_best, str(checkpoint_file)))
                        saver.save(sess, checkpoint_file, global_step=epoch_num)
                else:
                    checkpoint_file = os.path.join(SAVE_PATH, '%s_checkpoint' % self.name )
                    print('train_acc: %f \t val_acc: %f \tsaving checkpoint to file: %s' % (train_acc, val_acc, str(checkpoint_file)))
                    saver.save(sess, checkpoint_file, global_step=epoch_num)
            # END ALL EPOCHS
        return history, train_acc_best, val_acc_best

    def predict(self, **kwargs):
        """Use a trained model for prediction."""
        # unpack args:
        X = kwargs.get('X', None)
        batch_size_int = kwargs.get('batch_size', None)
        LOAD_PATH = kwargs.get('load_path', None)
        if LOAD_PATH is None:
            input('Error: no LOAD_PATH has been specified. Randomly initialized model should not be used for prediciton.'
                  '\nPress enter to continue anyways:')

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            # Generate placeholders for the input images and the gold-standard class labels
            inputs_pl = tf.placeholder(tf.float32, [None, self.dim_img_int, self.dim_img_int, self.nb_channels_int])

            # Build a Graph that computes encodings
            preds = self._encode(inputs_pl)

            # Create a saver for restoring the model from the latest/best training checkpoints.
            saver = tf.train.Saver()

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # Instantiate a SummaryWriter to output summaries and the Graph.
            if LOAD_PATH is not None:
                checkpoint_path = tf.train.latest_checkpoint(LOAD_PATH)
                saver.restore(sess, checkpoint_path)
                print('model restored from checkpoint file: %s' % str(checkpoint_path))
            else:
                print('No LOAD_PATH specified, so initializing the model with random weights')
                # Add the variable initializer Op.
                init = tf.initialize_all_variables()
                # Run the Op to initialize the variables.
                sess.run(init)

            # Do Prediction:
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = {inputs_pl: X}

            pred_vals = sess.run([preds], feed_dict=feed_dict)
            duration = time.time() - start_time

            # Print status to stdout.
            print('Prediction took: %f for batch_size of: %d  --> %f per example' % (duration, self.batch_size_int, (duration / self.batch_size_int)))

            print('pred_vals.shape')
            print(type(pred_vals))
            print(pred_vals.shape)
            print(pred_vals)
            print('...pred_vals...(during prediction)...')

            for samp_num in range(X.shape[0]):
                img = X[samp_num]
                scores = pred_vals[samp_num]
                probs = self.softmax(scores)
                class_pred_idx = np.argmax(probs)
                class_pred = idx2label[class_pred_idx]
                fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=2)
                ax[0].imshow(deprocess_image(img))
                txt = 'predicted class dist: %s\n' \
                      'predicted class: %s\n' \
                      % (str(probs), class_pred)
                ax[0].text(0, 0, txt, color='b', fontsize=15, fontweight='bold')
                plt.show()

    def evaluate(self, sess, eval_correct, inputs_pl, targets_pl, data_set, batch_size_int, lim=-1, reset=True):
        """Runs one evaluation against the full epoch (or first 'lim' samples) of data.

        Args:
          sess: The session in which the model has been trained.
          eval_correct: The Tensor that returns the number of correct predictions.
          inputs_pl: The images placeholder.
          targets_pl: The labels placeholder.
          data_set: The set of inputs and targets to evaluate (a DataSet object)
        """
        # And run one epoch of eval.
        true_count = 0  # Counts the number of correct predictions.
        nb_samples = data_set.nb_samples
        if lim > 0:
            if lim < nb_samples:
                nb_samples = lim
                print('only evaluating on %d of the samples in the dataset.' % lim)
        steps_per_epoch = nb_samples // batch_size_int
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        nb_samples = steps_per_epoch * batch_size_int
        for step in range(steps_per_epoch):
            feed_dict = data_set.fill_feed_dict(inputs_pl, targets_pl, batch_size_int)
            true_count += sess.run(eval_correct, feed_dict=feed_dict)
        precision = true_count / nb_samples
        print('  nb_samples: %d  Num correct: %d  Precision @ 1: %0.04f' %
              (nb_samples, true_count, precision))
        if reset:
            data_set.reset()
        return precision

    def load_weights_encoder(self, weights, sess):
        keys_map = {
            'conv1_1_W': 'encoder-conv1_1',
            'conv1_1_b': 'encoder-conv1_1',
            'conv1_2_W': 'encoder-conv1_2',
            'conv1_2_b': 'encoder-conv1_2',
            'conv2_1_W': 'encoder-conv2_1',
            'conv2_1_b': 'encoder-conv2_1',
            'conv2_2_W': 'encoder-conv2_2',
            'conv2_2_b': 'encoder-conv2_2',
            'conv3_1_W': 'encoder-conv3_1',
            'conv3_1_b': 'encoder-conv3_1',
            'conv3_2_W': 'encoder-conv3_2',
            'conv3_2_b': 'encoder-conv3_2',
            'conv3_3_W': 'encoder-conv3_3',
            'conv3_3_b': 'encoder-conv3_3',
            'conv4_1_W': 'encoder-conv4_1',
            'conv4_1_b': 'encoder-conv4_1',
            'conv4_2_W': 'encoder-conv4_2',
            'conv4_2_b': 'encoder-conv4_2',
            'conv4_3_W': 'encoder-conv4_3',
            'conv4_3_b': 'encoder-conv4_3',
            'conv5_1_W': 'encoder-conv5_1',
            'conv5_1_b': 'encoder-conv5_1',
            'conv5_2_W': 'encoder-conv5_2',
            'conv5_2_b': 'encoder-conv5_2',
            'conv5_3_W': 'encoder-conv5_3',
            'conv5_3_b': 'encoder-conv5_3',
            'fc6_W': 'encoder-fc1',
            'fc6_b': 'encoder-fc1',
            'fc7_W': 'encoder-fc2',
            'fc7_b': 'encoder-fc2',
            # 'fc8_W': 'encoder-fc3',
            # 'fc8_b': 'encoder-fc3',
        }

        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            encoder_key = keys_map.get(k, None)
            if encoder_key is None:
                continue
            if '_W' in k: # weights
                sess.run(self.encoder[encoder_key][P]['w'].assign(weights[k]))
            elif '_b' in k: # biases
                sess.run(self.encoder[encoder_key][P]['b'].assign(weights[k]))
            else:
                raise ValueError('unrecognized key in pretrained weights file: %s' % str(k))

    def load_weights(self, weight_file, sess, load_encoder=True):
        weights = np.load(weight_file)
        if load_encoder:
            self.load_weights_encoder(weights=weights, sess=sess)
        print('...weights loaded successfully...')

def prep_cmdln_parser():
    usage = "usage: %prog [options]"
    cmdln = argparse.ArgumentParser(usage)
    cmdln.add_argument("--batch-size", action="store", dest="BATCH_SIZE_INT", default=32, type=int,
                                help="batch-size to use during training [default: %default].")
    cmdln.add_argument("--nb-channels", action="store", dest="NB_CHANNELS_INT", default=3, type=int,
                                help="number of channels in the training data images [default: %default].")
    cmdln.add_argument("--dim-img", action="store", dest="DIM_IMG_INT", default=224, type=int,
                                help="dimensionality of the training data [default: %default].")
    cmdln.add_argument("--train-lim", action="store", dest="TRAIN_LIM", default=1000, type=int,
                                help="amount of training data to use [default: %default].")
    cmdln.add_argument("--val-lim", action="store", dest="VAL_LIM", default=20, type=int,
                                help="amount of validation data to use [default: %default].")
    cmdln.add_argument("--save-summaries-every", action="store", dest="SAVE_SUMMARIES_EVERY", default=100, type=int,
                                help="save summary data for displaying in tensorboard every how many batches [default: %default].")
    cmdln.add_argument("--display-every", action="store", dest="DISPLAY_EVERY", default=1, type=int,
                                help="print (and optionally display) predicted results from final batch of training "
                                     "every how many epochs (at end of epoch only) [default: %default].")
    cmdln.add_argument("--display", action="store_true", dest="DISPLAY", default=False,
                                help="display images when printing predicted results from subset of data at end of epoch [default: %default].")
    cmdln.add_argument("--nb-to-display", action="store", dest="NB_TO_DISPLAY", type=int, default=5,
                                help="how many images to display predicted results from at end of epoch [default: %default].")
    cmdln.add_argument("--nb-epochs", action="store", dest="NB_EPOCHS", type=int, default=100,
                                help="how many epochs to train the model for [default: %default].")
    cmdln.add_argument("--save-best-only", action="store", dest="SAVE_BEST_ONLY", type=int, default='save_all',
                                help="save all models at checkpoints or only save if the model has a new best "
                                     "training_acc or val_acc (specify which) [default: %default].")
    cmdln.add_argument("--load-path", action="store", dest="LOAD_PATH", type=int, default='',
                                help="path from which to load the pretrained weights for the model [default: %default].")

    cmdln.add_argument("--save-path", action="store", dest="SAVE_PATH", type=int, default='',
                                help="path at which to save the model checkpoints during training [default: %default].")
    return  cmdln


if __name__ == '__main__':
    label_map = {
        'O': [1, 0, 0],
        'BAD': [1, 0, 0],
        'C': [0, 1, 0],
        'U': [0, 0, 1],
    }
    idx2label = ['O', 'C', 'U']

    # Constants describing the training process.
    MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
    NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
    LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
    INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.
    ENCODER = 'encoder'

    # BATCH_SIZE_INT = 32
    # # BATCH_SIZE_INT = 1
    # NB_CHANNELS_INT = 3
    #
    # DIM_IMG_INT = 224
    # TRAIN_LIM = 1000
    # # TRAIN_LIM = 1
    # VAL_LIM = 20
    # # VAL_LIM = 1
    # SAVE_SUMMARIES_EVERY = 100
    # DISPLAY_EVERY = 1
    # DISPLAY = False
    # NB_TO_DISPLAY = 5
    # NB_EPOCHS = 100
    # SAVE_BEST_ONLY = 'save_all'  # 'save_best_train' or 'save_best_val'
    # # LOAD_PATH = 'models/vgg/vgg16_weights_pretrained.npz'
    # LOAD_PATH = '/scratch/cluster/joeliven/carproject/models/vgg/vgg16_weights_pretrained.npz'
    # # SAVE_PATH = 'models/vgg'
    # SAVE_PATH = '/scratch/cluster/joeliven/carproject/models/vgg'

    cmdln = prep_cmdln_parser()
    args = cmdln.parse_args()

    vgg = VGG1(batch_size=args.BATCH_SIZE_INT,
                    nb_channels=args.NB_CHANNELS_INT,
                    dim_img=args.DIM_IMG_INT)


    # X_file = 'data/preprocessed/gdc_3s/X_train.npy'
    X_file = '/scratch/cluster/joeliven/carproject/data/preprocessed/gdc_3s/X_train.npy'
    # y_file = 'data/preprocessed/gdc_3s/y_train.npy'
    y_file = '/scratch/cluster/joeliven/carproject/data/preprocessed/gdc_3s/y_train.npy'

    X = np.load(X_file)
    y = np.load(y_file)

    print('X.shape')
    print(X.shape)
    if X.shape[-1] != 3:
        X = np.transpose(X, axes=(0, 2, 3, 1))
    X_train = X[0:args.TRAIN_LIM]
    y_train = y[0:args.TRAIN_LIM]
    # X_train = X_train[18].reshape((-1,224,224,3))
    print('X_train.shape')
    print(X_train.shape)
    print('y_train.shape')
    print(y_train.shape)

    X_val = X[args.TRAIN_LIM: args.TRAIN_LIM + args.VAL_LIM]
    y_val = y[args.TRAIN_LIM: args.TRAIN_LIM + args.VAL_LIM]

    print('X_train.shape')
    print(X_train.shape)
    print('y_train.shape')
    print(y_train.shape)
    print('X_val.shape')
    print(X_val.shape)
    print('y_val.shape')
    print(y_val.shape)

    data_train_ = DataSet(X=X_train, y=y_train, batch_size=args.BATCH_SIZE_INT)
    data_val_ = DataSet(X=X_val, y=y_val, batch_size=args.BATCH_SIZE_INT)

    TRAIN = True
    # TRAIN = False

    if TRAIN:
        history, best_train_acc, best_val_acc = \
            vgg.train(data_train=data_train_, data_val=data_val_,
                  batch_size=args.BATCH_SIZE_INT,
                  save_path=args.SAVE_PATH,
                  weights=args.LOAD_PATH,
                  save_summaries_every=args.SAVE_SUMMARIES_EVERY,
                  display_every=args.args.DISPLAY_EVERY,
                  display=args.DISPLAY,
                  nb_to_display=args.NB_TO_DISPLAY,
                  nb_epochs=args.NB_EPOCHS,
                  save_best_only=args.SAVE_BEST_ONLY)

    else:
        vgg.predict(X=X[0:5],
                    batch_size=args.BATCH_SIZE_INT,
                    load_path=args.SAVE_PATH)