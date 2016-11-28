########################################################################################
# Adapted from Davi Frossard, 2016                                                                  #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
########################################################################################
import tensorflow as tf
import numpy as np
from collections import defaultdict

I = 'inputs'
O = 'outputs'
P = 'params'
ENCODER = 'encoder'

def get_encoder(**kwargs):
    # unpack args:
    inputs_pl = kwargs.get('inputs_pl')
    batch_size_int = kwargs.get('batch_size',32)
    dim_img_int = kwargs.get('dim_img', 224)
    nb_channels_int = kwargs.get('nb_channels', 3)
    nb_classes = kwargs.get('nb_classes', 2) # number of target classes we are predicting
    encoder = kwargs.get('encoder', defaultdict(dict))# each key maps to a dict with keys: inputs, outputs, params

    # conv1_1
    prev_layer = inputs_pl
    layer_name = '%s-conv1_1' % (ENCODER)
    with tf.name_scope(layer_name) as scope:
        print(inputs_pl)
        print('tf.shape(inputs_pl)')
        print(tf.shape(inputs_pl))
        print('inputs_pl.get_shape().as_list()')
        print(inputs_pl.get_shape().as_list())
        encoder[layer_name][I] = prev_layer
        kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=False)
        conv = tf.nn.conv2d(inputs_pl, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             name='biases', trainable=False)
        out = tf.nn.bias_add(conv, biases)
        encoder[layer_name][O] = tf.nn.relu(out, name=scope)
        encoder[layer_name][P] = {'w': kernel, 'b': biases}

    # conv1_2
    prev_layer = encoder[layer_name]
    layer_name = '%s-conv1_2' % (ENCODER)
    with tf.name_scope(layer_name) as scope:
        encoder[layer_name][I] = prev_layer[O]
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=False)
        conv = tf.nn.conv2d(encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             name='biases', trainable=False)
        out = tf.nn.bias_add(conv, biases)
        encoder[layer_name][O] = tf.nn.relu(out, name=scope)
        encoder[layer_name][P] = {'w': kernel, 'b': biases}

    # pool1
    prev_layer = encoder[layer_name]
    layer_name = '%s-pool1' % (ENCODER)
    with tf.name_scope(layer_name) as scope:
        encoder[layer_name][I] = prev_layer[O]
        encoder[layer_name][O] = tf.nn.max_pool(encoder[layer_name][I],
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')
        encoder[layer_name][P] = None

    # conv2_1
    prev_layer = encoder[layer_name]
    layer_name = '%s-conv2_1' % (ENCODER)
    with tf.name_scope(layer_name) as scope:
        encoder[layer_name][I] = prev_layer[O]
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=False)
        conv = tf.nn.conv2d(encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             name='biases', trainable=False)
        out = tf.nn.bias_add(conv, biases)
        encoder[layer_name][O] = tf.nn.relu(out, name=scope)
        encoder[layer_name][P] = {'w': kernel, 'b': biases}

    prev_layer = encoder[layer_name]
    # conv2_2
    layer_name = '%s-conv2_2' % (ENCODER)
    with tf.name_scope(layer_name) as scope:
        encoder[layer_name][I] = prev_layer[O]
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=False)
        conv = tf.nn.conv2d(encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             name='biases', trainable=False)
        out = tf.nn.bias_add(conv, biases)
        encoder[layer_name][O] = tf.nn.relu(out, name=scope)
        encoder[layer_name][P] = {'w': kernel, 'b': biases}

    # pool2
    prev_layer = encoder[layer_name]
    layer_name = '%s-pool2' % (ENCODER)
    with tf.name_scope(layer_name) as scope:
        encoder[layer_name][I] = prev_layer[O]
        encoder[layer_name][O] = tf.nn.max_pool(encoder[layer_name][I],
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')
        encoder[layer_name][P] = None

    # conv3_1
    prev_layer = encoder[layer_name]
    layer_name = '%s-conv3_1' % (ENCODER)
    with tf.name_scope(layer_name) as scope:
        encoder[layer_name][I] = prev_layer[O]
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=False)
        conv = tf.nn.conv2d(encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             name='biases', trainable=False)
        out = tf.nn.bias_add(conv, biases)
        encoder[layer_name][O] = tf.nn.relu(out, name=scope)
        encoder[layer_name][P] = {'w': kernel, 'b': biases}

    # conv3_2
    prev_layer = encoder[layer_name]
    layer_name = '%s-conv3_2' % (ENCODER)
    with tf.name_scope(layer_name) as scope:
        encoder[layer_name][I] = prev_layer[O]
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=False)
        conv = tf.nn.conv2d(encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             name='biases', trainable=False)
        out = tf.nn.bias_add(conv, biases)
        encoder[layer_name][O] = tf.nn.relu(out, name=scope)
        encoder[layer_name][P] = {'w': kernel, 'b': biases}

    # conv3_3
    prev_layer = encoder[layer_name]
    layer_name = '%s-conv3_3' % (ENCODER)
    with tf.name_scope(layer_name) as scope:
        encoder[layer_name][I] = prev_layer[O]
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights', trainable=False)
        conv = tf.nn.conv2d(encoder[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             name='biases', trainable=False)
        out = tf.nn.bias_add(conv, biases)
        encoder[layer_name][O] = tf.nn.relu(out, name=scope)
        encoder[layer_name][P] = {'w': kernel, 'b': biases}
    # tf.shape should be: (None,56,56,256)

    # pool3
    prev_layer = encoder[layer_name]
    layer_name = '%s-pool3' % (ENCODER)
    with tf.name_scope(layer_name) as scope:
        encoder[layer_name][I] = prev_layer[O]
        encoder[layer_name][O] = tf.nn.max_pool(encoder[layer_name][I],
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')
        encoder[layer_name][P] = None
    # tf.shape should be: (None,28,28,256)

    # fc1
    prev_layer = encoder[layer_name]
    layer_name = '%s-fc1' % (ENCODER)
    with tf.name_scope(layer_name) as scope:
        encoder[layer_name][I] = prev_layer[O]
        shape = int(np.prod(encoder[layer_name][I].get_shape()[1:]))
        fc1w = tf.Variable(tf.truncated_normal([shape, 100],
                                                     dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
        fc1b = tf.Variable(tf.constant(0.0, shape=[100], dtype=tf.float32),
                             name='biases', trainable=False)
        pool5_flat = tf.reshape(encoder[layer_name][I], [-1, shape])
        # tf.shape should be: (None,200704)
        fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
        encoder[layer_name][O] = tf.nn.relu(fc1l, name=scope)
        encoder[layer_name][P] = {'w': fc1w, 'b': fc1b}
    # tf.shape should be: (None,100)

    # fc2
    prev_layer = encoder[layer_name]
    layer_name = '%s-fc2' % (ENCODER)
    with tf.name_scope(layer_name) as scope:
        encoder[layer_name][I] = prev_layer[O]
        fc2w = tf.Variable(tf.truncated_normal([100, 100],
                                                     dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=False)
        fc2b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                             name='biases', trainable=False)
        fc2l = tf.nn.bias_add(tf.matmul(encoder[layer_name][I], fc2w), fc2b)
        encoder[layer_name][O] = tf.nn.relu(fc2l, name=scope)
        encoder[layer_name][P] = {'w': fc2w, 'b': fc2b}
    # tf.shape should be: (None,100)

    # fc3
    prev_layer = encoder[layer_name]
    layer_name = '%s-fc3' % (ENCODER)
    with tf.name_scope(layer_name) as scope:
        encoder[layer_name][I] = prev_layer[O]
        fc3w = tf.Variable(tf.truncated_normal([100, nb_classes],
                                                     dtype=tf.float32,
                                                     stddev=1e-1), name='weights', trainable=True)
        fc3b = tf.Variable(tf.constant(0.0, shape=[nb_classes], dtype=tf.float32),
                             name='biases', trainable=True)
        fc3l = tf.nn.bias_add(tf.matmul(encoder[layer_name][I], fc3w), fc3b, name=scope)
        encoder[layer_name][O] = fc3l # no ReLU since this is the final (unormalized) prediction layer
        encoder[layer_name][P] = {'w': fc3w, 'b': fc3b}
    # tf.shape should be: (None,self.nb_classes) --> (None,2) by default

    return encoder[layer_name][O], encoder # --> encoder['encoder-fc3']['output']
