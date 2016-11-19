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
    '/Users/joeliven/repos/object-tracking',
    '/scratch/cluster/joeliven/object-tracking',
    '/u/joeliven/repos/object-tracking',
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

from collections import defaultdict
from DSAE.lib_tf.data_set import DataSet
from DSAE.lib.image_utils import deprocess_image
from DSAE.lib_tf.imagenet_classes import class_names
I = 'inputs'
O = 'outputs'
P = 'params'

class DSAE_ST_1(object):
    def __init__(self, **kwargs):
        self.name = 'dsae_st_1_testB'
        # unpack args:
        self.batch_size_int = kwargs.get('batch_size',32)
        self.batch_size_strict = kwargs.get('batch_size_strict', False)
        self.nb_channels_int = kwargs.get('nb_channels', 3)
        self.dim_img_int = kwargs.get('dim_img', 224)
        self.dim_recon_int = kwargs.get('dim_recon', 224)
        self.dim_patch_int = kwargs.get('dim_patch', 11)
        self.dim_conv_reducer_int = kwargs.get('dim_conv_reducer', 64)
        self.dim_fc1_int = kwargs.get('dim_fc1', 256)
        self.dim_descriptors_int = kwargs.get('dim_descriptors', 4)
        self.nb_feats_int = kwargs.get('nb_feats', 1)
        # self.params = {} # each key maps to a weight matrix (stored in a tf.Variable)
        self.encoder = defaultdict(dict) # each key maps to a dict with keys: inputs, outputs, params
        self.attention = defaultdict(dict) # each key maps to a layer  (i.e. a tf.Tensor)
        self.decoder = [] # each element in the decoder will be a dict mapping keys to layers  (i.e. tf.Tensors)


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
                                 trainable=False, name='biases')
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
                                 trainable=False, name='biases')
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
                                 trainable=False, name='biases')
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
                                 trainable=False, name='biases')
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
                                 trainable=False, name='biases')
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
                                 trainable=False, name='biases')
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
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.encoder[layer_name][O] = tf.nn.relu(out, name=scope)
            self.encoder[layer_name][P] = {'w': kernel, 'b': biases}
        # tf.shape should be: (None,56,56,256)

        return self.encoder[layer_name][O] # --> self.encoder['encoder-conv3_3']['output']

    def _attention(self, encoded):
        # encoded is a tensor resulting from the output of the conv layers ('conv3_3' under current settings)
        # tf.shape(encoded) should bed: (None,56,56,256)

        # conv_reducer
        prev_layer = encoded
        layer_name = '%s-conv_reducer' % (ATTENTION)
        with tf.name_scope(layer_name) as scope:
            self.attention[layer_name][I] = prev_layer
            kernel = tf.Variable(tf.truncated_normal([1, 1, 256, self.dim_conv_reducer_int], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.attention[layer_name][I], kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[self.dim_conv_reducer_int], dtype=tf.float32),
                                 trainable= True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.attention[layer_name][O] = tf.nn.relu(out, name=scope)
            self.attention[layer_name][P] = {'w': kernel, 'b': biases}
        # shape should be: (None,56,56,dim_conv_reducer)

        # flatten
        # reshape the tensor from shape: (None,56,56,dim_conv_reducer) --> to shape: (None,3136*dim_conv_reducer)
        prev_layer = self.attention[layer_name]
        layer_name = '%s-flatten1' % (ATTENTION)
        with tf.name_scope(layer_name) as scope:
            self.attention[layer_name][I] = prev_layer[O]
            conv_shape = tf.shape(self.attention[layer_name][I])
            dim_flattened = tf.reduce_prod(conv_shape[1:])
            print('in flatten1: conv_shape')
            print(conv_shape)
            print('dim_flattened')
            print(dim_flattened)
            print('in flatten1: self.encoder[layer_name][I]')
            print(self.attention[layer_name][I])
            self.attention[layer_name][O] = tf.reshape(self.attention[layer_name][I], shape=[conv_shape[0],dim_flattened])
            self.attention[layer_name][P] = None
            # shape should be: (None,3136*dim_conv_reducer)

        # fc_relu
        prev_layer = self.attention[layer_name]
        layer_name = '%s-fc_relu' % (ATTENTION)
        with tf.name_scope(layer_name) as scope:
            self.attention[layer_name][I] = prev_layer[O]
            # w_shape = [tf.shape(self.attention[layer_name][I])[1], self.dim_fc1_int]
            w_shape = [56*56*self.dim_conv_reducer_int, self.dim_fc1_int]
            print('here...w_shape is')
            print(w_shape)
            # input('...sdf...')
            fc_relu_w = tf.Variable(tf.truncated_normal(w_shape,
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc_relu_b = tf.Variable(tf.constant(0.0, shape=[self.dim_fc1_int], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc_relu = tf.nn.bias_add(tf.matmul(self.attention[layer_name][I], fc_relu_w), fc_relu_b)
            self.attention[layer_name][O] = tf.nn.relu(fc_relu)
            self.attention[layer_name][P] = {'w': fc_relu_w, 'b': fc_relu_b}
            # shape should be: (None,dim_fc1_int)


        # fc_relu2
        prev_layer = self.attention[layer_name]
        layer_name = '%s-fc_relu2' % (ATTENTION)
        with tf.name_scope(layer_name) as scope:
            self.attention[layer_name][I] = prev_layer[O]
            dim_attention_flat = self.nb_feats_int * (2 + self.dim_descriptors_int)  # (i,j) coords for each transformer plus a descriptor vector of dimension dim_descriptors
            # w_shape = [tf.shape(self.attention[layer_name][I])[1], dim_attention_flat]
            w_shape = [self.dim_fc1_int, self.dim_fc1_int]
            fc_relu2_w = tf.Variable(tf.truncated_normal(w_shape,
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc_relu2_b = tf.Variable(tf.constant(0.0, shape=[self.dim_fc1_int], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc_relu2 = tf.nn.bias_add(tf.matmul(self.attention[layer_name][I], fc_relu2_w), fc_relu2_b)
            # TODO: do I really want this to be a relu?? Probably not...maybe a sigmoid or tanh or maybe no nonlinearity for this one???
            self.attention[layer_name][O] = tf.nn.relu(fc_relu2)
            self.attention[layer_name][P] = {'w': fc_relu2_w, 'b': fc_relu2_b}
            # input('...aaa...')

        # fc_linear
        prev_layer = self.attention[layer_name]
        layer_name = '%s-fc_linear' % (ATTENTION)
        with tf.name_scope(layer_name) as scope:
            self.attention[layer_name][I] = prev_layer[O]
            dim_attention_flat = self.nb_feats_int * (2 + self.dim_descriptors_int)  # (i,j) coords for each transformer plus a descriptor vector of dimension dim_descriptors
            w_shape = [self.dim_fc1_int, dim_attention_flat]
            fc_linear_w = tf.Variable(tf.truncated_normal(w_shape,
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc_linear_b = tf.Variable(tf.constant(0.0, shape=[dim_attention_flat], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.attention[layer_name][O] = tf.nn.bias_add(tf.matmul(self.attention[layer_name][I], fc_linear_w), fc_linear_b)
            self.attention[layer_name][P] = {'w': fc_linear_w, 'b': fc_linear_b}
            # input('...aaa...')

        # attention (this is really just a reshape layer)
        prev_layer = self.attention[layer_name]
        layer_name = '%s-attention' % (ATTENTION)
        with tf.name_scope(layer_name) as scope:
            self.attention[layer_name][I] = prev_layer[O]

            unflattend_shape = [-1, self.nb_feats_int, 2+self.dim_descriptors_int]
            print('in attention layer, unflattend_shape is:')
            print(unflattend_shape)
            self.attention[layer_name][O] = tf.reshape(self.attention[layer_name][I], shape=unflattend_shape)
            self.attention[layer_name][P] = None
            # shape should be: (None, nb_feats_int, 2+dim_descriptors_int)

        assert self.nb_feats_int == self.attention[layer_name][O].get_shape()[1]
        assert 2 + self.dim_descriptors_int == self.attention[layer_name][O].get_shape()[2]

        return self.attention[layer_name][O] # --> self.attention['attention-attention']['output']

    def __deconv_one(self, descriptor, feat_n):
        """
        feat_n: int
            the spatial feature number corresponding to this patch decoder
        Input: descriptor
            tf Tensor: shape=(batch_size, dim_descriptors) default is (batch_size, 2+4) = (batch_size, 6)
        Output:
            tf Tensor: shape=(batch_size, dim_patch, dim_patch, nb_channels) default is (batch_size,11,11,3)
        :param kwargs:
        :return:
        """
        print('len(self.decoder)')
        print(len(self.decoder))
        print('feat_n')
        print(feat_n)
        assert len(self.decoder) == feat_n
        print('descriptor.get_shape().as_list()')
        print(descriptor.get_shape().as_list())
        print('descriptor')
        print(descriptor)
        # add another deconv_decoder to our list of decoders (one for each spatial feature)
        self.decoder.append(defaultdict(dict))

        # decoder-fc1
        prev_layer = descriptor
        # shape should be: (batch_size, dim_descriptors)
        layer_name = '%s-%d-fc1' % (DECODER,feat_n)
        with tf.name_scope(layer_name) as scope:
            self.decoder[feat_n][layer_name][I] = prev_layer
            w_shape = [self.dim_descriptors_int,3*3*64]
            fc1w = tf.Variable(tf.truncated_normal(w_shape,
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(0.0, shape=[3*3*64], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc1l = tf.nn.bias_add(tf.matmul(self.decoder[feat_n][layer_name][I], fc1w), fc1b)
            self.decoder[feat_n][layer_name][O] = tf.nn.relu(fc1l)
            self.decoder[feat_n][layer_name][P] = {'w': fc1w, 'b': fc1b}
            # shape should be: (batch_size, 3*3*64) --> (batch_size, 576)

        # decoder-fc2
        prev_layer = self.decoder[feat_n][layer_name]
        layer_name = '%s-%d-fc2' % (DECODER,feat_n)
        with tf.name_scope(layer_name) as scope:
            self.decoder[feat_n][layer_name][I] = prev_layer[O]
            w_shape = [3*3*64,3*3*64] # --> [576,576]
            fc2w = tf.Variable(tf.truncated_normal(w_shape,
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(0.0, shape=[3*3*64], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.decoder[feat_n][layer_name][I], fc2w), fc2b)
            self.decoder[feat_n][layer_name][O] = tf.nn.relu(fc2l)
            self.decoder[feat_n][layer_name][P] = {'w': fc2w, 'b': fc2b}
            # shape should be: (batch_size, 3*3*64) --> (batch_size, 576)

        # decoder-deconv0 --> reshape the above fc layer into a multi-channel conv-shaped layer appropriate for deconv-ing
        prev_layer = self.decoder[feat_n][layer_name]
        layer_name = '%s-%d-deconv0' % (DECODER,feat_n)
        with tf.name_scope(layer_name) as scope:
            self.decoder[feat_n][layer_name][I] = prev_layer[O]
            # conv_shape = [self.batch_size_int,3,3,64]
            conv_shape = [-1,3,3,64]
            self.decoder[feat_n][layer_name][O] = tf.reshape(self.decoder[feat_n][layer_name][I], shape=conv_shape)
            self.decoder[feat_n][layer_name][P] = None
            # shape should be: (batch_size, 3, 3, 64)

        # deconv1
        prev_layer = self.decoder[feat_n][layer_name]
        layer_name = '%s-%d-deconv1' % (DECODER,feat_n)
        with tf.name_scope(layer_name) as scope:
            self.decoder[feat_n][layer_name][I] = prev_layer[O]
            #                                         H  W Cout Cin
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            deconv = tf.nn.conv2d_transpose(value=self.decoder[feat_n][layer_name][I], filter=kernel,
                                            output_shape=[self.batch_size_int,5,5,64],strides=[1,1,1,1], padding='VALID')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(deconv, biases)
            self.decoder[feat_n][layer_name][O]= tf.nn.relu(out, name=scope)
            self.decoder[feat_n][layer_name][P] = {'w': kernel, 'b': biases}
            # shape should now be: (batch_size, 5, 5, 64)

        # deconv2
        prev_layer = self.decoder[feat_n][layer_name]
        layer_name = '%s-%d-deconv2' % (DECODER,feat_n)
        with tf.name_scope(layer_name) as scope:
            self.decoder[feat_n][layer_name][I] = prev_layer[O]
            #                                         H  W Cout Cin
            kernel = tf.Variable(tf.truncated_normal([3, 3, 32, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            deconv = tf.nn.conv2d_transpose(value=self.decoder[feat_n][layer_name][I], filter=kernel,
                                            output_shape=[self.batch_size_int,7,7,32],strides=[1,1,1,1], padding='VALID')
            biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(deconv, biases)
            self.decoder[feat_n][layer_name][O]= tf.nn.relu(out, name=scope)
            self.decoder[feat_n][layer_name][P] = {'w': kernel, 'b': biases}
            # shape should now be: (batch_size, 7, 7, 32)

        # deconv3
        prev_layer = self.decoder[feat_n][layer_name]
        layer_name = '%s-%d-deconv3' % (DECODER,feat_n)
        with tf.name_scope(layer_name) as scope:
            self.decoder[feat_n][layer_name][I] = prev_layer[O]
            #                                         H  W Cout Cin
            kernel = tf.Variable(tf.truncated_normal([3, 3, 16, 32], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            deconv = tf.nn.conv2d_transpose(value=self.decoder[feat_n][layer_name][I], filter=kernel,
                                            output_shape=[self.batch_size_int,9,9,16],strides=[1,1,1,1], padding='VALID')
            biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(deconv, biases)
            self.decoder[feat_n][layer_name][O]= tf.nn.relu(out, name=scope)
            self.decoder[feat_n][layer_name][P] = {'w': kernel, 'b': biases}
            # shape should now be: (batch_size, 9, 9, 16)

        # deconv4
        prev_layer = self.decoder[feat_n][layer_name]
        layer_name = '%s-%d-deconv4' % (DECODER,feat_n)
        with tf.name_scope(layer_name) as scope:
            self.decoder[feat_n][layer_name][I] = prev_layer[O]
            #                                         H  W Cout Cin
            kernel = tf.Variable(tf.truncated_normal([3, 3, 8, 16], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            deconv = tf.nn.conv2d_transpose(value=self.decoder[feat_n][layer_name][I], filter=kernel,
                                            output_shape=[self.batch_size_int,11,11,8],strides=[1,1,1,1], padding='VALID')
            biases = tf.Variable(tf.constant(0.0, shape=[8], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(deconv, biases)
            self.decoder[feat_n][layer_name][O]= tf.nn.relu(out, name=scope)
            self.decoder[feat_n][layer_name][P] = {'w': kernel, 'b': biases}
            # shape should now be: (batch_size, 11, 11, 8)

        # deconv5
        prev_layer = self.decoder[feat_n][layer_name]
        layer_name = '%s-%d-deconv5' % (DECODER,feat_n)
        with tf.name_scope(layer_name) as scope:
            self.decoder[feat_n][layer_name][I] = prev_layer[O]
            #                                         H  W Cout Cin
            kernel = tf.Variable(tf.truncated_normal([1, 1, 3, 8], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            deconv = tf.nn.conv2d_transpose(value=self.decoder[feat_n][layer_name][I], filter=kernel,
                                            output_shape=[self.batch_size_int,11,11,3],strides=[1,1,1,1], padding='VALID')
            biases = tf.Variable(tf.constant(0.0, shape=[3], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(deconv, biases)
            self.decoder[feat_n][layer_name][O]= tf.nn.relu(out, name=scope)
            self.decoder[feat_n][layer_name][P] = {'w': kernel, 'b': biases}
            # shape should now be: (batch_size, 11, 11, 3)

        return self.decoder[feat_n][layer_name][O] # --> self.[feat_n]['decoderX-deconv5']['output']

    def __deconv_all(self, attention):
        dim_patch_flat_int = (self.dim_patch_int ** 2) * self.nb_channels_int # e.g. 11x11x3
        ijs = attention[:,:,0:2] # shape should be (batch_size, nb_feats, 2)
        print('ijs')
        print(ijs)
        descriptors = attention[:,:,2:] # shape should be (batch_size, nb_feats, dim_descriptors)
        print('descriptors')
        print(descriptors)
        patches_l = list()

        for feat_n in range(self.nb_feats_int):
            print('feat_n:%d' % feat_n)
            descriptor = descriptors[:,feat_n,:]
            print('type(descriptor)')
            print(type(descriptor))
            patch_T = self.__deconv_one(descriptor=descriptor, feat_n=feat_n)

            print('type(patch_T)')
            print(type(patch_T))
            print('patch_T.get_shape()')
            print(patch_T.get_shape().as_list())
            print('tf.shape(patch_T)')
            print(tf.shape(patch_T))

            patches_l.append(tf.reshape(patch_T, shape=(self.batch_size_int,dim_patch_flat_int)))

        patches_T = tf.pack(patches_l, axis=1)
        print('type(patches_T)')
        print(type(patches_T))
        print('tf.shape(patches_T)')
        print(tf.shape(patches_T))
        print('patches_T.get_shape()')
        print(patches_T.get_shape())
        print('patches_T')
        print(patches_T)
        ijs_patches = tf.concat(2, [ijs,patches_T])
        print('type(ijs_patches)')
        print(type(ijs_patches))
        print('tf.shape(ijs_patches)')
        print(tf.shape(ijs_patches))
        print('ijs_patches.get_shape()')
        print(ijs_patches.get_shape())
        # ijs_patches shape should be: (batch_size, nb_feats, 2+dim_patch_flat)
        return ijs_patches

#####################################################################################################################
#####################################################################################################################
    def __interpolate(self, ijs_patches, nb_channels_int):
        """
        Interpolates patches from a a tensor with shape (batch_size, nb_feats, (dim_patch_recon**2)*3) to a new tensor
        with shape (batch_size, nb_feats, dim_patch_recon+1, dim_patch_recon+1, 3) using bilinear interpolation to
        transform each reconstructed patch.
        For example, if dim_path_recon = 11, then this function transforms each 11,11 reconstructed patch
        (in each color channel) into a 12,12 patch.
        Note, the bilinear interpolation is performed independently in each color channel.
        :param ijs_patches:
        :return:
        """
        with tf.name_scope('interpolate') as scope:

            # ijs_patches shape: (batch_size, nb_feats, 2+(dim_patch_recon**2)*nb_channels)
            # Note: the first 2 entries of last dim are i,j attention locations

            # grab some shape info right away to make code cleaner down below
            ijs_patches_shape = tf.shape(ijs_patches)
            batch_size = ijs_patches_shape[0]
            nb_feats = ijs_patches_shape[1]

            # flatten first two dimensions (batch_size, nb_feats) into one dim of (batch_size*nb_feats,)
            ijs_patches = tf.reshape(ijs_patches, shape=(batch_size * nb_feats, -1))
            # ijs_patches shape: (batch_size*nb_feats, (dim_patch_recon**2)*nb_channels_int)

            iexact = ijs_patches[:, 0]  # shape should be (batch_size*nb_feats, 1)
            jexact = ijs_patches[:, 1]  # shape should be (batch_size*nb_feats, 1)
            patches = ijs_patches[:, 2:]  # shape should be (batch_size*nb_feats, (dim_patch_recon**2)*nb_channels)
            patches_shape = tf.shape(patches)
            dim_patch_flat = patches_shape[1]
            dim_patch = dim_patch_flat / nb_channels_int
            dim_patch = tf.sqrt(dim_patch)
            dim_patch = tf.to_int32(dim_patch)
            patches = tf.reshape(patches,
                                 shape=(batch_size * nb_feats, dim_patch, dim_patch, nb_channels_int))
            # patches shape: (batch_size*nb_feats, dim_patch_recon, dim_patch_recon, nb_channels_int)
            #                       0,                      1,              2,              3
            patches = tf.transpose(patches, perm=(0, 3, 1, 2))  # move the channels dim before the rows,cols dims
            pad_l = [
                [0, 0],  # batch_size*nb_feats dim: no pre or post padding
                [0, 0],  # channels dim: no pre or post padding
                [1, 1],  # rows (i) dim: 0 prepad and 0 postpad
                [1, 1],  # cols (j) dim: 0 prepad and 0 postpad
            ]
            paddings = tf.constant(pad_l)
            padded = tf.pad(patches, paddings, mode='CONSTANT')
            # padded shape: (batch_size*nb_feats, nb_channels, dim_patch_recon, dim_patch_recon)
            #                       0,                1,              2,              3
            # move the batch_size*nb_feats dim to the end to allow broadcasting for the interpolation computation below
            padded = tf.transpose(padded, perm=(1, 2, 3, 0))
            # padded shape: (nb_channels_int, dim_patch_recon, dim_patch_recon, batch_size*nb_feats)
            #                       0,                1,              2,              3
            P00 = padded[:, 0:-1, 0:-1, :]
            P10 = padded[:, 1:, 0:-1, :]
            P01 = padded[:, 0:-1, 1:, :]
            P11 = padded[:, 1:, 1:, :]
            # return P00
            _1 = tf.constant(1., dtype='float32')
            ibar = tf.floor(iexact)
            jbar = tf.floor(jexact)
            i = iexact - ibar
            j = jexact - jbar
            # do actual interpolation
            interp = i * j * P00 + (_1 - i) * j * P10 + i * (_1 - j) * P01 + (_1 - i) * (_1 - j) * P11
            # interp shape:   (nb_channels, dim_patch+1, dim_patch+1, batch_size*nb_feats)
            #                       0,                1,              2,              3

            # restore the channels dim to the last dimension and batch_size*nb_feats dim to the first dimension
            interp_transposed = tf.transpose(interp, perm=(3, 1, 2, 0))
            # interp shape:   (batch_size*nb_feats, dim_patch+1, dim_patch+1, nb_channels)
            #                          0,                1,              2,              3

            # separate dim0 from batch_size*nb_feats back to batch_size,nb_feats
            # and then flatten out the final dimesion over row,cols,channels
            interp_flat = tf.reshape(interp_transposed, shape=(batch_size, nb_feats, -1))
            print('interp_flat')
            print(interp_flat)
            # input('...')
            # interp shape:   (batch_size, nb_feats, ((dim_patch+1)**2)*nb_channels)
            #                       0,        1,             2
            iexact = tf.reshape(iexact, shape=(batch_size, nb_feats,-1))
            jexact = tf.reshape(jexact, shape=(batch_size, nb_feats, -1))
            # tf.gradients()
        return interp_flat, iexact, jexact

    def __prep_paste(self, patches, iexact, jexact, dim_recon_int, nb_channels_int):
        with tf.name_scope('prep_paste') as scope:
            # patches shape: (batch_size, nb_feats, ((dim_patch+1)**2)*nb_channels)
            # constants:
            dim_recon = tf.constant(dim_recon_int, dtype='int32')
            patches_shape = tf.shape(patches)
            batch_size = patches_shape[0]
            nb_feats = patches_shape[1]
            dim_patch_interp_flat = patches_shape[2]
            dim_patch_interp = tf.cast( tf.sqrt(dim_patch_interp_flat / nb_channels_int), tf.float32)
            # flatten batch_size, nb_feats into one dimension
            print('batch_size')
            print(batch_size)
            print('nb_feats')
            print(nb_feats)
            print('dim_patch_interp_flat')
            print(dim_patch_interp_flat)
            iexact = tf.reshape(iexact, shape=(batch_size * nb_feats,))
            jexact = tf.reshape(jexact, shape=(batch_size * nb_feats,))
            print('in prep_paste: patches')
            print(patches)
            patches = tf.reshape(patches, shape=(batch_size * nb_feats, dim_patch_interp_flat))
            print('in prep_paste: patches now')
            print(patches)

            # if i or j are out of bounds in either direction ( i or j < -dim_patch+1 or i or j > 223),
            # then set all of the reconstructed patch to zero
            print('iexact')
            print(iexact)
            print('dim_patch_interp')
            print(dim_patch_interp)
            # i_ob = tf.logical_or(tf.less(iexact, tf.cast(-dim_patch_interp,'float32')), tf.greater(iexact, 223))
            # j_ob = tf.logical_or(tf.less(jexact, tf.cast(-dim_patch_interp,'float32')), tf.greater(jexact, 223))
            # i_or_j_ob = tf.logical_or(i_ob, j_ob)
            # patches = tf.select(i_or_j_ob, tf.zeros_like(patches), patches)

            # clip i and j attention locations so that all i,j >= -patch_dim and <=223
            iclipped = tf.clip_by_value(iexact, tf.cast(-1 * dim_patch_interp, dtype='float32'),
                                        tf.cast(dim_recon - 1, dtype='float32'))
            jclipped = tf.clip_by_value(jexact, tf.cast(-1 * dim_patch_interp, dtype='float32'),
                                        tf.cast(dim_recon - 1, dtype='float32'))

            # add dim_patch_interp to all i,j attention locs so that each one is between [0,233]
            iscaled = iclipped + dim_patch_interp
            jscaled = jclipped + dim_patch_interp
            print('in prep_paste: patches at end')
            print(patches)
            print('iscaled')
            print(iscaled)
            # input('...')
        return patches, iscaled, jscaled

    def __paste(self, patches, i, j, dim_patch_int, dim_recon_int, batch_size_int, nb_feats_int, nb_channels_int):
        with tf.name_scope('paste') as scope:
            # Philipp's idea: use padding to make the 12,12,3 patches become 224,224,3 each (rather than using SparseTensor)
            # then add them all together --> that way we don't lose the gradient information (hopefully!)

            # patches: Tensor with shape: (batch_size*nb_feats, ((dim_patch)**2)*nb_channels) )

            # reshape patches to: (batch_size*nb_feats, nb_channels, dim_patch, dim_patch)
            # so that the channel dim is before the rows and cols dims
            # Note, dim_patch_int now includes the +1 from interpolation --> i.e. dim_patch_int = self.dim_patch_int + 1
            patches = tf.reshape(patches, shape=(-1,nb_channels_int,dim_patch_int,dim_patch_int))

            # dim_recon_int = dim_recon_int + 2*dim_patch_int
            print('patches')
            print(patches)
            i = tf.cast(tf.floor(i), dtype='int32')
            j = tf.cast(tf.floor(j), dtype='int32')
            print('i')
            print(i)
            print('j')
            print(j)
            # input('examin...')
            # dim_recon = tf.constant(dim_recon_int, dtype='int32')
            # patches_shape_int = patches.get_shape().as_list()
            # print('patches_shape_int')
            # print(patches_shape_int)
            # patches_shape = tf.shape(patches)
            # print('patches_shape')
            # print(patches_shape)
            recons_l = list()
            for f in range(batch_size_int * nb_feats_int):
                print('***********')
                print('f:   %d' % f)
                patch = patches[f]
                print('patch')
                print(patch)
                i_f = i[f]
                j_f = j[f]
                print('i_f')
                print(i_f)
                print('j_f')
                print(j_f)
                # input('...patch, i_f, j_f...')
                pad_l = [
                    # [0, 0],     # batch_size*nb_feats dim: no pre or post padding
                    [0, 0],     # channels dim: no pre or post padding
                                # rows (i) dim:
                    [i_f,                                       #   top:    i_f
                     (dim_recon_int + dim_patch_int) - i_f],    #   bottom: (dim_recon_int + dim_patch_int) - i_f
                    [j_f,                                       #   top:    j_f
                     (dim_recon_int + dim_patch_int) - j_f]     #   bottom: (dim_recon_int + dim_patch_int) - j_f
                ]
                # padding = tf.constant(pad_l)
                recon = tf.pad(patch, pad_l, mode='CONSTANT')
                print(recon)
                print(recon.get_shape())
                print(tf.shape(recon))
                # input('...recon...')
                recons_l.append(recon)
            recons = tf.pack(recons_l, axis=0)
            print(recons)
            print(recons.get_shape())
            print(tf.shape(recons))
            # input('...recons, recons.get_shape(), tf.shape(recons)...')

            # crop the out of bounds parts of the reconstruction
            # recons = recons[:,:,:,dim_patch_int:-dim_patch_int,dim_patch_int:-dim_patch_int]
            recons = recons[:,:,dim_patch_int:-dim_patch_int,dim_patch_int:-dim_patch_int]

            print(recons)
            print(recons.get_shape())
            print(tf.shape(recons))
            # input('...after cropping --> recons, recons.get_shape(), tf.shape(recons)...')
            # (5, 1, 3, 224, 224) --> (batch_size*nb_feats, nb_channels, dim_recon, dim_recon)

            # return recons
            return tf.reshape(recons, shape=(-1,nb_feats_int,nb_channels_int,dim_recon_int,dim_recon_int))

    def __merge_recons(self, recons):
        with tf.name_scope('merge_recons') as scope:
            # recons has shape: (batch_size, nb_feats, nb_channels, dim_recon, dim_recon)
            recons = tf.transpose(recons, perm=(0, 2, 1, 3, 4))
            # recons now has shape: (batch_size, nb_channels, nb_feats, dim_recon, dim_recon)
            merged = tf.reduce_sum(recons, reduction_indices=[2], keep_dims=False)
            # merged has shape: (batch_size, nb_channels, dim_recon, dim_recon) where the nb_feats dim has been summed over (thus collapsed)
            merged = tf.transpose(merged, perm=(0,2,3,1)) # restore tf dim ordering where channels dim is last
        return merged

#####################################################################################################################
#####################################################################################################################




    def _decode(self, attention):
        with tf.name_scope('decoder') as scope:
            ijs_patches = self.__deconv_all(attention=attention)
            print('ijs_patches')
            print(ijs_patches)
            # return ijs_patches

            interpolated_patches, iexact, jexact = self.__interpolate(ijs_patches, self.nb_channels_int)
            # return interpolated_patches

            # interpolated shape: (batch_size, nb_feats, (dim_interp_patch**2)*nb_channels)
            interpolated_patches, iscaled, jscaled = self.__prep_paste(interpolated_patches, iexact, jexact,
                                                                       self.dim_recon_int, self.nb_channels_int)
            # return interpolated_patches

            recons = self.__paste(interpolated_patches, iscaled, jscaled, dim_patch_int=self.dim_patch_int+1, dim_recon_int=self.dim_recon_int,
                                  batch_size_int=self.batch_size_int, nb_feats_int=self.nb_feats_int, nb_channels_int=self.nb_channels_int)
            # return recons

            recons_merged = self.__merge_recons(recons)
            # RIGHT HERE
            # recons_merged should have shape: (batch_size, dim_recon, dim_recon, nb_channels)
            return recons_merged, iexact, jexact, interpolated_patches


    def _loss_mse_only(self, reconstructions, targets):
        """
        Add summary for "Loss" and "Loss/avg".
                  of shape [batch_size]
        Returns:
          Loss tensor of type float.
        """
        # Calculate the average cross entropy loss across the batch.
        print('tf.shape(targets)')
        print(tf.shape(targets))
        print(targets)
        print('tf.shape(reconstructions)')
        print(tf.shape(reconstructions))
        print(reconstructions)
        # reconstructions = tf.cast(tf.reshape(reconstructions, shape=[tf.shape(reconstructions)[0], -1]), tf.float32)
        # targets = tf.cast(tf.reshape(targets, shape=[tf.shape(targets)[0], -1]), tf.float32)
        reconstructions = tf.reshape(reconstructions, shape=[tf.shape(reconstructions)[0], -1])
        targets = tf.reshape(targets, shape=[tf.shape(targets)[0], -1])
        print(reconstructions)
        print(targets)
        mse_loss = tf.reduce_mean(tf.square(reconstructions - targets))
        tf.scalar_summary('training loss', mse_loss)
        tf.add_to_collection('losses', mse_loss)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def _loss(self, reconstructions, targets, iexact, jexact):
        """
        Add summary for "Loss" and "Loss/avg".
                  of shape [batch_size]
        Returns:
          Loss tensor of type float.
        """
        iexact = tf.reshape(iexact, shape=(-1,1))
        jexact = tf.reshape(jexact, shape=(-1,1))
        print('in _loss() iexact')
        print(iexact)
        print('in _loss() jexact')
        print(jexact)
        i_ob_loss = tf.maximum(tf.maximum(-1*(self.dim_patch_int + 1) - iexact, iexact - (self.dim_recon_int + self.dim_patch_int + 1 )), 0)
        i_ob_loss = tf.reduce_mean(i_ob_loss)
        j_ob_loss = tf.maximum(tf.maximum(-1*(self.dim_patch_int + 1) - jexact, jexact - (self.dim_recon_int + self.dim_patch_int + 1 )), 0)
        j_ob_loss = tf.reduce_mean(j_ob_loss)

        tf.scalar_summary('i_ob_loss', i_ob_loss)
        tf.scalar_summary('j_ob_loss', j_ob_loss)
        tf.add_to_collection('losses', i_ob_loss)
        tf.add_to_collection('losses', j_ob_loss)

        print('tf.shape(targets)')
        print(tf.shape(targets))
        print(targets)
        print('tf.shape(reconstructions)')
        print(tf.shape(reconstructions))
        print(reconstructions)
        # reconstructions = tf.cast(tf.reshape(reconstructions, shape=[tf.shape(reconstructions)[0], -1]), tf.float32)
        # targets = tf.cast(tf.reshape(targets, shape=[tf.shape(targets)[0], -1]), tf.float32)
        reconstructions = tf.reshape(reconstructions, shape=[tf.shape(reconstructions)[0], -1])
        targets = tf.reshape(targets, shape=[tf.shape(targets)[0], -1])
        print(reconstructions)
        print(targets)
        mse_loss = tf.reduce_mean(tf.square(reconstructions - targets))
        tf.scalar_summary('training_loss', mse_loss)
        tf.add_to_collection('losses', mse_loss)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')


    def _training(self, total_loss, lr, global_step):
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
        # get optimizer: TODO: switch this to Adam
        # optimizer = tf.train.GradientDescentOptimizer(lr)
        optimizer = tf.train.AdamOptimizer()
        # Compute gradients:
        grads = optimizer.compute_gradients(total_loss)
        print(type(grads))
        print(len(grads))
        # Apply gradients:
        print('tf.trainable_variables()')
        print(tf.trainable_variables())
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
                # pass
                tf.histogram_summary(var.op.name + '/gradients', grad) # TODO: restore after debugging
        # input('...grads...')

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op

    def _evaluate(self, reconstructions, targets):
        """Evaluate the quality of the reconstruction.

        Args:
          reconstructions: predicted reconstructed images tensor, float - [batch_size, dim_patch, dim_patch, nb_channels].
          targets: ground truth input images tensor, float - [batch_size, dim_patch, dim_patch, nb_channels].
        Returns:
          A scalar float32 tensor with the avg mean squared error across the entire batch of data
        """
        mse_loss = tf.reduce_mean(tf.square(reconstructions - targets))
        batch_size = tf.cast(tf.shape(targets)[0], tf.float32)
        # Return the number of true entries.
        return mse_loss / batch_size


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
        epoch_size = kwargs.get('epoch_size', None)  # TODO
        nb_epochs = kwargs.get('nb_epochs', 100)
        max_iters = kwargs.get('max_iters', 5000)  # TODO
        lr = kwargs.get('lr', 0.001)
        l2 = kwargs.get('l2', 0.0001)
        SAVE_PATH = kwargs.get('save_path', None)
        if SAVE_PATH is None:
            print('Warning: no SAVE_PATH has been specified.')
        weights = kwargs.get('weights', None)
        load_encoder = kwargs.get('load_encoder', True)
        load_attention = kwargs.get('load_attention', False)
        load_decoder = kwargs.get('load_decoder', False)

        # infinite = kwargs.get('infinite',True)
        # generator = kwargs.get('generator',False)


        # ensure batch_size is set appropriately:
        if batch_size_int is None:
            if self.batch_size_int is None:
                raise ValueError(
                    'batch_size must be specified either in model instantiation or passed into this training method,'
                    'but batch_size cannnot be None in both cases.')
            batch_size_int = self.batch_size_int
        if self.batch_size_int is not None and batch_size_int != self.batch_size_int:
            if self.batch_size_strict:
                raise ValueError('batch_size %d passed to train() method does not match the batch_size %d that'
                                 'the model was built with and batch_size_strict is True.' % (
                                 batch_size_int, self.batch_size_int))
            else:  # self.batch_size_strict == False
                print('Warning: batch_size %d passed to train() method does not match the batch_size %d that'
                      'the model was built with, but proceeding since batch_size_strict is False.' % (
                      batch_size_int, self.batch_size_int))


        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            # Generate placeholders for the input images and the gold-standard reconstructions images (same as input) TODO: clean this up to use same input/output
            inputs_pl = tf.placeholder(tf.float32, [None, self.dim_img_int, self.dim_img_int, self.nb_channels_int])
            # targets_pl = tf.placeholder(tf.float32, [None, self.nb_feats_int, 2 + self.dim_descriptors_int])
            # targets_pl = tf.placeholder(tf.float32, [None, self.nb_feats_int, 2 + (self.dim_patch_int**2)*self.nb_channels_int])
            # targets_pl = tf.placeholder(tf.float32, [None, self.nb_feats_int, ((self.dim_patch_int+1)**2)*self.nb_channels_int])
            # targets_pl = tf.placeholder(tf.float32, [None, self.nb_feats_int, self.nb_channels_int, self.dim_recon_int, self.dim_recon_int])
            # RIGHT HERE
            targets_pl = tf.placeholder(tf.float32, [None, self.dim_img_int, self.dim_img_int, self.nb_channels_int]) # TODO: restore when debugged

            # Create a variable to track the global step.
            global_step = tf.Variable(0, name='global_step', trainable=False)

            # Build a Graph that computes encodings
            encoded = self._encode(inputs_pl)

            # Add to the Graph the Ops for computing attention from encoded
            attended = self._attention(encoded)

            # Add to the Graph the Ops for computing reconstructed images (decodings)
            # from the attention derived from the encodings
            reconstructions, iexact, jexact, patches = self._decode(attention=attended) # TODO: restore when debugged
            # reconstructions = attended # TODO: remove once debugged

            # Add to the Graph the Ops for loss calculation.
            loss = self._loss(reconstructions, targets_pl, iexact, jexact)

            # Add to the Graph the Ops that calculate and apply gradients.
            train_op = self._training(total_loss=loss, lr=lr, global_step=global_step)

            # Add the Op to compute the avg mse loss for evaluation purposes (e.g. for evaluating mse on val data).
            avg_loss_op = self._evaluate(reconstructions, targets_pl)

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

            # And then after everything is built:

            # Run the Op to initialize the variables.
            sess.run(init)

            # load pretrained weights if desired:
            if weights is not None and sess is not None:
                self.load_weights(weights, sess, load_encoder=load_encoder, load_attention=load_attention, load_decoder=load_decoder)

            steps_per_epoch = data_train.nb_samples // batch_size_int
            # TODO: make the training loop a nested for loop that loops over all training data (in inner for loop) nb_epochs number of times (outter for loop)
            # Start the training loop.
            for step in range(max_iters):
                start_time = time.time()

                # Fill a feed dictionary with the actual set of images and labels
                # for this particular training step.
                feed_dict = data_train.fill_feed_dict(inputs_pl, targets_pl, batch_size_int)

                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.  To
                # inspect the values of your Ops or variables, you may include them
                # in the list passed to sess.run() and the value tensors will be
                # returned in the tuple from the call.
                _, loss_value, attention_values, reconstruction_values, patch_vals = sess.run([train_op, loss, attended, reconstructions, patches],
                                         feed_dict=feed_dict)
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

                # Write the summaries and print an overview fairly often.
                # if step > 0 and step % 5 == 0: # 100
                if step % 50 == 0: # 100
                    # Print status to stdout.
                    # print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                    # Update the events file.
                    print('attention_values.shape')
                    print(type(attention_values))
                    print(attention_values.shape)
                    print(attention_values)
                    print('patch_vals.shape')
                    print(type(patch_vals))
                    print(patch_vals.shape)
                    print(patch_vals)
                    # input('...attention_values...(during training)...')
                    print('reconstruction_values.shape')
                    print(type(reconstruction_values))
                    print(reconstruction_values.shape)
                    # input('...reconstruction_values...(during training)...')
                    for samp_num in range(attention_values.shape[0]):
                        for feat_num in range(attention_values.shape[1]):
                            ifeat = attention_values[samp_num, feat_num, 0]
                            jfeat = attention_values[samp_num, feat_num, 1]
                            print('ifeat: %f\tjfeat: %f' % (ifeat, jfeat))
                        recon_img = reconstruction_values[samp_num]
                        fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=2)
                        ax[0].imshow(deprocess_image(recon_img))
                        ax[1].imshow(deprocess_image(feed_dict[inputs_pl][samp_num]))
                        plt.show()

                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    if SAVE_PATH is not None:
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % 250 == 0 or (step + 1) == max_iters: # 1000
                    checkpoint_file = os.path.join(SAVE_PATH, '%s_checkpoint' % self.name )
                    saver.save(sess, checkpoint_file, global_step=step)

                    # Evaluate against the training set.
                    print('Training Data Eval:')
                    self.evaluate(sess, avg_loss_op, inputs_pl, targets_pl, data_train, batch_size_int)

                    # Evaluate against the validation set.
                    print('Validation Data Eval:')
                    self.evaluate(sess, avg_loss_op, inputs_pl, targets_pl, data_val, batch_size_int)
                    for samp_num in range(attention_values.shape[0]):
                        for feat_num in range(attention_values.shape[1]):
                            ifeat = attention_values[samp_num,feat_num,0]
                            jfeat = attention_values[samp_num,feat_num,1]
                            print('ifeat: %f\tjfeat: %f' % (ifeat,jfeat))
                        recon_img = reconstruction_values[samp_num]
                        # fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=2)
                        # ax[0].imshow(deprocess_image(recon_img))
                        # ax[1].imshow(deprocess_image(feed_dict[inputs_pl][samp_num]))
                        # plt.show()

    def predict(self, **kwargs):
        """Use a trained model for prediction."""
        # unpack args:
        X = kwargs.get('X', None)
        batch_size_int = kwargs.get('batch_size', None)
        LOAD_PATH = kwargs.get('load_path', None)
        if LOAD_PATH is None:
            input('Error: no LOAD_PATH has been specified. Randomly initialized model should not be used for prediciton.'
                  '\nPress enter to continue anyways:')

        # ensure batch_size is set appropriately:
        if batch_size_int is None:
            if self.batch_size_int is None:
                raise ValueError(
                    'batch_size must be specified either in model instantiation or passed into this training method,'
                    'but batch_size cannnot be None in both cases.')
            batch_size_int = self.batch_size_int
        if self.batch_size_int is not None and batch_size_int != self.batch_size_int:
            if self.batch_size_strict:
                raise ValueError('batch_size %d passed to train() method does not match the batch_size %d that'
                                 'the model was built with and batch_size_strict is True.' % (
                                 batch_size_int, self.batch_size_int))
            else:  # self.batch_size_strict == False
                print('Warning: batch_size %d passed to train() method does not match the batch_size %d that'
                      'the model was built with, but proceeding since batch_size_strict is False.' % (
                      batch_size_int, self.batch_size_int))


        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            # Generate placeholders for the input images and the gold-standard reconstructions images (same as input) TODO: clean this up to use same input/output
            inputs_pl = tf.placeholder(tf.float32, [None, self.dim_img_int, self.dim_img_int, self.nb_channels_int])
            # targets_pl = tf.placeholder(tf.float32, [None, self.nb_feats_int, 2 + self.dim_descriptors_int])
            # targets_pl = tf.placeholder(tf.float32, [None, self.nb_feats_int, 2 + (self.dim_patch_int**2)*self.nb_channels_int])
            # targets_pl = tf.placeholder(tf.float32, [None, self.nb_feats_int, ((self.dim_patch_int+1)**2)*self.nb_channels_int])
            # targets_pl = tf.placeholder(tf.float32, [None, self.nb_feats_int, self.nb_channels_int, self.dim_recon_int, self.dim_recon_int])

            # Build a Graph that computes encodings
            encoded = self._encode(inputs_pl)

            # Add to the Graph the Ops for computing attention from encoded
            attended = self._attention(encoded)

            # Add to the Graph the Ops for computing reconstructed images (decodings)
            # from the attention derived from the encodings
            reconstructions, iexact, jexact, patches = self._decode(attention=attended)


            # Create a saver for writing training checkpoints.
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

            reconstruction_values, attention_values, ilocs, jlocs, patch_vals = sess.run([reconstructions, attended, iexact, jexact, patches], feed_dict=feed_dict)
            duration = time.time() - start_time

            # Print status to stdout.
            print('Prediction took: %f for batch_size of: %d  --> %f per example' % (duration, self.batch_size_int, (duration / self.batch_size_int)))

            print('attention_values.shape')
            print(type(attention_values))
            print(attention_values.shape)
            print(attention_values)
            print('patch_vals.shape')
            print(type(patch_vals))
            print(patch_vals.shape)
            print(patch_vals)
            # input('...attention_values...(during training)...')
            print('reconstruction_values.shape')
            print(type(reconstruction_values))
            print(reconstruction_values.shape)
            # input('...reconstruction_values...(during training)...')

            for samp_num in range(attention_values.shape[0]):
                for feat_num in range(attention_values.shape[1]):
                    ifeat = attention_values[samp_num,feat_num,0]
                    jfeat = attention_values[samp_num,feat_num,1]
                    print('ifeat: %f\tjfeat: %f' % (ifeat,jfeat))
                recon_img = reconstruction_values[samp_num]
                fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=2)
                ax[0].imshow(deprocess_image(recon_img))
                ax[1].imshow(deprocess_image(feed_dict[inputs_pl][samp_num]))
                plt.show()

    def evaluate(self, sess, avg_loss_op, inputs_pl, targets_pl, data_set, batch_size_int):
        """Runs one evaluation against the full epoch of data.

        Args:
          sess: The session in which the model has been trained.
          avg_loss_op: The Tensor that returns the avg mse loss across a batch of data
          inputs_pl: The images placeholder.
          targets_pl: The labels placeholder.
          data_set: The set of inputs and targets to evaluate (a DataSet object)
        """
        # And run one epoch of eval.
        avg_loss = 0.  # tracks the avg loss over the full epoch of data
        steps_per_epoch = data_set.nb_samples // batch_size_int
        num_examples = steps_per_epoch * batch_size_int
        for step in range(steps_per_epoch):
            feed_dict = data_set.fill_feed_dict(inputs_pl, targets_pl, batch_size_int)
            avg_loss += sess.run(avg_loss_op, feed_dict=feed_dict)
        avg_loss /= steps_per_epoch
        print('  Num examples: %d  AvgLoss: %f ' %
              (num_examples, avg_loss))


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

    def load_weights(self, weight_file, sess, load_encoder=True, load_attention=False, load_decoder=False):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        # pdb.set_trace()
        if load_encoder:
            self.load_weights_encoder(weights=weights, sess=sess)

        # input('...weights loaded successfully...')

if __name__ == '__main__':
    # Constants describing the training process.
    MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
    NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
    LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
    INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.
    ENCODER = 'encoder'
    ATTENTION = 'attention'
    # DECODER = 'decoder'
    DECODER = 'deconver'

    # BATCH_SIZE_INT = 5
    BATCH_SIZE_INT = 1
    # BATCH_SIZE_STRICT = False
    NB_CHANNELS_INT = 3
    NB_FEATS_INT = 1
    DIM_IMG_INT = 224
    DIM_RECON_INT = 224
    DIM_PATCH_INT = 11
    DIM_DESCRIPTORS_INT = 4
    DIM_CONV_REDUCER_INT = 64
    DIM_FC1_INT = 256
    # TRAIN_LIM = 20
    TRAIN_LIM = 1
    # VAL_LIM = 10
    VAL_LIM = 1

    dsae = DSAE_ST_1(batch_size=BATCH_SIZE_INT,
                    nb_feats=NB_FEATS_INT,
                    nb_channels=NB_CHANNELS_INT,
                    dim_img=DIM_IMG_INT,
                    dim_recon=DIM_RECON_INT,
                    dim_patch=DIM_PATCH_INT,
                    dim_conv_reducer=DIM_CONV_REDUCER_INT,
                    dim_fc1=DIM_FC1_INT)

    LOAD_PATH = 'models/tf_VGG/vgg16_weights.npz'
    # weights = np.load(LOAD_PATH)
    # print(type(weights))
    # print('weights.files')
    # print(weights.files)
    # input('...')
    # w = weights['conv4_3_W']
    # print('type(w)')
    # print(type(w))
    # print('w.shape')
    # print(w.shape)
    # input('...')
    # print(w)
    # input('...w...')
    SAVE_PATH = 'models/ST'

    X_train_file = 'data/preprocessed/coco_all/X_train_0.npy'
    X_val_file = 'data/preprocessed/coco_all/X_val_0.npy'
    # X_train_file = 'data/preprocessed/circles/circles_simple2/X_train_0.npy'
    # X_val_file = 'data/preprocessed/circles/circles_simple2/X_val_0.npy'
    X_train = np.load(X_train_file)
    print(X_train.shape)
    if X_train.shape[-1] != 3:
        X_train = np.transpose(X_train, axes=(0,2,3,1))
    # X_train = X_train[0:TRAIN_LIM]
    X_train = X_train[18].reshape((-1,224,224,3))
    print(X_train.shape)

    X_val = np.load(X_val_file)
    print(X_val.shape)
    if X_val.shape[-1] != 3:
        X_val = np.transpose(X_val, axes=(0,2,3,1))
    # X_val = X_val[0:VAL_LIM]
    X_val = X_val[18].reshape((-1,224,224,3))
    print(X_val.shape)


    # y_train_file = 'data/preprocessed/coco_all/X_train_0.npy'
    # y_val_file = 'data/preprocessed/coco_all/X_val_0.npy'
    #
    # y_train = np.load(y_train_file)
    # print(y_train.shape)
    # y_train = np.transpose(y_train, axes=(0,2,3,1))
    # y_train = y_train[0:TRAIN_LIM]
    # print(y_train.shape)
    #
    # y_val = np.load(y_val_file)
    # print(y_val.shape)
    # y_val = np.transpose(y_val, axes=(0,2,3,1))
    # y_val = y_val[0:VAL_LIM]
    # print(y_val.shape)

    # tf.Graph.

    # # use this when stopping after reconstructions = self._attention()
    # y_train = np.random.random_sample(size=(TRAIN_LIM,NB_FEATS_INT,2+DIM_DESCRIPTORS_INT))
    # print('y_train.shape')
    # print(y_train.shape)
    # y_val = np.random.random_sample(size=(VAL_LIM,NB_FEATS_INT,2+DIM_DESCRIPTORS_INT))
    # print('y_val.shape')
    # print(y_val.shape)

    # # use this when stopping after reconstructions = self._decode() but returning after ijs_patches = __deconv_all()
    # y_train = np.random.random_sample(size=(TRAIN_LIM,NB_FEATS_INT,2+((DIM_PATCH_INT**2)*NB_CHANNELS_INT)))
    # print('y_train.shape')
    # print(y_train.shape)
    # y_val = np.random.random_sample(size=(VAL_LIM,NB_FEATS_INT,2+((DIM_PATCH_INT**2)*NB_CHANNELS_INT)))
    # print('y_val.shape')
    # print(y_val.shape)

    # # use this when stopping after reconstructions = self._decode() but returning after interpolated_patches, iexact, jexact = self.__interpolate(ijs_patches, self.nb_channels_int)
    # y_train = np.random.random_sample(size=(TRAIN_LIM,NB_FEATS_INT,((DIM_PATCH_INT+1)**2)*NB_CHANNELS_INT))
    # print('y_train.shape')
    # print(y_train.shape)
    # y_val = np.random.random_sample(size=(VAL_LIM,NB_FEATS_INT,((DIM_PATCH_INT+1)**2)*NB_CHANNELS_INT))
    # print('y_val.shape')
    # print(y_val.shape)

    # # use this when stopping after reconstructions = self._decode() but returning after recons = self.__paste()
    # y_train = np.random.random_sample(size=(TRAIN_LIM,NB_FEATS_INT,NB_CHANNELS_INT,DIM_RECON_INT,DIM_RECON_INT)) * 224
    # print('y_train.shape')
    # print(y_train.shape)
    # y_val = np.random.random_sample(size=(VAL_LIM,NB_FEATS_INT,NB_CHANNELS_INT,DIM_RECON_INT,DIM_RECON_INT)) * 224
    # print('y_val.shape')
    # print(y_val.shape)

    # use this when stopping after reconstructions = self._decode() but returning after recons = self.__paste()
    y_train = np.copy(X_train)
    print('y_train.shape')
    print(y_train.shape)
    y_val = np.copy(X_val)
    print('y_val.shape')
    print(y_val.shape)


    data_train_ = DataSet(X=X_train, y=y_train, batch_size=BATCH_SIZE_INT)
    data_val_ = DataSet(X=X_val, y=y_val, batch_size=BATCH_SIZE_INT)

    TRAIN = True
    # TRAIN = False

    if TRAIN:
        dsae.train(data_train=data_train_, data_val=data_val_,
                   batch_size=BATCH_SIZE_INT,
                   save_path=SAVE_PATH,
                   weights=LOAD_PATH)
    else:
        dsae.predict(X=X_train[0:5],
                   batch_size=BATCH_SIZE_INT,
                   load_path=SAVE_PATH)