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

import tensorflow as tf
import numpy as np
import time
from DSAE.lib_tf.data_set import DataSet
from DSAE.lib_tf.imagenet_classes import class_names


class dsae(object):
    def __init__(self, **kwargs):
        self.name = 'vggtest'
        # unpack args:
        self.batch_size_int = kwargs.get('batch_size',32)
        self.batch_size_strict = kwargs.get('batch_size_strict', False)
        self.nb_channels_int = kwargs.get('nb_channels', 3)
        self.img_dim_int = kwargs.get('img_dim', 224)
        self.recon_dim_int = kwargs.get('recon_dim', 224)
        self.patch_dim_int = kwargs.get('patch_dim', 11)


    def _inference(self, inputs_pl):
        self.parameters = []

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(inputs_pl, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

        # # softmax
        # with tf.name_scope('softmax') as scope:
        #     self.probs = tf.nn.softmax(self.fc3l)

        return self.fc3l

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
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            predictions, targets, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

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
        # get optimizer:
        optimizer = tf.train.GradientDescentOptimizer(lr)
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

    def _evaluate(self, predictions, targets):
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
        correct = tf.nn.in_top_k(predictions, targets, 1)
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
        epoch_size = kwargs.get('epoch_size', None)  # TODO
        nb_epochs = kwargs.get('nb_epochs', 100)
        max_iters = kwargs.get('max_iters', 100)  # TODO
        lr = kwargs.get('lr', 0.001)
        l2 = kwargs.get('l2', 0.0001)
        SAVE_PATH = kwargs.get('save_path', None)
        if SAVE_PATH is None:
            print('Warning: no SAVE_PATH has been specified.')
        weights = kwargs.get('weights', None)

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
            # Generate placeholders for the images and labels.
            inputs_pl = tf.placeholder(tf.float32, [None, self.img_dim_int, self.img_dim_int, self.nb_channels_int])
            # targets_pl = tf.placeholder(tf.float32, [None, self.recon_dim_int, self.recon_dim_int, self.nb_channels_int])
            targets_pl = tf.placeholder(tf.int32, [None,])

            # Create a variable to track the global step.
            global_step = tf.Variable(0, name='global_step', trainable=False)

            # Build a Graph that computes predictions from the inference model.
            predictions = self._inference(inputs_pl)

            # Add to the Graph the Ops for loss calculation.
            loss = self._loss(predictions, targets_pl)

            # Add to the Graph the Ops that calculate and apply gradients.
            train_op = self._training(total_loss=loss, lr=lr, global_step=global_step)

            # Add the Op to compare the predictions to the targets during evaluation.
            eval_correct = self._evaluate(predictions, targets_pl)

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
                self.load_weights(weights, sess)

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
                _, loss_value = sess.run([train_op, loss],
                                         feed_dict=feed_dict)

                duration = time.time() - start_time
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                # Write the summaries and print an overview fairly often.
                if step % 1 == 0: # 100
                    # Print status to stdout.
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                    # Update the events file.
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    if SAVE_PATH is not None:
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % 20 == 0 or (step + 1) == max_iters: # 1000
                    checkpoint_file = os.path.join(SAVE_PATH, '%s_checkpoint' % self.name )
                    saver.save(sess, checkpoint_file, global_step=step)

                    # Evaluate against the training set.
                    print('Training Data Eval:')
                    self.evaluate(sess, eval_correct, inputs_pl, targets_pl, data_train, batch_size_int)

                    # Evaluate against the validation set.
                    print('Validation Data Eval:')
                    self.evaluate(sess, eval_correct, inputs_pl, targets_pl, data_val, batch_size_int)


    def evaluate(self, sess, eval_correct, inputs_pl, targets_pl, data_set, batch_size_int):
        """Runs one evaluation against the full epoch of data.

        Args:
          sess: The session in which the model has been trained.
          eval_correct: The Tensor that returns the number of correct predictions.
          inputs_pl: The images placeholder.
          targets_pl: The labels placeholder.
          data_set: The set of inputs and targets to evaluate (a DataSet object)
        """
        # And run one epoch of eval.
        true_count = 0  # Counts the number of correct predictions.
        steps_per_epoch = data_set.nb_samples // batch_size_int
        num_examples = steps_per_epoch * batch_size_int
        for step in range(steps_per_epoch):
            feed_dict = data_set.fill_feed_dict(inputs_pl, targets_pl, batch_size_int)
            true_count += sess.run(eval_correct, feed_dict=feed_dict)
        precision = true_count / num_examples
        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
              (num_examples, true_count, precision))


    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))


if __name__ == '__main__':
    # Constants describing the training process.
    MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
    NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
    LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
    INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

    BATCH_SIZE_INT = 5
    # BATCH_SIZE_STRICT = False
    NB_CHANNELS_INT = 3
    IMG_DIM_INT = 224
    RECON_DIM_INT = 224
    PATCH_DIM_INT = 11
    TRAIN_LIM = 20
    VAL_LIM = 10

    vgg = dsae(batch_size=BATCH_SIZE_INT,
               nb_channels=NB_CHANNELS_INT,
               img_dim=IMG_DIM_INT,
               recon_dim=RECON_DIM_INT,
               patch_dim=PATCH_DIM_INT)

    LOAD_PATH = 'lib_tf/vgg16_weights.npz'
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
    X_train = np.load(X_train_file)
    print(X_train.shape)
    X_train = np.transpose(X_train, axes=(0,2,3,1))
    X_train = X_train[0:TRAIN_LIM]
    print(X_train.shape)

    X_val = np.load(X_val_file)
    print(X_val.shape)
    X_val = np.transpose(X_val, axes=(0,2,3,1))
    X_val = X_val[0:VAL_LIM]
    print(X_val.shape)

    y_train = np.random.randint(0,1000, size=(TRAIN_LIM))
    print('y_train.shape')
    print(y_train.shape)
    y_val = np.random.randint(0,1000, size=(VAL_LIM))
    print('y_val.shape')
    print(y_val.shape)

    data_train = DataSet(X=X_train, y=y_train)
    data_val = DataSet(X=X_val, y=y_val)

    vgg.train(data_train=data_train, data_val=data_val, batch_size=BATCH_SIZE_INT, save_path=SAVE_PATH, weights=LOAD_PATH)
