########################################################################################
# Adapted from Davi Frossard, 2016                                                                  #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
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
from lib.vgg6t_encoder import get_encoder

from collections import defaultdict
from lib.data_set import DataSet
from lib.image_utils import deprocess_image

I = 'inputs'
O = 'outputs'
P = 'params'
ENCODER = 'encoder'


class VGG6t(object):
    def __init__(self, **kwargs):
        self.name = 'vgg6t_a'
        # unpack args:
        self.batch_size_int = kwargs.get('batch_size',32)
        self.dim_img_int = kwargs.get('dim_img', 224)
        self.nb_channels_int = kwargs.get('nb_channels', 3)
        self.nb_classes = kwargs.get('nb_classes', 2) # number of target classes we are predicting
        self.encoder = defaultdict(dict) # each key maps to a dict with keys: inputs, outputs, params

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _encode(self, inputs_pl):
        preds, self.encoder = get_encoder(inputs_pl= inputs_pl,
                                      batch_size=self.batch_size_int,
                                      dim_img=self.dim_img_int,
                                      nb_channels=self.nb_channels_int,
                                      nb_classes=self.nb_classes,
                                      encoder=self.encoder)

        return preds # --> self.encoder['encoder-fc3']['output']

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
        batch_size_int = self.batch_size_int
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
        WEIGHTS_PATH = kwargs.get('weights_path', None)
        RESTORE_PATH = kwargs.get('restore_path', None)
        load_encoder = kwargs.get('load_encoder', True)

        # ensure batch_size is set appropriately:
        if batch_size_int is None:
            raise ValueError('batch_size must be specified in model instantiation.')

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

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver(max_to_keep=10)

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # Instantiate a SummaryWriter to output summaries and the Graph.
            if SAVE_PATH is not None:
                summary_writer = tf.train.SummaryWriter(SAVE_PATH, sess.graph)
            else:
                print('WARNING: SAVE_PATH is not specified...cannot save model file')

            if RESTORE_PATH not in {None, ''}:
                # checkpoint_dir = os.path.dirname(RESTORE_PATH)
                # checkpoint_name = os.path.basename(RESTORE_PATH)
                # if checkpoint_name not in {'', None}:
                #     checkpoint_path = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir, latest_filename=checkpoint_name)
                # else:
                #     checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
                # saver.restore(sess, checkpoint_path)
                # print('model restored from checkpoint file: %s' % str(checkpoint_path))
                saver.restore(sess, RESTORE_PATH)
                print('model restored from checkpoint file: %s' % str(RESTORE_PATH))
            else:
                print('No RESTORE_PATH specified, so initializing the model with random weights for training...')
                # Add the variable initializer Op.
                init = tf.initialize_all_variables()
                # Run the Op to initialize the variables.
                sess.run(init)

            # load pretrained weights if desired:
            if WEIGHTS_PATH not in {None, ''} and sess is not None:
                self.load_weights(WEIGHTS_PATH, sess, load_encoder=load_encoder)

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
                epoch_tot_loss = 0.0
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

                    epoch_tot_loss += loss_value
                    epoch_avg_loss = epoch_tot_loss / batch_num
                    # Print status to stdout.
                    print('\tbatch_num %d / %d : batch_loss = %.3f \tepoch_avg_loss = %.3f \t(%.3f sec)' % (batch_num, nb_batches_per_epoch, loss_value, epoch_avg_loss, batch_duration))

                    # Write the summaries and print an overview fairly often.
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
        RESTORE_PATH = kwargs.get('restore_path', None)
        if RESTORE_PATH is None:
            input('Error: no RESTORE_PATH has been specified. Randomly initialized model should not be used for prediciton.'
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
            if RESTORE_PATH is not None:
                # checkpoint_dir = os.path.dirname(RESTORE_PATH)
                # checkpoint_name = os.path.basename(RESTORE_PATH)
                # if checkpoint_name not in {'', None}:
                #     print('checkpoint_dir')
                #     print(checkpoint_dir)
                #     print('checkpoint_name')
                #     print(checkpoint_name)
                #     checkpoint_path = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir, latest_filename=checkpoint_name)
                # else:
                #     checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
                #     print('checkpoint_path')
                #     print(checkpoint_path)
                # saver.restore(sess, checkpoint_path)
                # print('model restored from checkpoint file: %s' % str(checkpoint_path))
                saver.restore(sess, RESTORE_PATH)
                print('model restored from checkpoint file: %s' % str(RESTORE_PATH))
            else:
                print('No RESTORE_PATH specified, so initializing the model with random weights')
                # Add the variable initializer Op.
                init = tf.initialize_all_variables()
                # Run the Op to initialize the variables.
                sess.run(init)

            # Do Prediction:
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = {inputs_pl: X}
            nb_samples = X.shape[0]

            pred_vals = sess.run([preds], feed_dict=feed_dict)[0] # [0] since sess.run([preds]) returns a list of len 1 in this case

            duration = time.time() - start_time

            # Print status to stdout.
            print('Prediction took: %f for %d samples  --> %f per sample' % (duration, nb_samples, (duration / nb_samples)))

            print('pred_vals.shape')
            print(type(pred_vals))
            print(pred_vals.shape)
            print(pred_vals)
            print('...pred_vals...(during prediction)...')

            for samp_num in range(X.shape[0]):
                img = X[samp_num]
                # print('X.shape')
                # print(X.shape)
                # print(X[0,0,0:5])
                # input('check....')
                scores = pred_vals[samp_num]
                probs = self.softmax(scores)
                class_pred_idx = np.argmax(probs)
                class_pred = idx2label[class_pred_idx]
                plt.imshow(deprocess_image(img))
                txt = 'predicted class dist: %s\n' \
                      'predicted class: %s\n' \
                      % (str(probs), class_pred)
                plt.text(0, 0, txt, color='b', fontsize=15, fontweight='bold')
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
            # 'conv1_2_W': 'encoder-conv1_2',
            # 'conv1_2_b': 'encoder-conv1_2',
            # 'conv2_1_W': 'encoder-conv2_1',
            # 'conv2_1_b': 'encoder-conv2_1',
            # 'conv2_2_W': 'encoder-conv2_2',
            # 'conv2_2_b': 'encoder-conv2_2',
            # 'conv3_1_W': 'encoder-conv3_1',
            # 'conv3_1_b': 'encoder-conv3_1',
            # 'conv3_2_W': 'encoder-conv3_2',
            # 'conv3_2_b': 'encoder-conv3_2',
            # 'conv3_3_W': 'encoder-conv3_3',
            # 'conv3_3_b': 'encoder-conv3_3',
            # 'conv4_1_W': 'encoder-conv4_1',
            # 'conv4_1_b': 'encoder-conv4_1',
            # 'conv4_2_W': 'encoder-conv4_2',
            # 'conv4_2_b': 'encoder-conv4_2',
            # 'conv4_3_W': 'encoder-conv4_3',
            # 'conv4_3_b': 'encoder-conv4_3',
            # 'conv5_1_W': 'encoder-conv5_1',
            # 'conv5_1_b': 'encoder-conv5_1',
            # 'conv5_2_W': 'encoder-conv5_2',
            # 'conv5_2_b': 'encoder-conv5_2',
            # 'conv5_3_W': 'encoder-conv5_3',
            # 'conv5_3_b': 'encoder-conv5_3',
            # 'fc6_W': 'encoder-fc1',
            # 'fc6_b': 'encoder-fc1',
            # 'fc7_W': 'encoder-fc2',
            # 'fc7_b': 'encoder-fc2',
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

    def load_weights(self, weight_path, sess, load_encoder=True):
        weights = np.load(weight_path)
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
    cmdln.add_argument("--train-lim", action="store", dest="TRAIN_LIM", default=-1, type=int,
                                help="amount of training data to use [default: %default].")
    cmdln.add_argument("--val-lim", action="store", dest="VAL_LIM", default=250, type=int,
                                help="amount of validation data to use [default: %default].")
    cmdln.add_argument("--test-lim", action="store", dest="TEST_LIM", default=100, type=int,
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
    cmdln.add_argument("--save-best-only", action="store", dest="SAVE_BEST_ONLY", default='save_all',
                                help="save all models at checkpoints or only save if the model has a new best "
                                     "training_acc or val_acc (specify which) [default: %default].")
    cmdln.add_argument("--weights-path", action="store", dest="WEIGHTS_PATH", default='',
                                help="path from which to load the pretrained weights for the model [default: %default].")
    cmdln.add_argument("--restore-path", action="store", dest="RESTORE_PATH", default='',
                                help="path from which to restore a previously trained model [default: %default].")
    cmdln.add_argument("--save-path", action="store", dest="SAVE_PATH", default='',
                                help="path at which to save the model checkpoints during training [default: %default].")

    cmdln.add_argument("--X-train", action="store", dest="X_TRAIN_PATH", default='',
                                help="path where the X_train data is stored [default: %default].")
    cmdln.add_argument("--X-val", action="store", dest="X_VAL_PATH", default='',
                                help="path where the X_val data is stored [default: %default].")
    cmdln.add_argument("--X-test", action="store", dest="X_TEST_PATH", default='',
                                help="path where the X_test data is stored [default: %default].")
    cmdln.add_argument("--y-train", action="store", dest="Y_TRAIN_PATH", default='',
                                help="path where the y_train data is stored [default: %default].")
    cmdln.add_argument("--y-val", action="store", dest="Y_VAL_PATH", default='',
                                help="path where the y_val data is stored [default: %default].")
    cmdln.add_argument("--y-test", action="store", dest="Y_TEST_PATH", default='',
                                help="path where the y_test data is stored [default: %default].")
    cmdln.add_argument("--train", action="store_true", dest="TRAIN", default=False,
                                help="specify whether to train the model or just use for prediction (default) [default: %default].")
    return  cmdln


if __name__ == '__main__':
    label_map = {
        'S': [1, 0, 0],
        'T': [0, 1, 0],
        'Bad': [0, 0, 1],
    }
    idx2label = ['S', 'T', 'Bad']

    # Constants describing the training process.
    MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
    NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
    LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
    INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

    cmdln = prep_cmdln_parser()
    args = cmdln.parse_args()

    vgg = VGG6t(batch_size=args.BATCH_SIZE_INT,
                    nb_channels=args.NB_CHANNELS_INT,
                    dim_img=args.DIM_IMG_INT)

    X_train = np.load(args.X_TRAIN_PATH)
    X_val = np.load(args.X_VAL_PATH)
    X_test = np.load(args.X_TEST_PATH)
    if X_train.shape[-1] != 3:
        X_train = np.transpose(X_train, axes=(0, 2, 3, 1))
    if X_val.shape[-1] != 3:
        X_val = np.transpose(X_val, axes=(0, 2, 3, 1))
    if X_test.shape[-1] != 3:
        X_test = np.transpose(X_test, axes=(0, 2, 3, 1))

    y_train = np.load(args.Y_TRAIN_PATH)
    y_val = np.load(args.Y_VAL_PATH)
    y_test = np.load(args.Y_TEST_PATH)

    if args.TRAIN_LIM > 0:
        X_train = X_train[0:args.TRAIN_LIM]
        y_train = y_train[0:args.TRAIN_LIM]
    if args.VAL_LIM > 0:
        X_val = X_val[0:args.VAL_LIM]
        y_val = y_val[0:args.VAL_LIM]
    if args.TEST_LIM > 0:
        X_test = X_test[0:args.TEST_LIM]
        y_test = y_test[0:args.TEST_LIM]

    print('X_train.shape: %s' % str(X_train.shape))
    print('y_train.shape: %s' % str(y_train.shape))
    print('X_val.shape: %s' % str(X_val.shape))
    print('y_val.shape: %s' % str(y_val.shape))
    print('X_test.shape: %s' % str(X_test.shape))
    print('y_test.shape: %s' % str(y_test.shape))

    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    data_train_ = DataSet(X=X_train, y=y_train, batch_size=args.BATCH_SIZE_INT)
    data_val_ = DataSet(X=X_val, y=y_val, batch_size=args.BATCH_SIZE_INT)
    data_test_ = DataSet(X=X_test, y=y_test, batch_size=args.BATCH_SIZE_INT)

    if args.TRAIN:
        history, best_train_acc, best_val_acc = \
            vgg.train(data_train=data_train_, data_val=data_val_,
                  save_path=args.SAVE_PATH,
                  weights_path=args.WEIGHTS_PATH,
                  restore_path=args.RESTORE_PATH,
                  save_summaries_every=args.SAVE_SUMMARIES_EVERY,
                  display_every=args.DISPLAY_EVERY,
                  display=args.DISPLAY,
                  nb_to_display=args.NB_TO_DISPLAY,
                  nb_epochs=args.NB_EPOCHS,
                  save_best_only=args.SAVE_BEST_ONLY)

    else:
        vgg.predict(X=X_test[0:5],
                    restore_path=args.RESTORE_PATH)