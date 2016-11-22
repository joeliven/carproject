"""Lightweight class for storing a data set"""
import tensorflow as tf
import numpy as np


class DataSet(object):

    def __init__(self, **kwargs):
        # unpack args:
        self.X = kwargs.get('X', None)
        if self.X is None:
            raise ValueError('X cannot be None')
        self.y = kwargs.get('y', None)
        if self.y is None:
            print('Warning: no targets supplied, so assuming targets are the same as inputs')
            self.y = self.X

        self.batch_size = kwargs.get('batch_size',32)
        self.infinite = kwargs.get('infinite',True)
        self.epoch_size = kwargs.get('epoch_size',True)
        self.generator = kwargs.get('generator',False)

        # set the current idx in the data:
        self.bookmark = 0
        self.cur_batch = None

        self.nb_samples = self.X.shape[0]
        assert self.nb_samples == self.y.shape[0]
        self._end_of_epoch = False

    def reset(self):
        self.bookmark = 0

    def end_of_epoch(self):
        return self._end_of_epoch

    def next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size != self.batch_size:
            print('Warning: using batch_size different than what was originally set for this DataSet.')

        X_batch = None
        y_batch = None
        if self.bookmark + batch_size > self.nb_samples:
            self._end_of_epoch = True
            if not self.infinite:
                raise StopIteration('No more data left in the DataSet and infinite is False.')
            else:
                leftover = self.nb_samples - self.bookmark
                dif = batch_size - leftover
                # grab the end of the data set and the first 'dif' samples at the start of the dataset
                X_end_part = self.X[self.bookmark:]
                X_begin_part = self.X[0:dif]
                X_batch = np.concatenate((X_end_part,X_begin_part), axis=0)
                assert X_batch.shape[0] == batch_size
                y_end_part = self.y[self.bookmark:]
                y_begin_part = self.y[0:dif]
                y_batch = np.concatenate((y_end_part,y_begin_part), axis=0)
                assert y_batch.shape[0] == batch_size
                self.bookmark_train = dif
        else:
            self._end_of_epoch = False
            X_batch = self.X[self.bookmark : self.bookmark + batch_size]
            y_batch = self.y[self.bookmark : self.bookmark + batch_size]
            self.bookmark += batch_size
        if self.cur_batch is None:
            self.cur_batch = {}
        self.cur_batch['X'] = X_batch
        self.cur_batch['y'] = y_batch
        return X_batch, y_batch

    def get_cur_batch(self):
        if self.cur_batch is not None:
            return self.cur_batch
        else:
            return  None

    def fill_feed_dict(self, inputs_pl, targets_pl, batch_size=None):
        """Fills the feed_dict for training the given step.

        A feed_dict takes the form of:
        feed_dict = {
          <placeholder>: <tensor of values to be passed for placeholder>,
          ....
        }

        Args:
        inputs_pl: The images placeholder
        targets_pl: The targets placeholder

        Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
        """
        # Create the feed_dict for the placeholders filled with the next
        # `batch size` examples.
        X_batch, y_batch = self.next_batch(batch_size=batch_size)
        feed_dict = {
          inputs_pl: X_batch,
          targets_pl: y_batch,
        }
        return feed_dict
