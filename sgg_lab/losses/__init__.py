
"""Losses."""

import tensorflow as tf


def count(matrix, value):
    """Count how many elements inside the matrix."""
    return tf.reduce_sum(tf.cast(tf.equal(matrix, value), tf.float32))


def count_all(matrix):
    """Count how many elements."""
    from keras import backend as K
    return tf.cast(tf.reduce_prod(K.shape(matrix)), dtype=tf.float32)
