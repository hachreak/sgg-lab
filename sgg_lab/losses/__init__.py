
"""Losses."""

import tensorflow as tf


def count(matrix, value):
    """Count how many elements inside the matrix."""
    return tf.reduce_sum(tf.cast(tf.equal(matrix, value), tf.int32))


def count_all(matrix):
    """Count how many elements."""
    return tf.reduce_prod(x.shape)
