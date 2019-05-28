
"""Rand Conv."""

import tensorflow as tf

from keras.layers import Conv2D
from keras import activations as act


class MaskConv(Conv2D):

    def __init__(self, **kwargs):
        self._output_dim = kwargs['filters']
        super(MaskConv, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.mask = self.add_weight(
            name='mask',
            shape=kernel_shape,
            initializer='uniform',
            trainable=True
        )
        super(MaskConv, self).build(input_shape)

    def call(self, x):
        mask = act.sigmoid(self.mask)
        self.kernel = tf.multiply(self.kernel, mask)
        return super(MaskConv, self).call(x)
