
"""Gaussian Conv2D."""

from keras import backend as K
from keras.layers import Conv2D, Dense


class GaussianConv2D(Conv2D):

    def __init__(self, *args, **kwargs):
        self._stddev_split = 10
        if 'stddev_split' in kwargs:
            self._stddev_split = kwargs['stddev_split']
        super(GaussianConv2D, self).__init__(*args, **kwargs)

    def call(self, inputs, training=None):
        def noised():
            stddev = K.mean(self.kernel) / self._stddev_split
            return K.random_normal(
                shape=K.shape(self.kernel), mean=0., stddev=stddev
            )
        self.kernel = self.kernel + K.in_train_phase(
            noised, 0., training=training
        )
        return super(GaussianConv2D, self).call(inputs)


class GaussianDense(Dense):

    def __init__(self, *args, **kwargs):
        self._stddev_split = 10
        if 'stddev_split' in kwargs:
            self._stddev_split = kwargs['stddev_split']
        super(GaussianDense, self).__init__(*args, **kwargs)

    def call(self, inputs, training=None):
        def noised():
            stddev = K.mean(self.kernel) / self._stddev_split
            return K.random_normal(
                shape=K.shape(self.kernel), mean=0., stddev=stddev
            )
        self.kernel = self.kernel + K.in_train_phase(
            noised, 0., training=training
        )
        return super(GaussianDense, self).call(inputs)
