
"""Datasets."""

import numpy as np
import cv2


def epochs(filenames, epochs=1):
    """Repeat filenames epochs times."""
    for _ in range(0, epochs):
        for name in filenames:
            yield name


def stream_batch(stream, fun, size):
    """Create batch on the fly."""
    while True:
        batch = []
        try:
            x = []
            y = []
            for i in range(size):
                x_value, y_value = next(stream)
                x.append(x_value)
                y.append(y_value)
            yield fun(x, y)
        except StopIteration:
            if not batch:
                raise StopIteration
            yield batch


def pack_elements(x, y):
    """Get x,y and pack as [x, [y1, y2, ...]] as numpy arrays."""
    y_zipped = zip(*y)
    return [
        np.array(x),
        [np.array(y_value) for y_value in y_zipped]
    ]


def extract_xy(x_key, y_keys):
    def f(value):
        return [value[x_key], [value[yk] for yk in y_keys]]
    return f


def _apply_adapters(batch, adapters):
    """Apply adapters to the batch."""
    for adapter in adapters:
        batch = adapter(batch)
    return batch


def adapt(batches, adapters):
    """Adapt a streaming batch."""
    for batch in batches:
        yield _apply_adapters(batch, adapters)


def apply_to_x(fun):
    def f(both):
        x, y = both
        return fun(x), y
    return f


def apply_to_y(fun):
    def f(both):
        x, y = both
        return x, fun(y)
    return f


def apply_to_xn(fun):
    """Apply a function to n inputs [x1, x2, .., xn]."""
    def f(xn):
        return [fun(x) for x in xn]
    return f


def stream(fun, stream):
    """Transform a function into a stream."""
    for value in stream:
        yield fun(value)


def apply_to_key(fun, src, dst):
    def f(value):
        value[dst] = fun(value[src])
        return value
    return f


def apply_to_keys(fun, src1, src2, dst):
    def f(x):
        x[dst] = fun(x[src1], x[src2])
        return x
    return f


def resize_img(input_shape, img_key, bbox_key):
    i_width, i_height = input_shape[-3:-1]

    def f(value):
        width, height = value[img_key].shape[-3:-1]
        # resize img
        value[img_key] = cv2.resize(value[img_key], (i_height, i_width))
        # resize bbox
        value[bbox_key][:, 0] = \
            value[bbox_key][:, 0] * (i_width / float(width))
        value[bbox_key][:, 1] = \
            value[bbox_key][:, 1] * (i_height / float(height))
        value[bbox_key][:, 2] = \
            value[bbox_key][:, 2] * (i_width / float(width))
        value[bbox_key][:, 3] = \
            value[bbox_key][:, 3] * (i_height / float(height))
        return value

    return f
