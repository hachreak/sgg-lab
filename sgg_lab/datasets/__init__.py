
"""Datasets."""

import numpy as np
import json
import csv

from PIL import Image, ImageOps


def randomize(objects):
    import random
    random.shuffle(objects)
    return objects


def epochs(filenames, epochs=1, random=True):
    """Repeat filenames epochs times."""
    for _ in range(0, epochs):
        filenames = randomize(filenames)
        for name in filenames:
            yield name


def stream_batch(stream, fun=None, size=None):
    """Create batch on the fly."""
    fun = fun or (lambda x, y: (np.array(x), np.array(y)))
    size = size or 5
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


def load_json(filename):
    """Load json file."""
    with open(filename) as f:
        data = json.load(f)
    return data


def load_csv(filename, skip_headers=True):
    """Load json file."""
    with open(filename) as f:
        reader = csv.reader(f)
        if skip_headers:
            next(reader)
        for row in reader:
            yield row


def save_json(filename, to_write):
    """Save stream in a json file."""
    with open(filename, 'w') as f:
        json.dump(to_write, f)


def get_color(count):
    """Transform a index into a color."""
    step = (255**3) / float(count)

    def f(index):
        id_ = int(index * step)
        red = id_ % 255
        channel_2 = (id_ // 255)
        green = channel_2 % 255
        channel_3 = (channel_2 // 255)
        blue = channel_3 % 255
        return red, green, blue
    return f


def colorize(img, rgb):
    """Colorize a image."""
    r, g, b = rgb
    # apply color to a RGB mask
    img = np.transpose(img, (2, 0, 1))
    img[0][img[0] > 0] = r
    img[1][img[1] > 0] = g
    img[2][img[2] > 0] = b
    img = np.transpose(img, (1, 2, 0))
    return img


def mask2image(mask):
    """Convert a binary mask to a black/white image."""
    mask = mask.copy().astype('uint8')
    mask[mask > 0] = 255
    return mask


def image2mask(mask):
    """Convert a image to a 0/1 image."""
    mask = mask.copy().astype('uint8')
    mask[mask > 0] = 1
    return mask


def sample_image(image, mask):
    """Extract portion of image relating to the mask."""
    image = image.copy()
    image[mask == 0] = 0
    return image


def show(image):
    """Show an image."""
    Image.fromarray(image).show()


def resize(shape):
    """Resize and mantain the aspect ratio."""
    def f(image):
        img = Image.fromarray(image)
        if image.shape[0] > shape[0] or image.shape[1] > shape[1]:
            img.thumbnail(shape)
        pad = 0, 0, shape[0] - img.size[0], shape[1] - img.size[1]
        return np.array(ImageOps.expand(img, pad))
    return f
