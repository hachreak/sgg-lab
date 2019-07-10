
"""Train Noise Removal on coco."""

import utils as u

import numpy as np
import os

#  from joblib import parallel_backend, Parallel, delayed

from sgg_lab import datasets as ds
from sgg_lab.callbacks import ModelSaveBestAvgAcc, filter_val_f1score


def to_regression(dataset, resizer, input_shape, image_id):
    masks, cls = dataset.load_mask(image_id)
    cls = cls / float(dataset.num_classes)
    result = np.zeros(input_shape[:2])
    mask = np.full(input_shape[:2], True, dtype=np.bool)
    for i in range(0, masks.shape[2]):
        m = resizer(masks[:, :, i].astype('uint8'))
        m[m > 0] = 1
        result[m == 1] = cls[i]
        mask = np.logical_and(m, mask)
    return result


def imageid2categoricalpixel(dataset, input_shape):
    """Convert image_id to categorical pixel."""
    resizer = ds.resize(input_shape)

    def f(image_id):
        return to_regression(dataset, resizer, input_shape, image_id)
    return f


def prepare(dataset, epochs, batch_size, input_shape, output_shape):
    i2c = imageid2categoricalpixel(dataset, input_shape)
    stream = ds.epochs(dataset.image_ids, epochs)
    stream = ds.stream(
        lambda x: (dataset._img_filenames[x], i2c(x)),
        stream)

    def adapt_y(x, y):
        y = np.array(y).transpose((2, 0, 1))
        y = y.reshape(y.shape + (1,))  # .astype('int32')
        return np.array(x), [v for v in y]

    batch = ds.stream_batch(stream, size=batch_size, fun=adapt_y)

    return batch


def save_all(gen, base_path, directory):
    """Save all resized images."""
    dest = os.path.join(base_path, directory)
    if not os.path.exists(dest):
        os.makedirs(dest)
    print('process {0}'.format(dest))
    for names, imgs in gen:
        for i in range(0, len(names)):
            res = np.array([imgs[j][i] for j in range(0, len(imgs))])
            path = os.path.join(dest, names[i])
            np.savez_compressed(path, res)


coco_path = '/media/hachreak/Magrathea/datasets/coco/coco'
model_path = ''
epochs = 1
batch_size = 3
input_shape = (320, 320, 3)
output_shape = (10, 10)

dest_path = '/media/hachreak/Magrathea/datasets/coco/v14_resize_{0}x{1}' \
        .format(*input_shape[:2])

# validation dataset
dataset_val = u.get_dataset(coco_path, 'val')
gen_val = prepare(dataset_val, epochs, batch_size, input_shape, output_shape)

save_all(gen_val, dest_path, 'val_output')

#  fuu = next(gen_val)
#  import ipdb; ipdb.set_trace()

# train dataset
dataset_train = u.get_dataset(coco_path, 'train')
gen_train = prepare(
    dataset_train, epochs, batch_size, input_shape, output_shape)

callback = ModelSaveBestAvgAcc(
    filepath="model-{epoch:02d}-{avgacc:.2f}.hdf5",
    verbose=True, cond=filter_val_f1score
)

save_all(gen_train, dest_path, 'train_output')

print('fine')
