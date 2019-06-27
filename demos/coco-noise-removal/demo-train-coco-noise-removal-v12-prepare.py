
"""Train Noise Removal on coco."""

import utils as u

import numpy as np
import os

#  from joblib import parallel_backend, Parallel, delayed

from sgg_lab import datasets as ds
from sgg_lab.callbacks import ModelSaveBestAvgAcc, filter_val_f1score


def imageid2categoricalpixel(dataset, input_shape):
    """Convert image_id to categorical pixel."""
    resizer = ds.resize(input_shape)

    def f(image_id):
        cat = dataset.to_categorical(image_id)
        # expand values to be better separable after resizing
        cat[cat > 0] = 255
        #
        class_ids = set(dataset.get_class_ids(image_id))
        result = np.zeros(list(input_shape[:2]) + [cat.shape[2]])
        for i in class_ids:
            result[:, :, i] = resizer(cat[:, :, i])
        # fix value generated from resizing
        result[result <= 127] = 0
        result[result > 127] = 1
        return result
    return f


def prepare(dataset, epochs, batch_size, input_shape, output_shape):
    i2c = imageid2categoricalpixel(dataset, input_shape)
    stream = ds.epochs(dataset.image_ids, epochs)
    stream = ds.stream(
        lambda x: (dataset._img_filenames[x], i2c(x)),
        stream)

    def adapt_y(x, y):
        y = np.array(y).transpose((3, 0, 1, 2))
        y = y.reshape(y.shape + (1,)).astype('int32')
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

dest_path = '/media/hachreak/Magrathea/datasets/coco/v12_resize_{0}x{1}' \
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
