
"""Train Noise Removal on coco."""

import utils as u

import os
import numpy as np

from PIL import Image

from sgg_lab import datasets as ds


def resize_all(dataset, output_shape):
    out_shape_1 = u.mul_shape(output_shape, 4)
    resizer_1 = ds.resize(out_shape_1)
    resizer_2 = ds.resize(u.mul_shape(output_shape, 2))
    resizer_3 = ds.resize(output_shape)

    def f((image_id, img)):
        cat = dataset.to_categorical(image_id)
        result = np.zeros(out_shape_1 + [cat.shape[2]])
        for i in range(0, cat.shape[2]):
            result[:, :, i] = resizer_1(cat[:, :, i])
        x = resizer_1(img)
        y = resizer_2(x)
        z = resizer_3(y)

        x = x.reshape(x.shape + (1,))
        y = y.reshape(y.shape + (1,))
        z = z.reshape(z.shape + (1,))
        return result, x, y, z

    return f


def prepare(dataset, input_shape, output_shape):
    stream = ds.epochs(dataset.image_ids, epochs=1)
    stream = ds.stream(
        lambda x: (dataset._img_filenames[x], (x, dataset.load_output(x))),
        stream)
    #  stream = ds.stream(ds.apply_to_x(ds.resize(input_shape)), stream)
    stream = ds.stream(ds.apply_to_y(
        resize_all(dataset, output_shape)), stream)

    stream = ds.bufferize(stream, size=10)

    batch = ds.stream_batch(stream, size=10, fun=ds.pack_elements)
    #  batch = ds.stream(ds.apply_to_y(check), batch)
    batch = ds.stream(ds.apply_to_y(ds.apply_to_xn(
        lambda x: ds.image2mask(x))), batch)

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
            np.save(path, res)


coco_path = '/media/hachreak/Magrathea/datasets/coco/coco'
model_path = ''
epochs = 100
batch_size = 3
input_shape = (320, 320, 3)
output_shape = (10, 10)

dest_path = '/media/hachreak/Magrathea/datasets/coco/resize_{0}x{1}'.format(
    *input_shape[:2])

# val dataset
dataset_val = u.get_dataset(coco_path, 'val')
gen_val = prepare(dataset_val, input_shape, output_shape)

save_all(gen_val, dest_path, 'val_output')

# train dataset
dataset_train = u.get_dataset(coco_path, 'train')
gen_train = prepare(dataset_train, input_shape, output_shape)

save_all(gen_train, dest_path, 'train_output')
print('fine')
