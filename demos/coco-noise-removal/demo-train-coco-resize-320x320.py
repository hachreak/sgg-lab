
"""Train Noise Removal on coco."""

import utils as u

import os

from PIL import Image

from sgg_lab import datasets as ds


def prepare(dataset, input_shape, output_shape):
    resizer = ds.resize(u.mul_shape(output_shape, 32))

    stream = ds.epochs(dataset.image_ids, epochs=1)
    stream = ds.stream(
        lambda x: (dataset._img_filenames[x], resizer(dataset.load_image(x))),
        stream)

    return stream


def save_all(gen, base_path, directory):
    """Save all resized images."""
    dest = os.path.join(base_path, directory)
    if not os.path.exists(dest):
        os.makedirs(dest)
    print('process {0}'.format(dest))
    for name, img in gen:
        path = os.path.join(dest, name)
        Image.fromarray(img).save(path)


coco_path = '/media/hachreak/Magrathea/datasets/coco/coco'
input_shape = (320, 320, 3)
output_shape = (10, 10)

dest_path = '/media/hachreak/Magrathea/datasets/coco/size_{0}x{1}'.format(
    *input_shape[:2])

# val dataset
dataset_val = u.get_dataset(coco_path, 'val')
gen_val = prepare(dataset_val, input_shape, output_shape)

#  fuu = next(gen_val)
#  import ipdb; ipdb.set_trace()

save_all(gen_val, dest_path, 'val_output')

# train dataset
dataset_train = u.get_dataset(coco_path, 'train')
gen_train = prepare(dataset_train, input_shape, output_shape)

save_all(gen_train, dest_path, 'train_output')
print('fine')
