
"""Train Noise Removal on coco."""

import utils as u

import os

from PIL import Image

from sgg_lab import datasets as ds


def prepare(dataset, input_shape):
    stream = ds.epochs(dataset.image_ids, epochs=1)
    #  stream = ds.stream(check, stream)
    stream = ds.stream(lambda x: (
        dataset._img_filenames[x], dataset.load_image(x)
    ), stream)
    stream = ds.stream(ds.apply_to_y(ds.resize(input_shape)), stream)

    return stream


def save_all(gen, base_path, directory):
    """Save all resized images."""
    dest = os.path.join(base_path, directory)
    if not os.path.exists(dest):
        os.makedirs(dest)
    print('process {0}'.format(dest))
    for name, img in gen:
        path = os.path.join(dest, name)
        print(path)
        Image.fromarray(img).save(path)


coco_path = '/media/hachreak/Magrathea/datasets/coco/coco'
model_path = ''
epochs = 100
batch_size = 3
input_shape = (320, 320, 3)

dest_path = '/media/hachreak/Magrathea/datasets/coco/resize_{0}x{1}'.format(
    *input_shape[:2])

# val dataset
dataset_val = u.get_dataset(coco_path, 'val')
gen_val = prepare(dataset_val, input_shape)

save_all(gen_val, dest_path, 'val2017')

# train dataset
dataset_train = u.get_dataset(coco_path, 'train')
gen_train = prepare(dataset_train, input_shape)

save_all(gen_train, dest_path, 'train2017')
print('fine')
