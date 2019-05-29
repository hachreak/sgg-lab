
"""Utils."""

import scipy
import numpy as np

from keras import models, layers
from keras.applications import resnet50

from sgg_lab.datasets.coco import CocoDataset, join_masks
from sgg_lab import datasets as ds
from sgg_lab.losses.focal_loss import binary_focal_loss


def prepare(dataset, epochs, batch_size, input_shape, output_shape):
    stream = ds.epochs(dataset.image_ids, epochs)
    stream = ds.stream(check, stream)
    stream = ds.stream(
        lambda x: (dataset.load_image(x), dataset.load_output(x)),
        stream)
    stream = ds.stream(ds.apply_to_x(ds.resize(input_shape)), stream)
    stream = ds.stream(ds.apply_to_y(ds.resize(output_shape)), stream)
    #  stream = ds.stream(ds.apply_to_x(check), stream)

    stream = ds.bufferize(stream, size=20)

    batch = ds.stream_batch(stream, size=batch_size)
    batch = ds.stream(ds.apply_to_y(
        lambda x: ds.image2mask(x).reshape(x.shape + (1,))), batch)

    return batch


class NRCocoDataset(CocoDataset):

    def load_output(self, image_id):
        masks = self.load_mask(image_id)[0]
        return join_masks(masks).astype('uint8')


def load_model(path):
    return models.load_model(
        path, custom_objects={'focal_loss': binary_focal_loss()}
    )


def get_model(input_shape):
    model = resnet50.ResNet50(include_top=False, input_shape=input_shape)
    output = layers.Conv2D(1, (1, 1), activation='sigmoid')(model.output)
    return models.Model(inputs=model.inputs, outputs=output)


def resize_mask(mask, shape):
    new_shape = [s2//s1 for (s1, s2) in zip(mask.shape, shape)]
    return scipy.ndimage.zoom(mask, zoom=new_shape, order=0)


def check(x):
    #  import ipdb; ipdb.set_trace()
    print(x)
    return x


def get_dataset(coco_path, type_):
    # validation dataset
    dataset = NRCocoDataset()
    dataset.load_coco(coco_path, type_)
    dataset.prepare()
    return dataset


def get_img_cleaned(model, imgs):
    #  model = get_model('./model-03-0.90.hdf5')
    res = model.predict(imgs)
    res[res < 0.5] = 0
    res[res >= 0.5] = 255
    res = res.astype('uint8').reshape(res.shape[:3])

    masks = np.array([resize_mask(r, imgs.shape[1:3]) for r in res])

    imgs = imgs.copy()
    imgs[masks == 0] = 0
    return imgs
