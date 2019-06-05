
"""Utils."""

import scipy
import numpy as np

from keras import models, utils

from sgg_lab.datasets.coco import CocoDataset, join_masks
from sgg_lab.losses.focal_loss import binary_focal_loss


class NRCocoDataset(CocoDataset):

    def to_categorical(self, image_id):
        masks, cls = self.load_mask(image_id)
        mask = np.zeros(masks.shape[:2] + (self.num_classes,))
        for i in range(0, masks.shape[2]):
            m = masks[:, :, i]
            mask[m > 0] += utils.to_categorical(cls[i], self.num_classes)
        return mask

    def load_output(self, image_id):
        masks, _ = self.load_mask(image_id)
        return join_masks(masks).astype('uint8')


def load_model(path):
    return models.load_model(
        path, custom_objects={'focal_loss': binary_focal_loss()}
    )


def resize_mask(mask, shape):
    new_shape = [s2//s1 for (s1, s2) in zip(mask.shape, shape)]
    return scipy.ndimage.zoom(mask, zoom=new_shape, order=0)


def check(x):
    #  import ipdb; ipdb.set_trace()
    print(x)
    return x


def get_dataset(coco_path, type_, cocodataset=None):
    cocodataset = cocodataset or NRCocoDataset
    # validation dataset
    dataset = cocodataset()
    dataset.load_coco(coco_path, type_)
    dataset.prepare()
    return dataset


def get_img_cleaned(model, imgs):
    res = model.predict(imgs)
    res[res < 0.5] = 0
    res[res >= 0.5] = 255
    res = res.astype('uint8').reshape(res.shape[:3])

    masks = np.array([resize_mask(r, imgs.shape[1:3]) for r in res])

    imgs = imgs.copy()
    imgs[masks == 0] = 0
    return imgs


def mul_shape(shape, mult):
    """Multiply shape."""
    return [s*mult for s in shape]
