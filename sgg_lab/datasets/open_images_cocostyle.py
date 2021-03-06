
"""Open Images Dataset."""

import cv2
import os
import numpy as np

from mrcnn.utils import Dataset
from collections import defaultdict

from . import load_csv


SEGM_TRAIN = 'challenge-2019-train-segmentation-masks.csv'
SEGM_VALID = 'challenge-2019-validation-segmentation-masks.csv'


def count_images(path, types=['.jpg']):
    """Count how many images inside the dataset."""
    return len([
        f for f in os.listdir(path) if os.path.splitext(f)[1] in types
    ])


def get_segmentation_masks(filename):
    """Get visual relationships."""
    segm = defaultdict(dict)
    for r in load_csv(filename):
        segm[r[1]][r[2]] = {
            'path': r[0],
            'boxid': r[3],
            'bbox': [r[4], r[6], r[5], r[7]],
            'iou': r[8]
        }
    return segm


def get_classes(filename):
    """Get list of classes in format compatible with Dataset add_class()."""
    return [
        {'source': 'shapes', 'class_id': index, 'class_name': k}
        for (index, [k, v])
        in enumerate(load_csv(filename, skip_headers=False), start=1)
    ]


def get_image_path(image_path, image_id):
    """Build image path."""
    return os.path.join(image_path, '{0}.jpg'.format(image_id))


class OIDataset(Dataset):

    def __init__(self, dataset_filename, image_path, mask_path, classes):
        super(OIDataset, self).__init__(self)
        self._dataset = get_segmentation_masks(dataset_filename)
        self._image_path = image_path
        self._mask_path = mask_path
        # load classes
        for kwargs in classes:
            self.add_class(**kwargs)
        # load images
        for k in self._dataset.keys():
            self.add_image(
                source='shapes', image_id=k,
                path=get_image_path(self._image_path, k)
            )
        self._class_names = [c['class_name'] for c in classes]

    def load_image(self, image_id):
        img_path = get_image_path(
            self._image_path, self.image_info[image_id]['id']
        )
        #  print("img path: {0}".format(img_path))
        return cv2.imread(img_path)

    def load_mask(self, image_id):
        key = self.image_info[image_id]['id']
        # get masks and class ids
        masks = []
        ids = []
        #  import ipdb; ipdb.set_trace()
        for k, v in self._dataset[key].items():
            mask_path = os.path.join(self._mask_path, v['path'])
            #  print("mask path: ", os.path.join(self._mask_path, v['path']))
            masks.append(cv2.imread(
                mask_path, cv2.IMREAD_GRAYSCALE
            ))
            ids.append(self._class_names.index(k))
        masks = np.stack(masks, axis=-1)
        # return masks and class id for each instance
        return masks.astype(np.bool), np.array(ids)
