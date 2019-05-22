
"""Build training/validation file.

Extract to file the subset of the image/segmentation/relationship available.

Not all segmentation images has corresponding relationships and viceversa.
"""

import os

from sgg_lab.datasets import open_images


def get_stream(base, subdir, filename):
    get_mi = open_images.get_mask_image(
        os.path.join(base, subdir))
    vrds = open_images.get_relationship_triplets(
        os.path.join(base, '2019', filename))
    return open_images.stream_vrds(vrds, get_mi)


BASE = '/media/hachreak/Magrathea/datasets/open-images-v5'

train = get_stream(BASE, 'challenge-2019-train-masks', open_images.VRD_TRAIN)
open_images.stream_vrds_to_file(__file__.replace('.py', '_train.json'), train)

valid = get_stream(
    BASE, 'challenge-2019-validation-masks', open_images.VRD_VALID)
open_images.stream_vrds_to_file(__file__.replace('.py', '_valid.json'), valid)
