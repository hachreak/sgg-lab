
"""Coco-a dataset.

@deprecated in favor of Open Images Dataset V5
"""

#  import xml.etree.ElementTree as ET
#  import os
import numpy as np

from copy import deepcopy

from . import load_json


def get_annotations(filepath, types=None):
    """Get annotations."""
    types = types or ['1', '2', '3']
    cocoa = load_json(filepath)
    ann = []
    for t in types:
        ann.extend(cocoa['annotations'][t])
    return ann


def get_verbs(filepath):
    """Get verbs."""
    vvn = load_json(filepath)
    return {
        'actions': {f['id']: f for f in vvn['visual_actions']},
        'adverbs': {f['id']: f for f in vvn['visual_adverbs']},
    }


def get_interactions(annotations, image_id):
    return [x for x in annotations if x['image_id'] == image_id]


def translate_ids(annotations, verbs):
    res = []
    for ann in deepcopy(annotations):
        ann['t_adverbs'] = {
            id_: verbs['adverbs'][id_] for id_ in ann['visual_adverbs']
        }
        ann['t_actions'] = {
            id_: verbs['actions'][id_] for id_ in ann['visual_actions']
        }
        res.append(ann)
    return res


def ids2names(annotations, verbs, coco=None):
    res = []
    for ann in deepcopy(annotations):
        ann['n_adverbs'] = [
            verbs['adverbs'][id_]['name'] for id_ in ann['visual_adverbs']
        ]
        ann['n_actions'] = {
            verbs['actions'][id_]['name'] for id_ in ann['visual_actions']
        }
        if coco:
            try:
                ann['n_subject'] = coco.loadAnns(ann['subject_id'])[0]
            except KeyError:
                pass
            try:
                ann['n_object'] = coco.loadAnns(ann['object_id'])[0]
            except KeyError:
                pass
        res.append(ann)
    return res


def get_instance_segmented(subject, coco):
    """Get a subject/object segmentation inside the image."""
    # get mask and color
    img = coco.annToMask(subject).copy()
    r, g, b = _get_color(subject['category_id'])
    # apply color to a RGB mask
    img = np.repeat(img, 3, axis=1).reshape(img.shape + (3,))
    img = np.transpose(img, (2, 0, 1))
    img[0][img[0] == 1] = r
    img[1][img[1] == 1] = g
    img[2][img[2] == 1] = b
    img = np.transpose(img, (1, 2, 0))
    return img


def _get_color(category_id):
    """Get unique color by category id."""
    red = category_id % 255
    channel_2 = (category_id // 255)
    green = channel_2 % 255
    channel_3 = (channel_2 // 255)
    blue = channel_3 % 255
    return red, green, blue


def get_subcoco(ann_path, coco_path):
    """Get sub dataset."""
    ann = get_annotations(ann_path)
    coco = load_json(coco_path)
    # get annotation ids to extract from coco
    image_ids = set([a['image_id'] for a in ann])
    #  cat_ids = set([a['category_id'] for a in ann])
    #      [a['subject_id'] for a in ann] +
    #      [a['object_id'] for a in ann if a['object_id'] != -1]
    #  )
    c_images = [i for i in coco['images'] if i['id'] in image_ids]
    c_ann = [i for i in coco['annotations'] if i['image_id'] in image_ids]

    import ipdb; ipdb.set_trace()
    coco['images'] = c_images
    coco['annotations'] = c_ann
    return coco
