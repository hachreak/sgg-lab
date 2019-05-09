
"""Coco-a dataset."""

#  import xml.etree.ElementTree as ET
#  import os
#  import numpy as np

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
                ann['n_subject'] = coco.loadAnns([ann['subject_id']])[0]
            except KeyError:
                pass
            try:
                ann['n_object'] = coco.loadAnns([ann['object_id']])[0]
            except KeyError:
                pass
        res.append(ann)
    return res
