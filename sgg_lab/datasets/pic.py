
"""PIC 2018 dataset."""

import numpy as np
import os

from . import load_json


def relationships(path):
    """Get relationship names."""
    return load_json(path)


def get_relations(stream):
    """Get relation dictionary."""
    return {obj['name']: obj['relations'] for obj in stream}


def get_filenames(path, type_, random=True):
    img_path = os.path.join(path, 'image', type_)
    instance_path = os.path.join(path, 'segmentation', type_, 'instance')
    semantic_path = os.path.join(path, 'segmentation', type_, 'semantic')
    paths = os.listdir(img_path)
    if random:
        np.random.shuffle(paths)
    return [{
        "img": os.path.join(img_path, f),
        "instance": os.path.join(
            instance_path, f.replace('.jpg', '.png')),
        "semantic": os.path.join(
            semantic_path, f.replace('.jpg', '.png')),
        "name": f,
    } for f in paths]


def extract_shape(insLabel, segLabel, oriImg, colorMap):
    semantic = np.unique(segLabel)
    semantic = list(semantic[semantic != 0])
    for semanticId in semantic:
        thisCateInsImg = insLabel.copy()
        thisCateInsImg[segLabel != semanticId] = 0
        insId = np.unique(thisCateInsImg)
        insId = list(insId[insId != 0])
        for ins in insId:
            visualize_ins = np.zeros_like(oriImg)
            try:
                visualize_ins[insLabel == ins, ...] = colorMap[semanticId + 1]
                yield visualize_ins, semanticId, ins
            except IndexError:
                # some instance image are wrong!
                pass
