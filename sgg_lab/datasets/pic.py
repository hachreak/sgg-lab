
"""PIC 2018 dataset."""

import json
import numpy as np
import os


semantic_names = [
    "background",
    "human",
    "floor",
    "bed",
    "window",
    "cabinet",
    "door",
    "table",
    "potting-plant",
    "curtain",
    "chair",
    "sofa",
    "shelf",
    "rug",
    "lamp",
    "fridge",
    "stairs",
    "pillow",
    "kitchen-island",
    "sculpture",
    "sink",
    "document",
    "painting/poster",
    "barrel",
    "basket",
    "poke",
    "stool",
    "clothes",
    "bottle",
    "plate",
    "cellphone",
    "toy",
    "cushion",
    "box",
    "display",
    "blanket",
    "pot",
    "nameplate",
    "banners/flag",
    "cup",
    "pen",
    "digital",
    "cooker",
    "umbrella",
    "decoration",
    "straw",
    "certificate",
    "food",
    "club",
    "towel",
    "pet/animals",
    "tool",
    "household-appliances",
    "pram",
    "car/bus/truck",
    "grass",
    "vegetation",
    "water",
    "ground",
    "road",
    "street-light",
    "railing/fence",
    "stand",
    "steps",
    "pillar",
    "awnings/tent",
    "building",
    "mountrain/hill",
    "stone",
    "bridge",
    "bicycle",
    "motorcycle",
    "airplane",
    "boat/ship",
    "balls",
    "swimming-equipment",
    "body-building-apparatus",
    "gun",
    "smoke",
    "rope",
    "amusement-facilities",
    "prop",
    "military-equipment",
    "bag",
    "instruments"
]


def get_objects(filename):
    with open(filename, 'r') as f:
        objs = json.load(f)

    for obj in objs:
        yield obj


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
