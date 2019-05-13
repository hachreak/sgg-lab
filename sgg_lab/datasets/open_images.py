
"""Open Images Dataset."""

import os

from copy import deepcopy
from collections import defaultdict

from . import load_csv, save_json


#  OBJS_LABELS = 'challenge-2019-classes-description-500.csv'
#  OBJS_LABELS_SEGM = 'challenge-2019-classes-description-segmentable.csv'

VRD_OBJS_LABELS = 'challenge-2019-classes-vrd.csv'
VRD_RELS_LABELS = 'challenge-2019-relationships-description.csv'
VRD_TRAIN = 'challenge-2019-train-vrd.csv'
VRD_VALID = 'challenge-2019-validation-vrd.csv'

OD_BBOX_TRAIN = 'challenge-2019-train-detection-bbox.csv'


def get_relationship_triplets(filename):
    """Get visual relationships."""
    vrds = defaultdict(list)
    for r in load_csv(filename):
        vrds[r[0]].append({
            "image_id": r[0],
            "relationship": r[11],
            "objects": [
                {
                    "label": r[1],
                    "bbox": [r[3], r[5], r[4], r[6]]
                },
                {
                    "label": r[2],
                    "bbox": [r[7], r[9], r[8], r[10]]
                },
            ]
        })
    return vrds


def get_mask_image(images_path):
    """Convert in a image file."""
    filenames = {f.rsplit('_', 1)[0]: f for f in os.listdir(images_path)}

    def f(image_id, label_id):
        base_name = '{0}_{1}'.format(image_id, label_id.replace('/', ''))
        return filenames[base_name]
    return f


def stream_vrds(vrds, get_image_path):
    for k, v in vrds.items():
        v = deepcopy(v)
        pairs = []
        for pair in v:
            founds = 0
            for o in pair['objects']:
                try:
                    o['path'] = get_image_path(k, o['label'])
                    founds += 1
                except KeyError:
                    pass
            if len(pair['objects']) == founds:
                pairs.append(pair)
        if len(pairs) > 0:
            yield k, pairs


def stream_vrds_to_file(filename, stream):
    """Save stream vrds inside json file."""
    save_json(filename, {k: v for (k, v) in stream})


#  def get_vrd(filename):
#      """Get visual relationships."""
#      vrds = defaultdict(list)
#      for r in load_csv(filename):
#          vrds[r[0]].append({
#              "relationship": r[2],
#              "objects": [
#                  {
#                      "label": r[0],
#                  },
#                  {
#                      "label": r[1],
#                  },
#              ]
#          })
#      return vrds


#  def get_seg_bbox(filename):
#      """Get segmentation label and bbox."""
#      vrds = defaultdict(list)
#      for r in load_csv(filename):
#          vrds[r[0]].append({
#              "relationship": r[11],
#              "objects": [
#                  {
#                      "label": r[1],
#                      "bbox": [r[3], r[5], r[4], r[6]]
#                  },
#                  {
#                      "label": r[2],
#                      "bbox": [r[7], r[9], r[8], r[10]]
#                  },
#              ]
#          })
#      return vrds


def translate(filename):
    """Translate label."""
    labels = {k: v for [k, v] in load_csv(filename, skip_headers=False)}
    keys = labels.keys()

    def f(label_id):
        return labels[label_id]

    def index(label_id):
        return keys.index(label_id)

    return f, index, len(labels.keys())
