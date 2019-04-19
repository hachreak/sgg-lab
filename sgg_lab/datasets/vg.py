
"""Visual Genome dataset."""

import json


def get_objects(filename):
    with open(filename, 'r') as f:
        objs = json.load(f)

    for obj in objs:
        yield obj
