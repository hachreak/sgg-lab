
import cv2
import numpy as np
import tensorflow as tf
import itertools
import os

from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback

from sgg_lab.datasets import pic
from sgg_lab import datasets as ds
from sgg_lab.nets import vgg_multi_cls


def _dataset_lenght(relations):
    counter = 0
    for x in relations.values():
        rels = []
        for y in x:
            if y['object'] < y['subject']:
                rels.append((y['object'], y['subject'], y['relation']))
            else:
                rels.append((y['subject'], y['object'], y['relation']))
        counter += len(set(rels))
    return counter


def _get_relation(sbj_id, obj_id, relations):
    return [
        r for r in relations
        if r['subject'] in [sbj_id, obj_id] and r['object'] in [sbj_id, obj_id]
    ]


def load_shapes(images, relations, colorMap):
    """Load image and extract from them the shapes and category ids."""
    for img in images:
        insImg = cv2.imread(img['instance'], cv2.IMREAD_GRAYSCALE)
        semanticImg = cv2.imread(img['semantic'], cv2.IMREAD_GRAYSCALE)
        orig = cv2.imread(img['img'])
        # get all shapes + semantic ids + instance ids
        shapes = list(pic.extract_shape(insImg, semanticImg, orig, colorMap))
        # skip if there are no shapes (instance img have wrong size!)
        if len(shapes) > 0:
            imgs, sem, ins = zip(*[res for res in shapes])
            # combine instance ids
            for i, j in itertools.combinations(ins, 2):
                rel = _get_relation(i, j, relations[img['name']])
                if len(rel) > 0:
                    img_merged = imgs[ins.index(i)] + imgs[ins.index(j)]
                    labels = list(set([r['relation'] for r in rel]))
                    if len(labels) < 2:
                        labels.append(0)
                    yield (np.copy(img_merged), labels)


def get_dataset(filenames, relations, epochs, colorMap, batch_size,
                input_shape, output_shape):
    shapes = load_shapes(ds.epochs(filenames, epochs), relations, colorMap)
    shapes = ds.stream(ds.apply_to_x(
        lambda x: cv2.resize(x, input_shape[:2])
    ), shapes)
    shapes = ds.stream(ds.apply_to_y(
        lambda x: np.array(
            [to_categorical(xv, output_shape) for xv in x]
        ).sum(axis=0)
    ), shapes)
    shapes = ds.stream_batch(
        shapes,
        lambda x, y: [np.array(x), generate_y(np.array(y))], batch_size)
    return shapes


def generate_y(y):
    """Generate negative examples."""
    return [y[:, i] for i in range(y.shape[1])]


def config_tensorflow():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


def average_acc(batch, logs):
    print('\nAverage: {0}'.format(
        np.average([v for k, v in logs.items() if k.endswith('accuracy')])
    ))


path = '/media/hachreak/Magrathea/datasets/pic2018'
colorMap = np.load('segColorMap.npy')
input_shape = (300, 300, 3)
output_shape = len(pic.relationships(os.path.join(
    path, 'categories_list', 'relation_categories.json'
)))
epochs = 40
batch_size = 16
verbose = 1

training = pic.get_filenames(path, 'train')
validati = pic.get_filenames(path, 'val')

relations_train = pic.get_relations(pic.get_objects(
    os.path.join(path, 'relation', 'relations_train.json')))

relations_val = pic.get_relations(pic.get_objects(
    os.path.join(path, 'relation', 'relations_val.json')))

shapes = get_dataset(
    training, relations_train, epochs, colorMap, batch_size,
    input_shape, output_shape)
shapes_val = get_dataset(
    validati, relations_val, epochs, colorMap, batch_size,
    input_shape, output_shape)

callback = ModelCheckpoint(
    filepath="model-{epoch:02d}-{val_loss:.2f}.hdf5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    mode="min", verbose=1
)


#  fuu = next(shapes)
#  import ipdb; ipdb.set_trace()
with tf.Session(config=config_tensorflow()):

    config_tensorflow()

    model = vgg_multi_cls.get_model(input_shape, output_shape)
    model.compile(
        optimizer=Adam(lr=0.0001),
        loss=["binary_crossentropy" for i in range(output_shape)],
        metrics=['binary_accuracy']
    )
    model.fit_generator(
        shapes,
        steps_per_epoch=_dataset_lenght(relations_train) // batch_size,
        epochs=epochs,
        validation_data=shapes_val,
        validation_steps=_dataset_lenght(relations_val) // batch_size,
        callbacks=[callback, LambdaCallback(on_batch_end=average_acc)],
        verbose=verbose
    )
    print('fine')
