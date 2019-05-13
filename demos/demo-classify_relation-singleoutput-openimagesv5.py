
import cv2
import numpy as np
import tensorflow as tf
import os

from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from sgg_lab import datasets as ds
from sgg_lab.datasets import open_images as oi
from sgg_lab.nets import vgg_cls


#  def check(fuu):
#      import ipdb; ipdb.set_trace()
#      print(fuu)
#      return fuu


def foreach(dataset, basepath, color, label):
    for img in dataset.values():
        for pair in img:
            # get obj paths
            obj1 = os.path.join(basepath, pair['objects'][0]['path'])
            obj2 = os.path.join(basepath, pair['objects'][1]['path'])
            # get colors
            c1 = color(pair['objects'][0]['label'])
            c2 = color(pair['objects'][1]['label'])
            # merge colorized shapes
            #  img1 = ds.colorize(cv2.imread(obj1), color(l1))
            #  img2 = ds.colorize(cv2.imread(obj2), color(l2))
            #  img = img1 + img2
            # get label
            idx = label(pair['relationship'])
            yield ((obj1, c1), (obj2, c2)), idx


def get_dataset(dataset, epochs, input_shape, output_shape):
    shapes = ds.epochs(dataset, epochs)
    shapes = ds.stream(ds.apply_to_x(ds.apply_to_xn(
        lambda x: ds.colorize(cv2.imread(x[0]), x[1])
    )), shapes)
    shapes = ds.stream(ds.apply_to_x(sum), shapes)
    shapes = ds.stream(ds.apply_to_x(
        lambda x: cv2.resize(x, input_shape[:2])
    ), shapes)
    shapes = ds.stream(ds.apply_to_y(
        lambda x: to_categorical(x, output_shape)
    ), shapes)
    shapes = ds.stream_batch(
        shapes,
        lambda x, y: [np.array(x), np.array(y)], batch_size)
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


path = '/media/hachreak/Magrathea/datasets/open-images-v5'
#  vrd_path = os.path.join(path, '2019', open_images.VRD_TRAIN)
path_obj_classes = os.path.join(path, '2019', oi.VRD_OBJS_LABELS)
path_valid = os.path.join(path, 'challenge-2019-validation-masks-0')
path_train = os.path.join(path, 'challenge-2019-train-masks-0')
dirname = __file__.rsplit('/', 1)[0]

translate_objs, index_objs, count_objs = oi.translate(
    os.path.join(path, '2019', oi.VRD_OBJS_LABELS))
translate_rels, index_rels, count_rels = oi.translate(
    os.path.join(path, '2019', oi.VRD_RELS_LABELS))

train = ds.load_json(os.path.join(
    dirname, 'openimages-v5-segm_train.json'
))

validation = ds.load_json(os.path.join(
    dirname, 'openimages-v5-segm_valid.json'
))

color = ds.get_color(count_objs)

input_shape = (300, 300, 3)
epochs = 40
batch_size = 9
verbose = 1

output_shape = count_rels

val_list = list(foreach(
    validation, path_valid,
    lambda x: color(index_objs(x)),
    lambda y: index_rels(y)
))

train_list = list(foreach(
    train, path_train,
    lambda x: color(index_objs(x)),
    lambda y: index_rels(y)
))

valid_dataset = get_dataset(
    val_list, epochs, input_shape, output_shape)

train_dataset = get_dataset(
    train_list, epochs, input_shape, output_shape)

callback = ModelCheckpoint(
    filepath="model-{epoch:02d}-{val_acc:.2f}.hdf5",
    monitor="val_acc",
    save_best_only=True,
    save_weights_only=False,
    mode="max", verbose=1
)

#  bar = next(train_dataset)
#  import ipdb; ipdb.set_trace()

with tf.Session(config=config_tensorflow()):

    config_tensorflow()

    model = vgg_cls.get_model(input_shape, output_shape)
    model.compile(
        optimizer=Adam(lr=0.0001),
        loss="binary_crossentropy",
        metrics=['binary_accuracy']
    )
    model.fit_generator(
        train_dataset,
        steps_per_epoch=len(train_list) // batch_size,
        epochs=epochs,
        validation_data=valid_dataset,
        validation_steps=len(val_list) // batch_size,
        callbacks=[callback],
        verbose=verbose
    )
    print('fine')
