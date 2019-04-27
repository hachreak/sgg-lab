
import cv2
import numpy as np
import tensorflow as tf
import itertools
import os

from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import backend as K, losses

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
    #  def fu(x):
    #      ba = list(np.array(
    #          [to_categorical(xv, output_shape) for xv in x]
    #      ).sum(axis=0))
    #      import ipdb; ipdb.set_trace()
    #      return ba
    shapes = ds.stream(ds.apply_to_y(
        lambda x: np.array(
            [to_categorical(xv, output_shape) for xv in x]
        ).sum(axis=0)
    ), shapes)
    #  shapes = ds.stream(ds.apply_to_y(generate_neg), shapes)
    shapes = ds.stream_batch(
        shapes,
        lambda x, y: [np.array(x), generate_neg(np.array(y))], batch_size)
    return shapes


def generate_neg(y):
    """Generate negative examples."""
    #  # get positive indices
    #  pos_indices = np.where(y == 1)[0]
    #  # generate negative indices
    #  rands = list(range(0, y.shape[0]))
    #  for i, index in enumerate(pos_indices):
    #      del rands[index - i]
    #  neg_indices = np.random.choice(rands, size=pos_indices.shape[0])
    # convert zero values to -1
    #  y[np.where(y == 0)[0]] = -1
    #  # and set negative as zero
    #  y[neg_indices] = 0
    #  import ipdb; ipdb.set_trace()
    #  return [np.array([v]) for v in y.tolist()]
    return [y[:, i] for i in range(y.shape[1])]


def config_tensorflow():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


#  def loss_fun(y_true, y_pred):
def loss_fun(y_pred, y_true):
    #  # choose X pos, X neg
    #  condition = K.not_equal(y_true, -1)
    #  indices = tf.where(condition)
    #  # extract sub-matrix
    #  yp = tf.gather_nd(y_pred, indices)
    #  yt = tf.gather_nd(y_true, indices)
    #  return losses.binary_crossentropy(yt, yp)
    #  import ipdb; ipdb.set_trace()
    return losses.binary_crossentropy(y_true, y_pred)


path = '/media/hachreak/Magrathea/datasets/pic2018'
colorMap = np.load('segColorMap.npy')
input_shape = (300, 300, 3)
output_shape = len(pic.semantic_names)
epochs = 40
batch_size = 8
verbose = 2

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
    filepath="model-{epoch:02d}-{val_acc:.2f}.hdf5",
    monitor="val_acc",
    save_best_only=True,
    save_weights_only=False,
    mode="max", verbose=1
)

#  fuu = next(shapes)
#  import ipdb; ipdb.set_trace()
with tf.Session(config=config_tensorflow()):

    config_tensorflow()

    model = vgg_multi_cls.get_model(input_shape, output_shape)
    model.compile(
        optimizer=Adam(lr=0.0001),
        loss=["binary_crossentropy" for i in range(output_shape)],
        metrics=['accuracy']
    )
    model.fit_generator(
        shapes,
        steps_per_epoch=_dataset_lenght(relations_train) // batch_size,
        epochs=epochs,
        validation_data=shapes_val,
        validation_steps=_dataset_lenght(relations_val) // batch_size,
        callbacks=[callback], verbose=verbose
    )
    print('fine')
