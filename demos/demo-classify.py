
import cv2
import numpy as np
import tensorflow as tf

from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint


from sgg_lab.datasets import pic
from sgg_lab import datasets as ds
from sgg_lab.nets import vgg_cls


def load_shapes(images, colorMap):
    """Load image and extract from them the shapes and category ids."""
    for img in images:
        insImg = cv2.imread(img['instance'], cv2.IMREAD_GRAYSCALE)
        semanticImg = cv2.imread(img['semantic'], cv2.IMREAD_GRAYSCALE)
        orig = cv2.imread(img['img'])
        for img, sem, ins in pic.extract_shape(
                insImg, semanticImg, orig, colorMap):
            yield img, sem


def get_dataset(filenames, epochs, colorMap, batch_size, output_shape):
    shapes = load_shapes(ds.epochs(filenames, epochs), colorMap)
    shapes = ds.stream(ds.apply_to_x(
        lambda x: cv2.resize(x, input_shape[:2])
    ), shapes)
    shapes = ds.stream(ds.apply_to_y(
        lambda x: to_categorical(x, output_shape)
    ), shapes)
    shapes = ds.stream_batch(
        shapes, lambda x, y: [np.array(x), np.array(y)], batch_size)
    return shapes


def config_tensorflow():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


path = '/media/hachreak/Magrathea/datasets/pic2018'
colorMap = np.load('segColorMap.npy')
input_shape = (300, 300, 3)
output_shape = len(pic.relationships(os.path.join(
    path, 'categories_list', 'relation_categories.json'
)))
epochs = 40
batch_size = 8

training = pic.get_filenames(path, 'train')
validati = pic.get_filenames(path, 'val')

shapes = get_dataset(training, epochs, colorMap, batch_size, output_shape)
shapes_val = get_dataset(validati, epochs, colorMap, batch_size, output_shape)

callback = ModelCheckpoint(
    filepath="model-{epoch:02d}-{val_acc:.2f}.hdf5",
    monitor="val_acc",
    save_best_only=True,
    save_weights_only=False,
    mode="max", verbose=1
)

with tf.Session(config=config_tensorflow()):

    config_tensorflow()

    model = vgg_cls.get_model(input_shape, output_shape)
    model.compile(
        optimizer=Adam(lr=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit_generator(
        shapes,
        steps_per_epoch=len(training) // batch_size,
        epochs=epochs,
        validation_data=shapes_val,
        validation_steps=len(validati) // batch_size,
        callbacks=[callback]
    )
    print('fine')
