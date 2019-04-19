
import cv2
import os
import numpy as np
import tensorflow as tf

from PIL import Image
from sklearn.manifold import TSNE
from keras.applications import vgg19
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, \
        BatchNormalization

from sgg_lab.datasets import pic
from sgg_lab import datasets as ds


def get_dataset(filenames, epochs, colorMap, batch_size, output_shape):
    shapes = pic.load_shapes(ds.epochs(filenames, epochs), colorMap)
    shapes = ds.stream(ds.apply_to_x(
        lambda x: cv2.resize(x, input_shape[:2])
    ), shapes)
    #  shapes = ds.stream(ds.apply_to_x(np.array), shapes)
    #  shapes = ds.stream(ds.apply_to_y(np.array), shapes)
    shapes = ds.stream(ds.apply_to_y(
        lambda x: to_categorical(x, output_shape)
    ), shapes)
    shapes = ds.stream_batch(
        shapes,
        lambda x, y: (
            np.array(x),
            #  to_categorical(y.reshape(y.shape + (1,)), len(colorMap))),
            np.array(y)
        ),
        batch_size)
    return shapes


def _set_readonly(model, until=None):
    """Make a model weights readonly."""
    for layer in model.layers[:until]:
        layer.trainable = False
    return model


def _classification(model, output_shape):
    """Add classification layers."""
    x = model.output
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(output_shape, activation='softmax')(x)

    return Model(inputs=model.input, outputs=x)


def config_tensorflow():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


path = '/media/hachreak/Magrathea/datasets/pic2018'
colorMap = np.load('segColorMap.npy')
input_shape = (300, 300, 3)
output_shape = len(pic.semantic_names)
epochs = 20
batch_size = 4

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
    model = vgg19.VGG19(
        include_top=False, weights='imagenet', input_shape=input_shape
    )

    config_tensorflow()

    model = _set_readonly(model, 18)
    model = _classification(model, output_shape)
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
    import ipdb; ipdb.set_trace()
    print('fine')
