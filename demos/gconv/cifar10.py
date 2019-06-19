
"""Demo."""

import keras
from keras.datasets import cifar10
#  from keras.layers import Dense, Flatten
from keras.layers import Flatten
#  from keras.layers import Conv2D, MaxPooling2D
from keras.layers import MaxPooling2D, Input
from keras.models import Model
import os

from sgg_lab.layers.gaussianconv import GaussianConv2D as Conv2D, \
    GaussianDense as Dense


batch_size = 32
num_classes = 10
epochs = 100
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# model
inputs = Input(shape=x_train.shape[1:])

x = Conv2D(32, (3, 3), padding='same', activation='relu', use_bias=False)(
    inputs, training=True)
x = Conv2D(32, (3, 3), padding='same', activation='relu', use_bias=False)(
        x, training=True)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (3, 3), padding='same', activation='relu', use_bias=False)(
        x, training=True)
x = Conv2D(64, (3, 3), padding='same', activation='relu', use_bias=False)(
        x, training=True)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)
x = Dense(512, activation='relu', use_bias=False)(x, training=True)
x = Dense(num_classes, activation='softmax', use_bias=False)(x, training=True)

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model = Model(inputs=inputs, outputs=[x])
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(
    x_train, y_train, batch_size=batch_size, epochs=epochs,
    validation_data=(x_test, y_test), shuffle=True
)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
