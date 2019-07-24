from __future__ import absolute_import
from __future__ import print_function

import itertools
import numpy as np

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda
from keras.optimizers import RMSprop
from keras import backend as K

from sgg_lab import datasets as ds
from sgg_lab.callbacks import ModelSaveBestAvgAcc


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    #  x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    #  x = Dropout(0.1)(x)
    x = Dense(2)(x)
    return Model(input, x)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def repeat_elem(dataset, times=1):
    for value in dataset:
        for v in itertools.repeat(value, times):
            yield v


def skip_equals(xy1, xy2):
    for xy in itertools.izip(xy1, xy2):
        (_, y1), (_, y2) = xy
        if y1 != y2:
            yield xy


def build_diff(dataset1, dataset2):
    for xy1, xy2 in itertools.izip(dataset1, dataset2):
        (x1, y1), (x2, y2) = xy1
        (x3, y3), (x4, y4) = xy2
        label = (y1 == y3 and y2 == y4) or (y1 == y4 and y2 == y3)
        #  print(y1,y2,y3,y4)
        yield (np.abs(x1 - x2), np.abs(x3 - x4)), int(label)


def build_pairs(x, y, epochs):
    stream_1 = ds.epochs(zip(x, y), epochs=len(y))
    stream_1 = ds.epochs(stream_1, epochs=epochs, random=False)

    stream_2 = repeat_elem(zip(x, y), times=len(y))
    stream_2 = ds.epochs(stream_2, epochs=epochs, random=False)

    stream = skip_equals(stream_1, stream_2)
    return stream


def get_dataset(x, y, epochs, batch_size):
    s1 = build_pairs(x, y, epochs)
    s2 = build_pairs(x, y, epochs)
    stream = build_diff(s1, s2)

    stream = ds.stream_batch(
        stream,
        fun=lambda x, y: ([v for v in np.array(x).transpose(1, 0, 2, 3)], y),
        size=batch_size)

    return stream


def len_ds(y):
    count_discard = sum([
        x ** 2 for x in np.unique(y, return_counts=True)[1]
    ])
    count_total = len(y) ** 2
    return count_total - count_discard


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:1000]
y_test = y_test[:1000]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]
epochs = 1
batch_size = 1024  # * 128

gen_train = get_dataset(x_train, y_train, epochs, batch_size)
gen_val = get_dataset(x_test, y_test, epochs, batch_size)

#  fuu = next(gen_train)
#  import ipdb; ipdb.set_trace()

# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(
    euclidean_distance,
    output_shape=eucl_dist_output_shape
)([processed_a, processed_b])

callback = ModelSaveBestAvgAcc(
    filepath="model-{epoch:02d}-{acc:.2f}.hdf5",
    verbose=True
)

model = Model([input_a, input_b], distance)
model.summary()

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
model.fit_generator(
    gen_train,
    steps_per_epoch=len_ds(y_train) // batch_size,
    epochs=epochs,
    validation_data=gen_val,
    validation_steps=len_ds(y_test) // batch_size,
    callbacks=[callback],
)

# compute final accuracy on training and test sets
import ipdb; ipdb.set_trace()
print('ok')
