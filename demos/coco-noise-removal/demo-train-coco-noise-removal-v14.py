
"""Train Noise Removal on coco."""

import utils as u

import os
import numpy as np
from keras import optimizers as opt, models, layers
from keras.applications import resnet50
from keras.layers import Conv2D
from keras.losses import mse
#  from joblib import parallel_backend, Parallel, delayed

from sgg_lab.metrics import f1score, precision, recall
from sgg_lab import datasets as ds
from sgg_lab.losses.focal_loss import \
    adaptive_binary_focal_loss as binary_focal_loss
from sgg_lab.callbacks import ModelSaveBestAvgAcc, filter_val
#  from sgg_lab.layers.gaussianconv import GaussianConv2D as Conv2D


def load_masks(dataset, base_path):
    """Convert image_id to categorical pixel."""
    def f(image_id):
        file_path = '{0}.npz'.format(dataset._img_filenames[image_id])
        path = os.path.join(base_path, file_path)
        y = np.load(path, allow_pickle=True)
        return y
    return f


def fix_y(y):
    y = [[v['arr_0'] for v in y]]
    return [v for v in np.array(y)]  # .transpose((1, 0, 2, 3))]


def prepare(dataset, epochs, batch_size, input_shape, base_path):
    lm = load_masks(dataset, base_path)
    stream = ds.epochs(dataset.image_ids, epochs)
    stream = ds.stream(
        lambda x: (dataset.load_image(x), lm(x)),
        stream)
    #  stream = ds.stream(ds.apply_to_x(ds.resize(input_shape)), stream)

    batch = ds.stream_batch(
        stream, size=batch_size,
        fun=lambda x, y: (np.array(x), fix_y(y))
    )

    return batch


def mul_layer(back_layer, last):
    back = back_layer.output
    upsample = layers.Conv2DTranspose(
        int(last.shape[3]), (2, 2), strides=(2, 2))(last)
    to1024 = Conv2D(int(back.shape[3]), (1, 1), activation='elu')(upsample)
    mul = layers.Multiply()([back, to1024])
    return mul


def get_model(input_shape, num_classes):
    model = resnet50.ResNet50(include_top=False, input_shape=input_shape)

    mul2 = mul_layer(model.get_layer('activation_40'), model.output)
    mul3 = mul_layer(model.get_layer('activation_22'), mul2)
    mul4 = mul_layer(model.get_layer('activation_10'), mul3)
    mul5 = mul_layer(model.get_layer('activation_1'), mul4)
    mul6 = mul_layer(model.get_layer('input_1'), mul5)

    output7 = layers.Conv2D(1, (1, 1), activation='linear')(mul6)
    output7 = layers.BatchNormalization()(output7)

    return models.Model(inputs=model.inputs, outputs=output7)


coco_path = '/media/hachreak/Magrathea/datasets/coco/v14_resize_320x320'
model_path = ''
epochs = 100
batch_size = 3
input_shape = (320, 320, 3)
output_shape = (10, 10)

action = 'train'
# action = 'evaluate'

# validation dataset
dataset_val = u.get_dataset(coco_path, 'val')
gen_val = prepare(
    dataset_val, epochs, batch_size, input_shape,
    os.path.join(coco_path, 'val_output'))

#  fuu = next(gen_val)
#  import ipdb; ipdb.set_trace()

# train dataset
dataset_train = u.get_dataset(coco_path, 'train')
gen_train = prepare(
    dataset_train, epochs, batch_size, input_shape,
    os.path.join(coco_path, 'train_output'))

callback = ModelSaveBestAvgAcc(
    filepath="model-{epoch:02d}-{avgacc:.2f}.hdf5",
    verbose=True, cond=filter_val('evaluate')
)


def evaluate(y_true, y_pred):
    return 1 - mse(y_true, y_pred)


losses = []
losses.append(mse)
#  for i in range(0, 1):
#      losses.append(binary_focal_loss(gamma=2.))

#  import ipdb; ipdb.set_trace()
model = get_model(input_shape, dataset_val.num_classes)

model.compile(
    optimizer=opt.Adam(lr=1e-4),
    loss=losses,
    metrics=['accuracy', precision, evaluate]
)

model.summary()

model.fit_generator(
    gen_train,
    steps_per_epoch=len(dataset_train.image_ids) // batch_size,
    epochs=epochs,
    validation_data=gen_val,
    validation_steps=len(dataset_val.image_ids) // batch_size,
    callbacks=[callback],
    verbose=1
)

print('fine')
