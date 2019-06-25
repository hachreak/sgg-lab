
"""Train Noise Removal on coco."""

import utils as u

import numpy as np
import keras_metrics as km
from keras import optimizers as opt, models, layers
from keras.applications import resnet50
#  from joblib import parallel_backend, Parallel, delayed

from sgg_lab import datasets as ds
from sgg_lab.losses.focal_loss import \
    adaptive_binary_focal_loss as binary_focal_loss
from sgg_lab.callbacks import ModelSaveBestAvgAcc, filter_val_f1score
from sgg_lab.layers.gaussianconv import GaussianConv2D as Conv2D


def imageid2categoricalpixel(dataset, input_shape):
    """Convert image_id to categorical pixel."""
    resizer = ds.resize(input_shape)

    def f(image_id):
        cat = dataset.to_categorical(image_id)
        # expand values to be better separable after resizing
        cat[cat > 0] = 255
        #
        class_ids = set(dataset.get_class_ids(image_id))
        result = np.zeros(list(input_shape[:2]) + [cat.shape[2]])
        for i in class_ids:
            result[:, :, i] = resizer(cat[:, :, i])
        # fix value generated from resizing
        result[result <= 127] = 0
        result[result > 127] = 1
        return result
    return f


def prepare(dataset, epochs, batch_size, input_shape, output_shape):
    i2c = imageid2categoricalpixel(dataset, input_shape)
    stream = ds.epochs(dataset.image_ids, epochs)
    stream = ds.stream(
        lambda x: (dataset.load_image(x), i2c(x)),
        stream)
    stream = ds.stream(ds.apply_to_x(ds.resize(input_shape)), stream)

    def adapt_y(x, y):
        y = np.array(y).transpose((3, 0, 1, 2))
        y = y.reshape(y.shape + (1,))
        return np.array(x), [v for v in y]

    batch = ds.stream_batch(stream, size=batch_size, fun=adapt_y)

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

    output7 = [
        layers.Conv2D(1, (1, 1), activation='sigmoid')(mul6)
        for i in range(0, num_classes)
    ]

    return models.Model(inputs=model.inputs, outputs=output7)


coco_path = '/media/hachreak/Magrathea/datasets/coco/coco'
model_path = ''
epochs = 100
batch_size = 3
input_shape = (320, 320, 3)
output_shape = (10, 10)

action = 'train'
# action = 'evaluate'

# validation dataset
dataset_val = u.get_dataset(coco_path, 'val')
gen_val = prepare(dataset_val, epochs, batch_size, input_shape, output_shape)

#  fuu = next(gen_val)
#  import ipdb; ipdb.set_trace()

# train dataset
dataset_train = u.get_dataset(coco_path, 'train')
gen_train = prepare(
    dataset_train, epochs, batch_size, input_shape, output_shape)

callback = ModelSaveBestAvgAcc(
    filepath="model-{epoch:02d}-{avgacc:.2f}.hdf5",
    verbose=True, cond=filter_val_f1score
)

losses = []
for i in range(0, dataset_val.num_classes):
    losses.append(binary_focal_loss(gamma=2.))

model = get_model(input_shape, dataset_val.num_classes)
#  import ipdb; ipdb.set_trace()

model.compile(
    optimizer=opt.Adam(lr=1e-4),
    loss=losses,
    metrics=['accuracy', km.binary_precision(), km.binary_recall(),
             km.binary_f1_score(), km.false_positive()]
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
