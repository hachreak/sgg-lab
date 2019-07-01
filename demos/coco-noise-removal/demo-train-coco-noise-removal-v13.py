
"""Train Noise Removal on coco."""

import utils as u

import random
import keras_metrics as km
import numpy as np

from keras import optimizers as opt, models, layers
from keras.applications import resnet50

from sgg_lab import datasets as ds
from sgg_lab.losses.focal_loss import \
    adaptive_binary_focal_loss as binary_focal_loss
from sgg_lab.callbacks import ModelSaveBestAvgAcc, filter_val_f1score


def random_id(dataset, image_ids):
    """Get an image_id taking in account all except the one in input."""
    indices = range(0, len(dataset.image_info))
    for i in image_ids:
        indices.remove(i)
    return random.choice(indices)


def apply_aug(img_o, mask_o, img_r, mask_r):
    # remove random img and original img segmentation
    img_r[mask_o > 0] = 0
    img_r[mask_r > 0] = 0
    # remove original background
    img_o[mask_o == 0] = 0
    # apply new background
    return img_o + img_r, mask_o + mask_r


def load_augmented_image(dataset, resizer):
    def f(image_id):
        # get original
        img_o = resizer(dataset.load_image(image_id).copy())
        mask_o = resizer(dataset.load_output(image_id).copy())
        rand_ids = [image_id]
        for i in range(0, 5):
            # get random
            rand_ids.append(random_id(dataset, rand_ids))
            img_r = resizer(dataset.load_image(rand_ids[-1]).copy())
            mask_r = resizer(dataset.load_output(rand_ids[-1]).copy())
            img_o, _ = apply_aug(img_o, mask_o, img_r, mask_r)
            mask_o = np.sum(img_o, axis=2)
        return img_o
    return f


def resize_all(output_shape):
    resizer_orig = ds.resize(u.mul_shape(output_shape, 32))

    def f(input_):
        input_ = resizer_orig(input_)
        return input_,

    return f


def prepare(dataset, epochs, batch_size, input_shape, output_shape):
    resizer = ds.resize(input_shape[:2])
    lai = load_augmented_image(dataset, resizer)

    stream = ds.epochs(dataset.image_ids, epochs)
    stream = ds.stream(
        lambda x: (lai(x), dataset.load_output(x)),
        stream)
    #  stream = ds.stream(ds.apply_to_x(ds.resize(input_shape)), stream)
    stream = ds.stream(ds.apply_to_y(resize_all(output_shape)), stream)

    stream = ds.bufferize(stream, size=20)

    batch = ds.stream_batch(stream, size=batch_size, fun=ds.pack_elements)
    batch = ds.stream(ds.apply_to_y(ds.apply_to_xn(
        lambda x: ds.image2mask(x).reshape(x.shape + (1,)))), batch)

    return batch


def mul_layer(back_layer, last):
    back = back_layer.output
    upsample = layers.Conv2DTranspose(
        int(last.shape[3]), (2, 2), strides=(2, 2))(last)
    to1024 = layers.Conv2D(
        int(back.shape[3]), (1, 1), activation='elu')(upsample)
    mul = layers.Multiply()([back, to1024])
    l1x1 = layers.Conv2D(1, (1, 1), activation='sigmoid')(mul)
    return mul, l1x1


def get_model(input_shape):
    model = resnet50.ResNet50(include_top=False, input_shape=input_shape)

    mul2, output2 = mul_layer(model.get_layer('activation_40'), model.output)
    mul3, output3 = mul_layer(model.get_layer('activation_22'), mul2)
    mul4, output4 = mul_layer(model.get_layer('activation_10'), mul3)
    mul5, output5 = mul_layer(model.get_layer('activation_1'), mul4)
    mul6, output6 = mul_layer(model.get_layer('input_1'), mul5)

    return models.Model(
        inputs=model.inputs,
        outputs=[output6])


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
#  img_o = load_augmented_image(dataset_val, ds.resize((320,320)), 0)
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
for i in range(0, 1):
    losses.append(binary_focal_loss(gamma=2.))

model = get_model(input_shape)
model.compile(
    optimizer=opt.Adam(lr=1e-4),
    loss=losses,
    metrics=['accuracy', km.binary_precision(),
             km.binary_recall(), km.binary_f1_score()]
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
