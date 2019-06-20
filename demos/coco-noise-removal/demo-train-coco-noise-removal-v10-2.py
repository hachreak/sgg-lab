
"""Train Noise Removal on coco."""

import utils as u

from keras import optimizers as opt, models, layers
from keras.applications import resnet50
import keras_metrics as km

from sgg_lab import datasets as ds
from sgg_lab.losses.focal_loss import \
    adamptive_binary_focal_loss as binary_focal_loss
from sgg_lab.callbacks import ModelSaveBestAvgAcc, filter_val_f1score


def resize_all(output_shape):
    resizer_0 = ds.resize(u.mul_shape(output_shape, 16))
    #  resizer_1 = ds.resize(u.mul_shape(output_shape, 8))
    #  resizer_2 = ds.resize(u.mul_shape(output_shape, 4))
    #  resizer_3 = ds.resize(u.mul_shape(output_shape, 2))
    #  resizer_4 = ds.resize(output_shape)

    def f(x):
        w = resizer_0(x)
        #  x = resizer_1(x)
        #  y = resizer_2(x)
        #  z = resizer_3(y)
        #  v = resizer_4(x)

        return w,  # , x, y, z, v

    return f


def prepare(dataset, epochs, batch_size, input_shape, output_shape):
    stream = ds.epochs(dataset.image_ids, epochs)
    stream = ds.stream(
        lambda x: (dataset.load_image(x), dataset.load_output(x)),
        stream)
    stream = ds.stream(ds.apply_to_x(ds.resize(input_shape)), stream)
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
    #  output = layers.Conv2D(1, (1, 1), activation='sigmoid')(model.output)

    mul2, output2 = mul_layer(model.get_layer('activation_40'), model.output)
    mul3, output3 = mul_layer(model.get_layer('activation_22'), mul2)
    mul4, output4 = mul_layer(model.get_layer('activation_10'), mul3)
    mul5, output5 = mul_layer(model.get_layer('activation_1'), mul4)

    return models.Model(
        inputs=model.inputs,
        outputs=[
            output5  # , output4, output3, output2, output
        ])


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
