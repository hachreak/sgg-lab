
"""Train Noise Removal on coco."""

import utils as u

from keras import optimizers as opt, models, layers
from keras.applications import resnet50

from sgg_lab import datasets as ds
from sgg_lab.losses.focal_loss import \
    adaptive_binary_focal_loss as binary_focal_loss
from sgg_lab.callbacks import ModelSaveBestAvgAcc


def resize_all(output_shape):
    resizer_1 = ds.resize(u.mul_shape(output_shape, 4))
    resizer_2 = ds.resize(u.mul_shape(output_shape, 2))
    resizer_3 = ds.resize(output_shape)

    def f(x):
        x = resizer_1(x)
        y = resizer_2(x)
        z = resizer_3(y)
        return x, y, z

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
    upsample = layers.UpSampling2D((2, 2))(last)
    to1024 = layers.Conv2D(
        int(back.shape[3]), (1, 1), activation='relu')(upsample)
    mul = layers.Multiply()([back, to1024])
    return mul, layers.Conv2D(1, (1, 1), activation='sigmoid')(mul)


def get_model(input_shape):
    model = resnet50.ResNet50(include_top=False, input_shape=input_shape)
    output = layers.Conv2D(1, (1, 1), activation='sigmoid')(model.output)

    mul2, output2 = mul_layer(model.get_layer('activation_40'), model.output)
    mul3, output3 = mul_layer(model.get_layer('activation_22'), mul2)

    return models.Model(
        inputs=model.inputs, outputs=[output3, output2, output])


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
    filepath="model-{epoch:02d}-{acc:.2f}.hdf5",
    verbose=True
)

losses = []
for i in range(0, 3):
    losses.append(binary_focal_loss(gamma=2.))

model = get_model(input_shape)
model.compile(
    optimizer=opt.Adam(lr=1e-4),
    loss=losses,
    metrics=['accuracy']
)

model.summary()

model.fit_generator(
    gen_train,
    steps_per_epoch=len(dataset_train.image_ids) // batch_size,
    epochs=epochs,
    validation_data=gen_val,
    validation_steps=len(dataset_val.image_ids) // batch_size,
    callbacks=[callback],
)

print('fine')
