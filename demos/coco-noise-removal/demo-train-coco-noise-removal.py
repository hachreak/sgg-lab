
"""Train Noise Removal on coco."""

import utils as u

from keras import optimizers as opt, callbacks, models, layers
from keras.applications import resnet50

from sgg_lab.losses.focal_loss import binary_focal_loss
from sgg_lab import datasets as ds


def get_model(input_shape):
    model = resnet50.ResNet50(include_top=False, input_shape=input_shape)
    output = layers.Conv2D(1, (1, 1), activation='sigmoid')(model.output)
    return models.Model(inputs=model.inputs, outputs=output)


def prepare(dataset, epochs, batch_size, input_shape, output_shape):
    stream = ds.epochs(dataset.image_ids, epochs)
    #  stream = ds.stream(check, stream)
    stream = ds.stream(
        lambda x: (dataset.load_image(x), dataset.load_output(x)),
        stream)
    stream = ds.stream(ds.apply_to_x(ds.resize(input_shape)), stream)
    stream = ds.stream(ds.apply_to_y(ds.resize(output_shape)), stream)
    #  stream = ds.stream(ds.apply_to_x(check), stream)

    stream = ds.bufferize(stream, size=20)

    batch = ds.stream_batch(stream, size=batch_size)
    batch = ds.stream(ds.apply_to_y(
        lambda x: ds.image2mask(x).reshape(x.shape + (1,))), batch)

    return batch


coco_path = '/media/hachreak/Magrathea/datasets/coco/coco'
model_path = ''
epochs = 100
batch_size = 3
input_shape = (320, 320, 3)
output_shape = (10, 10)

action = 'train'
# action = 'evaluate'

dataset_val = u.get_dataset(coco_path, 'val')
gen_val = prepare(dataset_val, epochs, batch_size, input_shape, output_shape)

#  fuu = next(gen_val)
#  model = u.load_model('./model-03-0.90.hdf5')
#  res = u.get_img_cleaned(model, fuu[0])

# train dataset
dataset_train = u.get_dataset(coco_path, 'train')
gen_train = prepare(
    dataset_train, epochs, batch_size, input_shape, output_shape)

callback = callbacks.ModelCheckpoint(
    filepath="model-{epoch:02d}-{val_acc:.2f}.hdf5",
    monitor="val_acc",
    save_best_only=True, save_weights_only=False,
    mode="max", verbose=1
)

#  model = unet(input_shape=input_shape)
model = get_model(input_shape)
model.compile(
    optimizer=opt.Adam(lr=1e-4),
    loss=binary_focal_loss(),
    #  loss='binary_crossentropy',
    #  loss=losses.mean_squared_error,
    #  metrics=['accuracy', myacc(input_shape[0] * input_shape[1])]
    metrics=['accuracy']
)

model.summary()
#  import ipdb; ipdb.set_trace()

model.fit_generator(
    gen_train,
    steps_per_epoch=len(dataset_train.image_ids) // batch_size,
    epochs=epochs,
    validation_data=gen_val,
    validation_steps=len(dataset_val.image_ids) // batch_size,
    callbacks=[callback],
)

print('fine')
