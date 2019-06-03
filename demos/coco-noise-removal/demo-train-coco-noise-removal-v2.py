
"""Train Noise Removal on coco."""

import utils as u

from keras import optimizers as opt, losses, callbacks, models, layers
from keras.applications import resnet50

from sgg_lab import datasets as ds
from sgg_lab.losses.focal_loss import binary_focal_loss
from sgg_lab.callbacks import ModelSaveBestAvgAcc


def prepare(dataset, epochs, batch_size, input_shape, output_shape):
    stream = ds.epochs(dataset.image_ids, epochs)
    #  stream = ds.stream(check, stream)
    stream = ds.stream(
        lambda x: (dataset.load_image(x), dataset.load_output(x)),
        stream)
    stream = ds.stream(ds.apply_to_x(ds.resize(input_shape)), stream)
    stream = ds.stream(ds.apply_to_y(
        ds.resize([o*2 for o in output_shape])), stream)
    stream = ds.stream(ds.apply_to_y(lambda x: (x, x)), stream)
    stream = ds.stream(ds.apply_to_y(ds.apply_to_y(
        ds.resize(output_shape))), stream)
    #  stream = ds.stream(ds.apply_to_y(check), stream)

    stream = ds.bufferize(stream, size=20)

    batch = ds.stream_batch(stream, size=batch_size, fun=ds.pack_elements)
    #  batch = ds.stream(ds.apply_to_y(check), batch)
    batch = ds.stream(ds.apply_to_y(ds.apply_to_xn(
        lambda x: ds.image2mask(x).reshape(x.shape + (1,)))), batch)

    return batch


def get_model(input_shape):
    model = resnet50.ResNet50(include_top=False, input_shape=input_shape)
    output = layers.Conv2D(1, (1, 1), activation='sigmoid')(model.output)
    last_20x20 = model.get_layer('activation_40').output
    upsample = layers.UpSampling2D((2, 2))(output)
    add = layers.Add()([last_20x20, upsample])
    output2 = layers.Conv2D(1, (1, 1), activation='sigmoid')(add)

    return models.Model(inputs=model.inputs, outputs=[output2, output])


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
#  import ipdb; ipdb.set_trace()
#  model = u.load_model('./model-03-0.90.hdf5')
#  res = u.get_img_cleaned(model, fuu[0])
#  import ipdb; ipdb.set_trace()

# train dataset
dataset_train = u.get_dataset(coco_path, 'train')
gen_train = prepare(
    dataset_train, epochs, batch_size, input_shape, output_shape)

callback = ModelSaveBestAvgAcc(
    filepath="model-{epoch:02d}-{acc:.2f}.hdf5",
    verbose=True
)
#  callback = callbacks.ModelCheckpoint(
#      filepath="model-{epoch:02d}-{val_acc:.2f}.hdf5",
#      monitor="val_acc",
#      save_best_only=True, save_weights_only=False,
#      mode="max", verbose=1
#  )

#  model = unet(input_shape=input_shape)
model = get_model(input_shape)
model.compile(
    optimizer=opt.Adam(lr=1e-4),
    loss=[binary_focal_loss(), binary_focal_loss()],
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
