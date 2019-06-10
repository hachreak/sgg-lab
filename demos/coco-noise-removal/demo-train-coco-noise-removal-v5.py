
"""Train Noise Removal on coco."""

import utils as u

import numpy as np
import os

from keras import optimizers as opt, models, layers
from keras.applications import resnet50

from sgg_lab import datasets as ds
from sgg_lab.losses.focal_loss import \
    adamptive_binary_focal_loss as binary_focal_loss
from sgg_lab.callbacks import ModelSaveBestAvgAcc


def load_masks(dataset, base_path):
    """Load numpy file."""
    def f(image_id):
        file_path = '{0}.npy'.format(dataset._img_filenames[image_id])
        path = os.path.join(base_path, file_path)
        y = np.load(path, allow_pickle=True)
        ret = [y[0][:, :, i].reshape(y[0].shape[:2] + (1,))
               for i in range(0, y[0].shape[2])]
        for i in range(1, len(y)):
            ret.append(y[i])
        return ret
    return f


def prepare(dataset, epochs, batch_size, base_path):
    get_masks = load_masks(dataset, base_path)

    stream = ds.epochs(dataset.image_ids, epochs)
    stream = ds.stream(lambda x: (
        dataset.load_image(x), get_masks(x)
    ), stream)

    stream = ds.bufferize(stream, size=batch_size)
    batch = ds.stream_batch(stream, size=batch_size, fun=ds.pack_elements)

    return batch


def get_model(input_shape, num_classes):
    model = resnet50.ResNet50(include_top=False, input_shape=input_shape)
    output = layers.Conv2D(1, (1, 1), activation='sigmoid')(model.output)

    last_20x20 = model.get_layer('activation_40').output
    upsample = layers.UpSampling2D((2, 2))(output)
    mul1 = layers.Multiply()([last_20x20, upsample])
    output2 = layers.Conv2D(1, (1, 1), activation='sigmoid')(mul1)

    last_40x40 = model.get_layer('activation_22').output
    upsample2 = layers.UpSampling2D((2, 2))(output2)
    mul2 = layers.Multiply()([last_40x40, upsample2])
    output3 = layers.Conv2D(1, (1, 1), activation='sigmoid')(mul2)

    output4 = [
        layers.Conv2D(1, (1, 1), activation='sigmoid')(mul2)
        for i in range(0, num_classes)
    ]

    return models.Model(
        inputs=model.inputs, outputs=output4 + [output3, output2, output])


def get_class_masks(model, gen_dataset):
    """Get class masks y_true and y_pred."""
    from sklearn.metrics import confusion_matrix
    import ipdb; ipdb.set_trace()
    cm = np.zeros((2, 2))
    for x, y_true in gen_dataset:
        y_true = np.array(y_true[:81]).flatten()
        y_pred = np.array(model.predict(x)[:81]).flatten()
        y_pred[y_pred < 0.7] = 0
        y_pred[y_pred >= 0.7] = 1
        cm_res = confusion_matrix(y_true, y_pred)
        if cm_res.shape == (2, 2):
            cm += cm_res
    import ipdb; ipdb.set_trace()
    print('ok')


#  coco_path = '/media/hachreak/Magrathea/datasets/coco/resize_320x320'
coco_path = '/tmp/coco_resize_320x320'
model_path = ''
epochs = 100
batch_size = 10
input_shape = (320, 320, 3)
output_shape = (10, 10)
model_path = './model-52-0.92.hdf5'

#  action = 'train'
action = 'evaluate'

dataset_val = u.get_dataset(coco_path, 'val')
gen_val = prepare(
    dataset_val, epochs, batch_size, os.path.join(coco_path, 'val_output')
)

#  fuu = next(gen_val)
#  import ipdb; ipdb.set_trace()
#  model = u.load_model('./model-03-0.90.hdf5')
#  res = u.get_img_cleaned(model, fuu[0])
#  import ipdb; ipdb.set_trace()

if action == 'train':
    # train dataset
    dataset_train = u.get_dataset(coco_path, 'train')
    gen_train = prepare(
        dataset_train, epochs, batch_size,
        os.path.join(coco_path, 'train_output')
    )

    callback = ModelSaveBestAvgAcc(
        filepath="model-{epoch:02d}-{acc:.2f}.hdf5",
        verbose=True
    )

    losses = []
    for i in range(0, dataset_val.num_classes) + 3:
        losses.append(binary_focal_loss(gamma=2.))

    #  model = unet(input_shape=input_shape)
    model = get_model(input_shape, dataset_val.num_classes)
    model.compile(
        optimizer=opt.Adam(lr=1e-4),
        loss=losses,
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
        verbose=2
    )
else:
    model = u.load_model(model_path)
    get_class_masks(model, gen_val)
print('fine')
