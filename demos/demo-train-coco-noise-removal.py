
"""Train Noise Removal on coco."""

from keras import optimizers as opt, losses, callbacks

from sgg_lab.datasets.coco import CocoDataset, join_masks
from sgg_lab import datasets as ds
from sgg_lab.nets.unet import unet
from sgg_lab.losses.focal_loss import binary_focal_loss


def prepare(dataset, epochs, batch_size, input_shape):
    stream = ds.epochs(dataset.image_ids, epochs)
    stream = ds.stream(
        lambda x: (dataset.load_image(x), dataset.load_output(x)),
        stream)
    stream = ds.stream(ds.apply_to_xn(ds.resize(input_shape[:2])), stream)
    #  stream = ds.stream(ds.apply_to_x(check), stream)

    stream = ds.bufferize(stream, size=20)

    batch = ds.stream_batch(stream, size=batch_size)
    batch = ds.stream(ds.apply_to_y(
        lambda x: ds.mask2image(x).reshape(x.shape + (1,))), batch)

    return batch


class NRCocoDataset(CocoDataset):

    def load_output(self, image_id):
        masks = self.load_mask(image_id)[0]
        return join_masks(masks).astype('uint8')


#  def myacc(n_pixels):
#      """Compute accuracy."""
#      def f(y_true, y_pred):
#          yp = tf.cast(tf.math.greater(y_pred, 0.5), tf.float32)
#          return tf.reduce_sum(y_true * yp) / n_pixels
#      return f


def check(x):
    import ipdb; ipdb.set_trace()
    return x


coco_path = '/media/hachreak/Magrathea/datasets/coco/coco'
model_path = ''
epochs = 100
batch_size = 3
input_shape = (320, 320, 3)

action = 'train'
# action = 'evaluate'

# validation dataset
dataset_val = NRCocoDataset()
dataset_val.load_coco(coco_path, 'val')
dataset_val.prepare()
gen_val = prepare(dataset_val, epochs, batch_size, input_shape)

#  fuu = next(gen_val)
#  import ipdb; ipdb.set_trace()

# train dataset
dataset_train = NRCocoDataset()
dataset_train.load_coco(coco_path, 'train')
dataset_train.prepare()
gen_train = prepare(dataset_train, epochs, batch_size, input_shape)

callback = callbacks.ModelCheckpoint(
    filepath="model-{epoch:02d}-{val_acc:.2f}.hdf5",
    monitor="val_acc",
    save_best_only=True, save_weights_only=False,
    mode="max", verbose=1
)

model = unet(input_shape=input_shape)

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
import ipdb; ipdb.set_trace()
print('fine')
