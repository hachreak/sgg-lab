
"""Train Mask R-CNN on Open Images Dataset."""

import os

from mrcnn.config import Config
from mrcnn import model as modellib

from sgg_lab.datasets import open_images_cocostyle as oic, open_images as oi

path = '/media/hachreak/Magrathea/datasets/open-images-v5'
path_classes = os.path.join(path, oi.OD_CLASSES)
path_mask_train = os.path.join(path, oic.SEGM_TRAIN)
path_mask_valid = os.path.join(path, oic.SEGM_VALID)
path_image_train = os.path.join(path, 'train')
path_image_valid = os.path.join(path, 'validation')
path_image_mask_train = os.path.join(path, 'challenge-2019-train-masks')
path_image_mask_valid = os.path.join(path, 'challenge-2019-validation-masks')

model_dir = './logs'
path_coco_weigths = 'mask_rcnn_coco.h5'

batch_size = 64
epochs = 50

classes = oic.get_classes(path_classes)

count_train = oic.count_images(path_image_train)
count_valid = oic.count_images(path_image_valid)


class OICConfig(Config):

    NAME = 'open-images-v5'

    NUM_CLASSES = len(classes) + 1

    IMAGES_PER_GPU = 1

    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320

    #  RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    #  TRAIN_ROIS_PER_IMAGE = 32

    BATCH_SIZE = batch_size

    STEPS_PER_EPOCH = count_train // batch_size
    VALIDATION_STEPS = count_valid // batch_size


config = OICConfig()
config.display()

dataset_valid = oic.OIDataset(
    path_mask_valid, path_image_valid, path_image_mask_valid, classes
)
dataset_valid.prepare()

#  import ipdb; ipdb.set_trace()
#  fuu = next(dataset_valid)

dataset_train = oic.OIDataset(
    path_mask_train, path_image_train, path_image_mask_train, classes
)
dataset_train.prepare()


model = modellib.MaskRCNN(mode="training", config=config, model_dir=model_dir)
model.load_weights(
    path_coco_weigths, by_name=True,
    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
)

model.train(
    dataset_train, dataset_valid,
    learning_rate=config.LEARNING_RATE,
    epochs=epochs,
    layers='heads'
)

print('fine')
