
"""Train Mask R-CNN on Open Images Dataset."""

import cv2
import numpy as np
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

#  action = 'training'
action = 'inference'

img_path = os.path.join(path, 'validation/53921fc4d72f04d6.jpg')
#  weights = 'logs/open-images-v520190516T1742/mask_rcnn_open-images-v5_0022.h5'
weights = 'logs/open-images-v520190516T1742/mask_rcnn_open-images-v5_0023.h5'
#  weights = 'mask_rcnn_coco.h5'

model = modellib.MaskRCNN(mode=action, config=config, model_dir=model_dir)

if action == 'training':
    model.load_weights(
        weights, by_name=True,
        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox",
                 "mrcnn_mask"]
    )

    model.train(
        dataset_train, dataset_valid,
        learning_rate=config.LEARNING_RATE,
        epochs=epochs,
        layers='heads'
    )

else:
    model.load_weights(weights, by_name=True)
    #  res = model.detect(np.array([cv2.imread(img_path)]))
    import random
    from PIL import Image
    image_id = random.choice(dataset_valid.image_ids)
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_valid, config, image_id,
                               use_mini_mask=False)
    info = dataset_valid.image_info[image_id]
    results = model.detect_molded(
        np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)

    mask = results[0]['masks'].astype('uint8')
    mask = mask.reshape(mask.shape[:2])
    mask[mask > 0] = 255
    Image.fromarray(mask).show()
    Image.fromarray(image).show()
    print("class {0} - {1}".format(gt_class_id, results[0]['class_ids']))

print('fine')
