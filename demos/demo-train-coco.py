
"""Train Mask with COCO."""

from __future__ import print_function

from mrcnn.config import Config
from mrcnn import model as modellib

from sgg_lab.datasets.coco import CocoDataset, evaluate_coco


coco_path = '/media/hachreak/Magrathea/datasets/coco/coco'
model_path = None
model_path = 'mask_rcnn_coco.h5'
epochs = 100
batch_size = 1
eval_limit = None

#  action = 'train'
action = 'evaluate'


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    # COCO has 80 classes
    NUM_CLASSES = 1 + 80

    IMAGE_SHAPE = (320, 320, 3)
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320

    BATCH_SIZE = batch_size


class InferenceConfig(CocoConfig):

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0


if action == 'train':
    config = CocoConfig()

    config.display()

    model = modellib.MaskRCNN(mode="training", config=config, model_dir='logs')

    if model_path:
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)
    else:
        print("Loading imagenet weights ")
        model.load_weights(model.get_imagenet_weights(), by_name=True)

    # train dataset
    dataset_train = CocoDataset()
    dataset_train.load_coco(coco_path, 'train')
    dataset_train.prepare()

    # validation dataset
    dataset_val = CocoDataset()
    dataset_val.load_coco(coco_path, 'val')
    dataset_val.prepare()

    print('train..')
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=epochs, layers='all'
    )
else:
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir='logs')

    if model_path:
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)

    # validation dataset
    dataset_val = CocoDataset()
    coco = dataset_val.load_coco(coco_path, 'val', return_coco=True)
    dataset_val.prepare()

    print("Running COCO mask evaluation on {} images.".format(eval_limit))
    evaluate_coco(model, dataset_val, coco, "segm", limit=eval_limit)
    print("Running COCO bbox evaluation on {} images.".format(eval_limit))
    evaluate_coco(model, dataset_val, coco, "bbox", limit=eval_limit)
