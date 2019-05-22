
"""Train Mask with COCO."""

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from mrcnn.config import Config
from mrcnn import model as modellib, utils

from sgg_lab.datasets.coco import CocoDataset


coco_path = '/media/hachreak/Magrathea/datasets/coco/coco'
model_path = ''
epochs = 100
batch_size = 1

action = 'train'
# action = 'evaluate'


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
