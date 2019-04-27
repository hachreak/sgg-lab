
"""VGG 19 to classify."""

from keras.applications import vgg19
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, \
        BatchNormalization


def get_model(input_shape, output_shape):
    """Get model ready to use."""
    model = vgg19.VGG19(
        include_top=False, weights='imagenet', input_shape=input_shape
    )
    model = _set_readonly(model, 18)
    model = _classification(model, output_shape)
    return model


def _set_readonly(model, until=None):
    """Make a model weights readonly."""
    for layer in model.layers[:until]:
        layer.trainable = False
    return model


def _classification(model, output_shape):
    """Add classification layers."""
    x = model.output
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = [Dense(1, activation='sigmoid')(x) for i in range(0, output_shape)]

    return Model(inputs=model.input, outputs=out)
