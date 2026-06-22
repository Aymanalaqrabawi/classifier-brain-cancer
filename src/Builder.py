# ── model/builder.py ──────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D,
    BatchNormalization, Activation
)


def build_model(trainable_base=False, fine_tune_at=None):

    base = ResNet50(weights='imagenet', include_top=False,
                    input_shape=(224, 224, 3))

    base.trainable = trainable_base

    if fine_tune_at:
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(4, activation='softmax')(x)

    return Model(inputs, outputs)
