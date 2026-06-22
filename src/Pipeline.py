# ── data/pipeline.py ──────────────────────────────────────────────────────────
import tensorflow as tf
from config import IMG_SIZE, BATCH_SIZE, AUTOTUNE


def load_image(path, label):

    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def augment_train(img, label):

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    k   = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


def apply_resnet_preprocessing(img, label):

    img = img * 255.0
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, label


def make_dataset(paths, labels, training=True):

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment_train, num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(2048, seed=42)
    ds = ds.map(apply_resnet_preprocessing, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds
