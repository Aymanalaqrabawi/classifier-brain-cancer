# ── model/loss.py ─────────────────────────────────────────────────────────────
import tensorflow as tf


def focal_loss(gamma=2.0, alpha=0.25, num_classes=4):

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true = tf.cast(y_true, tf.int32)
        y_true_onehot = tf.one_hot(y_true, depth=num_classes)
        cross_entropy = -y_true_onehot * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma)
        focal = alpha * weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))

    return loss
