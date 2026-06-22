# ── training/trainer.py ───────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from config import P1_EPOCHS, P1_LR, CKPT_P1, CKPT_P2


def cosine_decay_with_warmup():

    return tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=1000
    )


def get_callbacks(ckpt, name):

    return [
        tf.keras.callbacks.ModelCheckpoint(ckpt, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(f'/content/{name}.csv')
    ]


def train_phase1(model, train_ds, val_ds, class_weight_dict):

    from model.loss import focal_loss

    model.compile(
        optimizer=Adam(P1_LR),
        loss=focal_loss(),
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=P1_EPOCHS,
        class_weight=class_weight_dict,
        callbacks=get_callbacks(CKPT_P1, 'p1')
    )
    return history
