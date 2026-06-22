# ── main.py ───────────────────────────────────────────────────────────────────
# Entry point — connects all modules together
# Replaces the notebook's top-to-bottom execution flow
# ─────────────────────────────────────────────────────────────────────────────

import os
import random
import warnings
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# ── Mount Drive (Colab only) ──────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── Config ────────────────────────────────────────────────────────────────────
from config import (
    TRAIN_PATH, TEST_PATH,
    CLASS_NAMES, NUM_CLASSES,
    CKPT_P1, SAVEDMODEL, TFLITE_PATH
)

# ── Data ──────────────────────────────────────────────────────────────────────
from data.loader   import count_classes, collect_paths_labels, get_splits, get_class_weights
from data.pipeline import make_dataset

# ── Model ─────────────────────────────────────────────────────────────────────
from model.builder import build_model
from model.loss    import focal_loss

# ── Training ──────────────────────────────────────────────────────────────────
from training.trainer import train_phase1

# ── Inference / XAI ───────────────────────────────────────────────────────────
from inference.gradcam import visualise_gradcam_mri, batch_gradcam_grid

# ── Visualization ─────────────────────────────────────────────────────────────
from utils.visualization import plot_class_distribution, plot_sample_grid, plot_roc_curves


# ── 1. Explore Data ───────────────────────────────────────────────────────────
train_counts = count_classes(TRAIN_PATH)
test_counts  = count_classes(TEST_PATH)
print(f'Train counts: {train_counts}')
print(f'Test counts : {test_counts}')

plot_class_distribution(train_counts)
plot_sample_grid(TRAIN_PATH)


# ── 2. Load Paths & Labels ────────────────────────────────────────────────────
train_paths, train_labels = collect_paths_labels(TRAIN_PATH)
test_paths,  test_labels  = collect_paths_labels(TEST_PATH)

train_paths, val_paths, train_labels, val_labels = get_splits(
    train_paths, train_labels, test_paths, test_labels
)

class_weight_dict = get_class_weights(train_labels)


# ── 3. Build tf.data Pipelines ────────────────────────────────────────────────
train_ds = make_dataset(train_paths, train_labels, training=True)
val_ds   = make_dataset(val_paths,   val_labels,   training=False)
test_ds  = make_dataset(test_paths,  test_labels,  training=False)


# ── 4. Phase 1 Training ───────────────────────────────────────────────────────
model_p1   = build_model(trainable_base=False)
history_p1 = train_phase1(model_p1, train_ds, val_ds, class_weight_dict)


# ── 5. Load Best Checkpoint & Evaluate ───────────────────────────────────────
best_model = tf.keras.models.load_model(
    CKPT_P1,
    compile=False,
    custom_objects={'focal_loss': focal_loss()}
)

y_pred = np.argmax(best_model.predict(test_ds), axis=1)
y_true = np.array(test_labels)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))


# ── 6. Grad-CAM ───────────────────────────────────────────────────────────────
print('Grad-CAM — One sample per class:\n')
for cls_name in CLASS_NAMES:
    cls_dir    = os.path.join(TEST_PATH, cls_name)
    sample_img = os.path.join(cls_dir, random.choice(os.listdir(cls_dir)))
    print(f'\nClass: {cls_name.upper()}')
    visualise_gradcam_mri(best_model, sample_img,
                          true_label=CLASS_NAMES.index(cls_name))

print('\nBatch Grad-CAM grid...')
batch_gradcam_grid(best_model, test_paths, test_labels)


# ── 7. Export ─────────────────────────────────────────────────────────────────
tf.saved_model.save(best_model, SAVEDMODEL)

converter    = tf.lite.TFLiteConverter.from_keras_model(best_model)
tflite_model = converter.convert()
with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f'\n[✓] SavedModel → {SAVEDMODEL}')
print(f'[✓] TFLite     → {TFLITE_PATH}')


# ── 8. ROC Curves ─────────────────────────────────────────────────────────────
plot_roc_curves(best_model, test_ds, test_labels)
