# ── utils/visualization.py ────────────────────────────────────────────────────
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from config import CLASS_NAMES, NUM_CLASSES, IMG_SIZE


def plot_class_distribution(train_counts, test_counts=None):

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(train_counts.keys(), train_counts.values(), label='Train')
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    plt.show()


def plot_sample_grid(train_path):

    fig, axes = plt.subplots(4, 5, figsize=(10, 5))

    for row, cls in enumerate(CLASS_NAMES):
        cls_path = os.path.join(train_path, cls)
        images   = random.sample(os.listdir(cls_path), 5)
        imge_label = CLASS_NAMES.index(cls)
        for col, img_name in enumerate(images):
            img = Image.open(os.path.join(cls_path, img_name)).resize((224, 224))
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'{CLASS_NAMES[imge_label]}', fontsize=10, fontweight='bold')
            axes[row, col].axis('off')

    plt.suptitle('Sample MRI Images per Class', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_roc_curves(best_model, test_ds, test_labels):

    y_score = best_model.predict(test_ds)

    y_true_binarized = label_binarize(test_labels, classes=np.arange(NUM_CLASSES))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(NUM_CLASSES):
        plt.plot(fpr[i], tpr[i],
                 label=f'ROC curve of class {CLASS_NAMES[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random guess (area = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
