# ── inference/gradcam.py ──────────────────────────────────────────────────────
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from config import IMG_SIZE, CLASS_NAMES


def compute_gradcam(model, img_array):

    resnet = model.get_layer('resnet50')

    last_conv = None
    for layer in reversed(resnet.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break

    img_input = np.expand_dims(img_array, axis=0)

    preds      = model.predict(img_input, verbose=0)
    pred_class = np.argmax(preds[0])
    confidence = preds[0][pred_class]

    grad_model = tf.keras.Model(
        inputs=resnet.input,
        outputs=[resnet.get_layer(last_conv).output, resnet.output]
    )

    with tf.GradientTape() as tape:
        conv_out, res_out = grad_model(img_input, training=False)

        head = res_out
        for layer in model.layers[2:]:
            head = layer(head, training=False)

        score = head[:, pred_class]

    grads   = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    cam = conv_out[0] @ weights[..., tf.newaxis]
    cam = tf.squeeze(cam)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)

    return cam.numpy(), pred_class, confidence


def visualise_gradcam_mri(model, img_path, true_label=None):

    img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE)).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0

    heatmap, pred, conf = compute_gradcam(model, img)

    heatmap = tf.image.resize(
        heatmap[..., np.newaxis], [IMG_SIZE, IMG_SIZE]
    ).numpy()[..., 0]

    overlay = np.clip(
        0.5 * img + 0.5 * cm.jet(heatmap)[..., :3],
        0, 1
    )

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1); plt.imshow(img);                    plt.title("Image");   plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(heatmap, cmap="jet");    plt.title("Heatmap"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(overlay);                plt.title("Overlay"); plt.axis("off")
    plt.show()

    return pred, conf


def batch_gradcam_grid(model, test_paths, test_labels, n=8):

    plt.figure(figsize=(16, 8))

    for i in range(n):
        path = test_paths[i]

        img = Image.open(path).resize((IMG_SIZE, IMG_SIZE)).convert('RGB')
        img = np.array(img).astype(np.float32) / 255.0

        heatmap, pred, conf = compute_gradcam(model, img)

        heatmap = tf.image.resize(
            heatmap[..., np.newaxis], [IMG_SIZE, IMG_SIZE]
        ).numpy()[..., 0]

        overlay = np.clip(
            0.5 * img + 0.5 * cm.jet(heatmap)[..., :3], 0, 1
        )

        plt.subplot(2, 4, i + 1)
        plt.imshow(overlay)
        plt.title(f"P:{CLASS_NAMES[pred]} | {conf:.0%}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
