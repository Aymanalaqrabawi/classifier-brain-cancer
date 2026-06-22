iBrain Cancer MRI Classifier

Production-grade Deep Learning pipeline for Brain MRI classification using TensorFlow and ResNet50.

Overview

This project implements a complete medical image classification system capable of identifying four brain MRI categories:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

The pipeline follows industry-level machine learning practices including:

- Transfer Learning with ResNet50
- Two-Phase Fine-Tuning Strategy
- Custom Focal Loss
- Class Imbalance Handling
- TensorFlow tf.data Pipeline
- Explainable AI using Grad-CAM
- Model Export (SavedModel & TensorFlow Lite)

---

Dataset Structure

Dataset/
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/

---

Features

Efficient Data Pipeline

- TensorFlow "tf.data"
- Parallel preprocessing
- Dataset prefetching
- On-the-fly augmentation
- GPU optimized loading

Data Augmentation

- Random horizontal flip
- Random vertical flip
- Brightness adjustment
- Contrast adjustment
- Random 90° rotations

Imbalanced Dataset Handling

- Class Weights
- Custom Focal Loss

Explainable AI

Grad-CAM visualizations allow interpretation of model predictions by highlighting important regions inside MRI scans.

Deployment Ready

Export supported formats:

- TensorFlow SavedModel
- TensorFlow Lite (.tflite)

---

Model Architecture

Backbone

ResNet50 pretrained on ImageNet

Classification Head

Input
  ↓
ResNet50 Backbone
  ↓
GlobalAveragePooling2D
  ↓
BatchNormalization
  ↓
Dropout(0.5)
  ↓
Dense(512)
  ↓
BatchNormalization
  ↓
ReLU
  ↓
Dropout(0.5)
  ↓
Dense(512)
  ↓
BatchNormalization
  ↓
ReLU
  ↓
Dropout(0.5)
  ↓
Dense(4, Softmax)

---

Training Strategy

Phase 1: Feature Extraction

Freeze all ResNet50 layers.

model_p1 = build_model(trainable_base=False)

Train only the classification head.

Phase 2: Fine-Tuning

Unfreeze deeper ResNet50 layers.

model_p2 = build_model(
    trainable_base=True,
    fine_tune_at=140
)

Train using a smaller learning rate to adapt ImageNet features to medical imaging.

---

Custom Focal Loss

Designed for class imbalance.

focal_loss(
    gamma=2.0,
    alpha=0.25
)

Benefits:

- Focuses on hard examples
- Reduces dominance of majority classes
- Improves minority-class recall

---

Learning Rate Scheduling

Cosine Decay Restarts:

tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-3,
    first_decay_steps=1000
)

Benefits:

- Better convergence
- Escapes local minima
- Improved fine-tuning performance

---

Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curves
- AUC

Example:

print(classification_report(
    y_true,
    y_pred
))

---

Explainability with Grad-CAM

Grad-CAM generates heatmaps showing which MRI regions contributed most to the prediction.

Outputs include:

- Original MRI
- Heatmap
- Overlay Visualization

This is especially important for medical AI transparency and trust.

---

Exporting the Model

TensorFlow SavedModel

tf.saved_model.save(
    best_model,
    SAVEDMODEL_PATH
)

TensorFlow Lite

converter = tf.lite.TFLiteConverter.from_keras_model(
    best_model
)

tflite_model = converter.convert()

---

Requirements

tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
Pillow

Install:

pip install -r requirements.txt

---

Running the Project

Clone repository:
Aymanalaqrabawi/classifier-brain-cancer https://share.google/iS89X0pax6dxPZ0ok

Enter project:

cd brain-mri-classifier

Run notebook:

jupyter notebook

---

Project Highlights

- Production-style TensorFlow pipeline
- Transfer Learning with ResNet50
- Medical Image Classification
- Custom Focal Loss
- Grad-CAM Explainability
- TensorFlow Lite Deployment
- Class Imbalance Mitigation
- Scalable tf.data Pipeline

---

Future Improvements

- Vision Transformers (ViT)
- EfficientNetV2
- Ensemble Learning
- Model Quantization
- FastAPI Deployment
- Docker Integration
- CI/CD Pipeline
- Cloud Deployment

---

Author

Ayman Al-Aqrabawi

Data Science & Artificial Intelligence Student

Hashemite University

2025
