# Texture-Based Image Classification using Machine Learning

A classical machine learning pipeline for image classification — built without a single neural network.

---

## Overview

This project explores how far hand-crafted texture features can take us in image classification. By combining intensity histograms and gradient-based descriptors, we build a lightweight, interpretable pipeline and benchmark it against Fashion-MNIST using two classifiers: Gaussian Naïve Bayes and Random Forest.

---

## Dataset

**Fashion-MNIST** — 60,000 training / 10,000 test grayscale images (28×28) across 10 clothing categories. Loaded automatically via `tensorflow.keras.datasets`.

---

## Pipeline

```
Raw Images → Normalization → Feature Extraction → Feature Fusion → Training → Evaluation
```

**Intensity Histograms** capture global pixel distribution. **Sobel Gradients** capture edge strength and texture variation. Both are concatenated into a single feature vector — this fusion consistently outperforms either descriptor alone.

---

## Results

| Model | Accuracy | Macro F1-Score |
|---|---|---|
| Gaussian Naïve Bayes | Moderate | Moderate |
| Random Forest | Higher | Higher |

*Fill in actual values after running the notebook.*

---

## How to Run

Open the notebook in Google Colab and run all cells. All dependencies (`numpy`, `opencv-python`, `scikit-learn`, `matplotlib`, `tensorflow`) are pre-installed in Colab.

---

## Tech Stack

Python · NumPy · OpenCV · Scikit-learn · Matplotlib · TensorFlow · Google Colab

---

## Future Work

- Add HOG or LBP texture descriptors
- Apply PCA for dimensionality reduction
- Use k-fold cross-validation
- Benchmark against a CNN

---

## About

Built as part of a Machine Learning coursework module, focused on classical feature engineering and model evaluation.

