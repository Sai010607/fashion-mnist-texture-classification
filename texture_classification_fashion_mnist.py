Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
... # 1. Imports
... 
... import numpy as np
... import matplotlib.pyplot as plt
... import cv2
... 
... from tensorflow.keras.datasets import fashion_mnist
... from sklearn.naive_bayes import GaussianNB
... from sklearn.ensemble import RandomForestClassifier
... from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
... 
... 
... # 2. Load Dataset
... 
... (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
... 
... print("Train shape:", X_train.shape)
... print("Test shape:", X_test.shape)
... 
... 
... # 3. Image Normalization
... 
... X_train = X_train.astype('float32') / 255.0
... X_test  = X_test.astype('float32') / 255.0
... 
... print("After normalization → Min:", X_train.min(), "Max:", X_train.max())
... 

# 4. Histogram Feature Extraction

def extract_histogram_features(images, bins=32):
    features = []
    for img in images:
        hist = np.histogram(img, bins=bins, range=(0,1))[0]
        hist = hist / np.sum(hist)
        features.append(hist)
    return np.array(features)


# 5. Gradient Feature Extraction

def extract_gradient_features(images):
    features = []
    for img in images:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features.append(grad_mag.flatten())
    return np.array(features)


# 6. Extract Features

print("\nExtracting histogram features...")
X_train_hist = extract_histogram_features(X_train)
X_test_hist  = extract_histogram_features(X_test)

print("Extracting gradient features...")
X_train_grad = extract_gradient_features(X_train)
X_test_grad  = extract_gradient_features(X_test)


# 7. Feature Fusion

X_train_feat = np.hstack([X_train_hist, X_train_grad])
X_test_feat  = np.hstack([X_test_hist, X_test_grad])

print("Final feature shape:", X_train_feat.shape)


# 8. Train Naïve Bayes

nb_model = GaussianNB()
nb_model.fit(X_train_feat, y_train)
y_pred_nb = nb_model.predict(X_test_feat)


# 9. Train Random Forest

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_feat, y_train)
y_pred_rf = rf_model.predict(X_test_feat)


# 10. Performance Evaluation

acc_nb = accuracy_score(y_test, y_pred_nb)
acc_rf = accuracy_score(y_test, y_pred_rf)

f1_nb = f1_score(y_test, y_pred_nb, average='macro')
f1_rf = f1_score(y_test, y_pred_rf, average='macro')

print("\n Performance Comparison")
print(f"Naïve Bayes → Accuracy: {acc_nb:.4f}, Macro F1: {f1_nb:.4f}")
print(f"Random Forest → Accuracy: {acc_rf:.4f}, Macro F1: {f1_rf:.4f}")


# 11. Confusion Matrix (RF)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf)
plt.title("Random Forest Confusion Matrix")
