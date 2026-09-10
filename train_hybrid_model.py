import os
import time
import joblib
import numpy as np
import pandas as pd

from feature_extraction import extract_features

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


print("=" * 75)
print("HYBRID URL PHISHING DETECTION MODEL")
print("=" * 75)


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("\n[1/7] Loading dataset...")

data = pd.read_csv(
    "dataset/phishing_dataset.csv"
)

data = data.dropna(
    subset=["URL", "Label"]
)

print("Total URLs:", len(data))

print("\nClass distribution:")
print(data["Label"].value_counts())


# =========================================================
# 2. PREPARE LABELS
# =========================================================

print("\n[2/7] Preparing labels...")

urls = data["URL"].astype(str).tolist()

y = np.array([
    1 if label == "Phishing" else 0
    for label in data["Label"]
])

print("Phishing:", np.sum(y == 1))
print("Legitimate:", np.sum(y == 0))


# =========================================================
# 3. EXTRACT STRUCTURAL FEATURES
# =========================================================

print("\n[3/7] Extracting 30 URL features...")

start = time.time()

structural_features = []

for i, url in enumerate(urls):

    structural_features.append(
        extract_features(url)
    )

    if (i + 1) % 20000 == 0:
        print(
            f"Processed {i + 1:,} / {len(urls):,}"
        )

structural_features = np.array(
    structural_features,
    dtype=float
)

print(
    f"\nFeature extraction time: "
    f"{time.time() - start:.2f} seconds"
)

print(
    "Structural feature matrix:",
    structural_features.shape
)


# =========================================================
# 4. TRAIN / TEST SPLIT
# =========================================================

print("\n[4/7] Creating train/test split...")

indices = np.arange(len(urls))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_urls = [
    urls[i] for i in train_idx
]

test_urls = [
    urls[i] for i in test_idx
]

y_train = y[train_idx]
y_test = y[test_idx]

struct_train = structural_features[
    train_idx
]

struct_test = structural_features[
    test_idx
]

print("Training:", len(train_urls))
print("Testing :", len(test_urls))


# =========================================================
# 5. CHARACTER-LEVEL TF-IDF
# =========================================================

print("\n[5/7] Building character-level TF-IDF...")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=2,
    max_features=60000,
    sublinear_tf=True
)

start = time.time()

X_train_text = vectorizer.fit_transform(
    train_urls
)

X_test_text = vectorizer.transform(
    test_urls
)

print(
    f"TF-IDF time: "
    f"{time.time() - start:.2f} seconds"
)

print(
    "TF-IDF training matrix:",
    X_train_text.shape
)


# =========================================================
# 6. SCALE STRUCTURAL FEATURES
# =========================================================

print("\n[6/7] Scaling structural features...")

scaler = StandardScaler()

struct_train_scaled = scaler.fit_transform(
    struct_train
)

struct_test_scaled = scaler.transform(
    struct_test
)


# =========================================================
# FEATURE FUSION
# =========================================================

print("\nCombining TF-IDF + structural features...")

from scipy.sparse import csr_matrix, hstack

struct_train_sparse = csr_matrix(
    struct_train_scaled
)

struct_test_sparse = csr_matrix(
    struct_test_scaled
)

X_train = hstack([
    X_train_text,
    struct_train_sparse
])

X_test = hstack([
    X_test_text,
    struct_test_sparse
])

print(
    "Final training matrix:",
    X_train.shape
)

print(
    "Final testing matrix:",
    X_test.shape
)


# =========================================================
# TRAIN HYBRID MODEL
# =========================================================

print("\nTraining Logistic Regression hybrid classifier...")

model = LogisticRegression(
    C=3.0,
    max_iter=1000,
    solver="liblinear",
    random_state=42
)

start = time.time()

model.fit(
    X_train,
    y_train
)

training_time = time.time() - start

print(
    f"Training time: "
    f"{training_time:.2f} seconds"
)


# =========================================================
# EVALUATION
# =========================================================

print("\nEvaluating hybrid model...")

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)


print("\n" + "=" * 65)
print("HYBRID MODEL RESULTS")
print("=" * 65)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1       : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print(
    f"PR-AUC   : {pr_auc:.4f}"
)

print(
    f"Training Time: {training_time:.2f} seconds"
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Legitimate",
            "Phishing"
        ],
        zero_division=0
    )
)


# =========================================================
# SAVE MODEL
# =========================================================

print("\n[7/7] Saving hybrid model...")

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    model,
    "model/phishing_hybrid_model.pkl"
)

joblib.dump(
    vectorizer,
    "model/hybrid_tfidf_vectorizer.pkl"
)

joblib.dump(
    scaler,
    "model/hybrid_scaler.pkl"
)


hybrid_info = {
    "model_name": "Hybrid TF-IDF + URL Features",
    "structural_features": 30,
    "tfidf_max_features": 60000,
    "tfidf_ngram_range": (2, 5),
    "dataset_size": len(data),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "training_samples": len(train_urls),
    "testing_samples": len(test_urls)
}


joblib.dump(
    hybrid_info,
    "model/hybrid_info.pkl"
)


print("\nSaved:")

print(
    "model/phishing_hybrid_model.pkl"
)

print(
    "model/hybrid_tfidf_vectorizer.pkl"
)

print(
    "model/hybrid_scaler.pkl"
)

print(
    "model/hybrid_info.pkl"
)


print("\n" + "=" * 75)
print("HYBRID MODEL TRAINING COMPLETED")
print("=" * 75)