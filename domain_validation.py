import time
import numpy as np
import pandas as pd

from urllib.parse import urlparse

from feature_extraction import extract_features

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier

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
print("DOMAIN-AWARE PHISHING DETECTION VALIDATION")
print("=" * 75)


# =========================================================
# SETTINGS
# =========================================================

RANDOM_STATE = 42


# =========================================================
# DOMAIN EXTRACTION
# =========================================================

def extract_domain(url):

    url = str(url).strip()

    if not url.startswith(
        ("http://", "https://")
    ):
        url = "http://" + url

    try:

        parsed = urlparse(url)

        domain = parsed.netloc.lower()

        # Remove port
        domain = domain.split(":")[0]

        # Remove www
        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    except Exception:

        return url.lower()


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("\n[1/6] Loading dataset...")

data = pd.read_csv(
    "dataset/phishing_dataset.csv"
)

data = data.dropna(
    subset=["URL", "Label"]
)

print(
    "Total URLs:",
    len(data)
)


# =========================================================
# 2. CREATE DOMAIN GROUPS
# =========================================================

print("\n[2/6] Extracting domains...")

data["Domain"] = data["URL"].apply(
    extract_domain
)

print(
    "Unique domains:",
    data["Domain"].nunique()
)

print("\nLargest domains by URL count:")

print(
    data["Domain"]
    .value_counts()
    .head(10)
)


# =========================================================
# 3. EXTRACT FEATURES
# =========================================================

print("\n[3/6] Extracting 30 URL features...")

X = []
y = []
groups = []

start = time.time()

for i, row in data.iterrows():

    X.append(
        extract_features(
            row["URL"]
        )
    )

    y.append(
        1
        if row["Label"] == "Phishing"
        else 0
    )

    groups.append(
        row["Domain"]
    )

    if (i + 1) % 20000 == 0:

        print(
            f"Processed "
            f"{i + 1:,} / "
            f"{len(data):,}"
        )


X = np.array(
    X,
    dtype=float
)

y = np.array(
    y
)

groups = np.array(
    groups
)


print(
    f"\nFeature extraction time: "
    f"{time.time() - start:.2f} seconds"
)

print(
    "Feature matrix:",
    X.shape
)


# =========================================================
# 4. DOMAIN-AWARE SPLIT
# =========================================================

print("\n[4/6] Creating domain-aware split...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

train_domains = set(
    groups[train_idx]
)

test_domains = set(
    groups[test_idx]
)

overlap = (
    train_domains &
    test_domains
)


print("\nDomain split:")
print(
    "Training URLs:",
    len(X_train)
)

print(
    "Testing URLs :",
    len(X_test)
)

print(
    "Training domains:",
    len(train_domains)
)

print(
    "Testing domains :",
    len(test_domains)
)

print(
    "Domain overlap:",
    len(overlap)
)


# =========================================================
# 5. TRAIN MODEL
# =========================================================

print("\n[5/6] Training Gradient Boosting...")

model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=RANDOM_STATE
)

start = time.time()

model.fit(
    X_train,
    y_train
)

training_time = (
    time.time() - start
)

print(
    f"Training time: "
    f"{training_time:.2f} seconds"
)


# =========================================================
# 6. EVALUATE
# =========================================================

print("\n[6/6] Evaluating on unseen domains...")

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
print("DOMAIN-AWARE VALIDATION RESULTS")
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
    f"Training Time: "
    f"{training_time:.2f} seconds"
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


print("\n" + "=" * 75)
print("DOMAIN-AWARE VALIDATION COMPLETED")
print("=" * 75)