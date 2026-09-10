import os
import time
import joblib
import pandas as pd

from feature_extraction import extract_features

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
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
print("ADVANCED ENSEMBLE PHISHING DETECTION MODEL")
print("=" * 75)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/6] Loading dataset...")

data = pd.read_csv(
    "dataset/phishing_dataset.csv"
)

print("Total URLs:", len(data))

print("\nClass distribution:")
print(data["Label"].value_counts())


# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================

print("\n[2/6] Extracting 30 URL features...")

X = []
y = []

start = time.time()

for i, row in data.iterrows():

    X.append(
        extract_features(row["URL"])
    )

    if row["Label"] == "Phishing":
        y.append(1)
    else:
        y.append(0)

    if (i + 1) % 10000 == 0:
        print(
            f"Processed {i + 1:,} / {len(data):,}"
        )

print(
    f"\nFeature extraction time: "
    f"{time.time() - start:.2f} seconds"
)

print("Number of features:", len(X[0]))


# ============================================================
# 3. TRAIN TEST SPLIT
# ============================================================

print("\n[3/6] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training:", len(X_train))
print("Testing :", len(X_test))


# ============================================================
# 4. CREATE BASE MODELS
# ============================================================

print("\n[4/6] Creating ensemble models...")

random_forest = RandomForestClassifier(
    n_estimators=250,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

extra_trees = ExtraTreesClassifier(
    n_estimators=250,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

gradient_boosting = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

hist_gradient_boosting = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42
)


# ============================================================
# 5. TRAIN INDIVIDUAL MODELS
# ============================================================

models = {
    "Random Forest": random_forest,
    "Extra Trees": extra_trees,
    "Gradient Boosting": gradient_boosting,
    "Hist Gradient Boosting": hist_gradient_boosting
}

results = {}

print("\n[5/6] Training individual models...")


for name, model in models.items():

    print("\n" + "-" * 65)
    print("Training:", name)
    print("-" * 65)

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

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

    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "time": time.time() - start
    }

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


# ============================================================
# 6. SOFT VOTING ENSEMBLE
# ============================================================

print("\n" + "=" * 65)
print("TRAINING SOFT-VOTING ENSEMBLE")
print("=" * 65)

ensemble = VotingClassifier(

    estimators=[
        ("rf", random_forest),
        ("extra", extra_trees),
        ("gb", gradient_boosting),
        ("hist", hist_gradient_boosting)
    ],

    voting="soft",

    weights=[
        2,
        2,
        3,
        3
    ],

    n_jobs=-1
)

start = time.time()

ensemble.fit(
    X_train,
    y_train
)

ensemble_time = time.time() - start


ensemble_predictions = ensemble.predict(
    X_test
)

ensemble_probabilities = ensemble.predict_proba(
    X_test
)[:, 1]


ensemble_accuracy = accuracy_score(
    y_test,
    ensemble_predictions
)

ensemble_precision = precision_score(
    y_test,
    ensemble_predictions,
    zero_division=0
)

ensemble_recall = recall_score(
    y_test,
    ensemble_predictions,
    zero_division=0
)

ensemble_f1 = f1_score(
    y_test,
    ensemble_predictions,
    zero_division=0
)

ensemble_roc_auc = roc_auc_score(
    y_test,
    ensemble_probabilities
)

ensemble_pr_auc = average_precision_score(
    y_test,
    ensemble_probabilities
)


print("\nENSEMBLE RESULTS")

print(
    f"Accuracy : {ensemble_accuracy:.4f}"
)

print(
    f"Precision: {ensemble_precision:.4f}"
)

print(
    f"Recall   : {ensemble_recall:.4f}"
)

print(
    f"F1       : {ensemble_f1:.4f}"
)

print(
    f"ROC-AUC  : {ensemble_roc_auc:.4f}"
)

print(
    f"PR-AUC   : {ensemble_pr_auc:.4f}"
)

print(
    f"Training Time: {ensemble_time:.2f} seconds"
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        ensemble_predictions
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        ensemble_predictions,
        target_names=[
            "Legitimate",
            "Phishing"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE ENSEMBLE
# ============================================================

print("\n[6/6] Saving advanced model...")

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    ensemble,
    "model/phishing_ensemble.pkl"
)


ensemble_info = {
    "model_name": "Soft Voting Ensemble",
    "features": 30,
    "dataset_size": len(data),
    "accuracy": ensemble_accuracy,
    "precision": ensemble_precision,
    "recall": ensemble_recall,
    "f1_score": ensemble_f1,
    "roc_auc": ensemble_roc_auc,
    "pr_auc": ensemble_pr_auc,
    "training_samples": len(X_train),
    "testing_samples": len(X_test)
}


joblib.dump(
    ensemble_info,
    "model/ensemble_info.pkl"
)


print("\nSaved:")
print("model/phishing_ensemble.pkl")
print("model/ensemble_info.pkl")

print("\n" + "=" * 75)
print("ADVANCED ENSEMBLE TRAINING COMPLETED")
print("=" * 75)