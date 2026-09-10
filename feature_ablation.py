import time
import joblib
import pandas as pd
import numpy as np

from feature_extraction import extract_features

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


FEATURE_NAMES = [
    "URL Length",
    "Domain Length",
    "Number of Dots",
    "Number of Hyphens",
    "Number of Digits",
    "Number of Letters",
    "Special Characters",
    "Number of Subdomains",
    "HTTPS",
    "IP Address",
    "@ Symbol",
    "Question Marks",
    "Equals Signs",
    "Ampersands",
    "Percent Encoding",
    "Path Length",
    "Query Length",
    "Fragment",
    "Suspicious Keywords",
    "URL Depth",
    "Consecutive Dots",
    "Consecutive Hyphens",
    "Domain Contains Digits",
    "Domain Hyphen Count",
    "Path Digit Count",
    "Query Parameter Count",
    "URL Shortener",
    "Longest Numeric Sequence",
    "Letter-Digit Ratio",
    "Character Diversity"
]


print("=" * 75)
print("FEATURE ABLATION ANALYSIS")
print("=" * 75)

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

print("\n[1/5] Loading dataset...")

data = pd.read_csv(
    "dataset/phishing_dataset.csv"
)

print("Total URLs:", len(data))

# ---------------------------------------------------------
# 2. EXTRACT FEATURES
# ---------------------------------------------------------

print("\n[2/5] Extracting features...")

X = []
y = []

start = time.time()

for i, row in data.iterrows():

    X.append(
        extract_features(row["URL"])
    )

    y.append(
        1 if row["Label"] == "Phishing" else 0
    )

    if (i + 1) % 20000 == 0:
        print(
            f"Processed {i + 1:,} / {len(data):,}"
        )

X = np.array(X)
y = np.array(y)

print(
    f"Feature extraction time: "
    f"{time.time() - start:.2f} seconds"
)

print("Feature matrix:", X.shape)

# ---------------------------------------------------------
# 3. SAME TRAIN/TEST SPLIT FOR ALL EXPERIMENTS
# ---------------------------------------------------------

print("\n[3/5] Creating fixed train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training:", len(X_train))
print("Testing :", len(X_test))

# ---------------------------------------------------------
# 4. EXPERIMENT FUNCTION
# ---------------------------------------------------------

def run_experiment(
    name,
    removed_features
):

    print("\n" + "-" * 70)
    print("Experiment:", name)
    print("-" * 70)

    columns_to_remove = [
        FEATURE_NAMES.index(feature)
        for feature in removed_features
    ]

    columns_to_keep = [
        i for i in range(len(FEATURE_NAMES))
        if i not in columns_to_remove
    ]

    X_train_exp = X_train[:, columns_to_keep]
    X_test_exp = X_test[:, columns_to_keep]

    print(
        "Removed:",
        removed_features
    )

    print(
        "Remaining features:",
        len(columns_to_keep)
    )

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    start = time.time()

    model.fit(
        X_train_exp,
        y_train
    )

    predictions = model.predict(
        X_test_exp
    )

    probabilities = model.predict_proba(
        X_test_exp
    )[:, 1]

    elapsed = time.time() - start

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
        f"Training Time: {elapsed:.2f} seconds"
    )

    return {
        "Experiment": name,
        "Removed Features": ", ".join(
            removed_features
        ),
        "Remaining Features": len(
            columns_to_keep
        ),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    }


# ---------------------------------------------------------
# 5. RUN ABLATION EXPERIMENTS
# ---------------------------------------------------------

print("\n[4/5] Running experiments...")

results = []

# Baseline
results.append(
    run_experiment(
        "Baseline - All Features",
        []
    )
)

# Remove HTTPS
results.append(
    run_experiment(
        "Without HTTPS",
        ["HTTPS"]
    )
)

# Remove Path Length
results.append(
    run_experiment(
        "Without Path Length",
        ["Path Length"]
    )
)

# Remove both
results.append(
    run_experiment(
        "Without HTTPS + Path Length",
        [
            "HTTPS",
            "Path Length"
        ]
    )
)

# Remove several dominant/simple URL features
results.append(
    run_experiment(
        "Without HTTPS + Path + URL Length",
        [
            "HTTPS",
            "Path Length",
            "URL Length"
        ]
    )
)

# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

print("\n[5/5] Saving ablation results...")

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "model/feature_ablation_results.csv",
    index=False
)

print("\n" + "=" * 75)
print("FEATURE ABLATION SUMMARY")
print("=" * 75)

print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(
    "model/feature_ablation_results.csv"
)

print("\n" + "=" * 75)
print("FEATURE ABLATION COMPLETED")
print("=" * 75)