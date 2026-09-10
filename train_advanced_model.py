import os
import time
import joblib
import pandas as pd

from feature_extraction import extract_features

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


print("=" * 70)
print("ADVANCED AI PHISHING WEBSITE DETECTION")
print("MODEL TRAINING PIPELINE")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n[1/7] Loading dataset...")

data = pd.read_csv(
    "dataset/phishing_dataset.csv"
)

print("Dataset loaded successfully!")
print("Total URLs:", len(data))

print("\nClass distribution:")
print(data["Label"].value_counts())


# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================

print("\n[2/7] Extracting 30 URL features...")

start_time = time.time()

X = []
y = []

for index, row in data.iterrows():

    url = row["URL"]
    label = row["Label"]

    X.append(
        extract_features(url)
    )

    # Phishing = 1
    # Legitimate = 0
    if label == "Phishing":
        y.append(1)
    else:
        y.append(0)

    if (index + 1) % 10000 == 0:

        print(
            f"Processed {index + 1:,} / "
            f"{len(data):,} URLs"
        )


elapsed = time.time() - start_time

print("\nFeature extraction completed!")
print("Features per URL:", len(X[0]))
print(
    f"Extraction time: {elapsed:.2f} seconds"
)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

print("\n[3/7] Creating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 4. DEFINE MODELS
# ============================================================

print("\n[4/7] Preparing ML models...")

models = {

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=None,
            min_samples_split=2,
            random_state=42
        ),

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
}


# ============================================================
# 5. TRAIN MODELS
# ============================================================

print("\n[5/7] Training models...")

results = {}

best_model = None
best_model_name = None
best_f1 = -1


for name, model in models.items():

    print("\n" + "-" * 70)
    print("Training:", name)
    print("-" * 70)

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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


    results[name] = {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC_AUC": roc_auc,

        "Training_Time": training_time
    }


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
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
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


    # --------------------------------------------------------
    # Best model based on F1
    # --------------------------------------------------------

    if f1 > best_f1:

        best_f1 = f1

        best_model = model

        best_model_name = name


# ============================================================
# 6. MODEL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

comparison = []

for name, result in results.items():

    comparison.append({

        "Model": name,

        "Accuracy": result["Accuracy"],

        "Precision": result["Precision"],

        "Recall": result["Recall"],

        "F1 Score": result["F1"],

        "ROC-AUC": result["ROC_AUC"],

        "Training Time": result["Training_Time"]
    })


comparison_df = pd.DataFrame(
    comparison
)

print(
    comparison_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format,
            "Training Time": "{:.2f}".format
        }
    )
)


print("\nBest Model:")
print(best_model_name)

print(
    "Best F1 Score:",
    round(best_f1, 4)
)


# ============================================================
# 7. SAVE MODEL
# ============================================================

print("\n[7/7] Saving final model...")

os.makedirs(
    "model",
    exist_ok=True
)


model_path = (
    "model/phishing_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)


# Save model information

best_result = results[
    best_model_name
]


model_info = {

    "model_name":
        best_model_name,

    "accuracy":
        best_result["Accuracy"],

    "precision":
        best_result["Precision"],

    "recall":
        best_result["Recall"],

    "f1_score":
        best_result["F1"],

    "roc_auc":
        best_result["ROC_AUC"],

    "number_of_features":
        30,

    "dataset_size":
        len(data),

    "training_samples":
        len(X_train),

    "testing_samples":
        len(X_test)
}


joblib.dump(
    model_info,
    "model/model_info.pkl"
)


# Save comparison results

comparison_df.to_csv(
    "model/model_comparison.csv",
    index=False
)


print("\nFinal model saved:")
print(model_path)

print(
    "Model information saved:"
)

print(
    "model/model_info.pkl"
)

print(
    "Comparison saved:"
)

print(
    "model/model_comparison.csv"
)


print("\n" + "=" * 70)
print("ADVANCED MODEL TRAINING COMPLETED")
print("=" * 70)