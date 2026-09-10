import pandas as pd
import joblib
import os

from feature_extraction import extract_features

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("AI PHISHING WEBSITE DETECTION - MODEL TRAINING")
print("=" * 60)

print("\nLoading dataset...")

data = pd.read_csv("dataset/phishing_dataset.csv")

print("Dataset loaded successfully!")
print("Total URLs:", len(data))

print("\nClass distribution:")
print(data["Label"].value_counts())


# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================

print("\nExtracting URL features...")

X = []
y = []

for index, row in data.iterrows():

    url = row["URL"]
    label = row["Label"]

    X.append(extract_features(url))

    # Phishing = 1
    # Legitimate = 0
    if label == "Phishing":
        y.append(1)
    else:
        y.append(0)

    if (index + 1) % 10000 == 0:
        print(f"Processed {index + 1} URLs...")


print("\nFeature extraction completed!")

print("Total samples:", len(X))
print("Number of features:", len(X[0]))


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 4. DEFINE MODELS
# ============================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}


# ============================================================
# 5. TRAIN AND COMPARE MODELS
# ============================================================

results = {}

best_model = None
best_model_name = None
best_f1 = 0


for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
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

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Legitimate", "Phishing"],
            zero_division=0
        )
    )

    # Select best model based on F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name


# ============================================================
# 6. MODEL COMPARISON
# ============================================================

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for name, result in results.items():

    print(
        f"{name:20} "
        f"Accuracy={result['Accuracy']:.4f} "
        f"Precision={result['Precision']:.4f} "
        f"Recall={result['Recall']:.4f} "
        f"F1={result['F1 Score']:.4f}"
    )


print("\nBest Model:", best_model_name)
print("Best F1 Score:", round(best_f1, 4))


# ============================================================
# 7. SAVE BEST MODEL
# ============================================================

os.makedirs("model", exist_ok=True)

model_path = "model/phishing_model.pkl"

joblib.dump(best_model, model_path)

print("\nBest model saved successfully!")
print("Model:", model_path)


# ============================================================
# 8. SAVE MODEL INFORMATION
# ============================================================

model_info = {
    "model_name": best_model_name,
    "accuracy": results[best_model_name]["Accuracy"],
    "precision": results[best_model_name]["Precision"],
    "recall": results[best_model_name]["Recall"],
    "f1_score": results[best_model_name]["F1 Score"],
    "number_of_features": len(X[0]),
    "dataset_size": len(data)
}

joblib.dump(
    model_info,
    "model/model_info.pkl"
)

print("Model information saved!")@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self' data:; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self';"
    )
    return response
print("File: model/model_info.pkl")

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)