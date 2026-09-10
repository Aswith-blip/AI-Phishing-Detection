import joblib
import pandas as pd
import numpy as np

from feature_extraction import extract_features


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


def get_feature_names():
    return FEATURE_NAMES


def load_model():
    return joblib.load(
        "model/phishing_model.pkl"
    )


def get_feature_importance(model):
    """
    Get model-specific feature importance
    from Gradient Boosting.
    """

    importance = model.feature_importances_

    results = []

    for index, value in enumerate(importance):

        results.append({
            "feature": FEATURE_NAMES[index],
            "importance": float(value)
        })

    results.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    return results


def get_top_features(model, limit=10):

    importance = get_feature_importance(
        model
    )

    return importance[:limit]


def explain_url(model, features, limit=5):
    """
    Combine the URL's actual characteristics
    with globally important model features.
    """

    importance = get_feature_importance(
        model
    )

    explanations = []

    for item in importance:

        index = FEATURE_NAMES.index(
            item["feature"]
        )

        value = features[index]

        if value == 0:
            continue

        explanations.append({
            "feature": item["feature"],
            "value": float(value),
            "importance": round(
                item["importance"] * 100,
                2
            )
        })

        if len(explanations) >= limit:
            break

    return explanations


if __name__ == "__main__":

    print("=" * 65)
    print("PHISHING MODEL FEATURE IMPORTANCE")
    print("=" * 65)

    model = load_model()

    print("\nTop 10 important features:\n")

    top_features = get_top_features(
        model,
        10
    )

    for position, item in enumerate(
        top_features,
        start=1
    ):

        print(
            f"{position}. "
            f"{item['feature']} "
            f"-> "
            f"{item['importance'] * 100:.2f}%"
        )

    test_url = (
        "http://secure-login-example.com/"
        "verify-account"
    )

    features = extract_features(
        test_url
    )

    print("\n" + "=" * 65)
    print("URL-SPECIFIC EXPLANATION")
    print("=" * 65)

    explanations = explain_url(
        model,
        features
    )

    for item in explanations:

        print(
            f"- {item['feature']}: "
            f"value={item['value']}, "
            f"importance={item['importance']}%"
        )