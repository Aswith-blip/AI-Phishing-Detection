import numpy as np


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


def calculate_feature_contribution(model, features):

    features = np.array(
        features,
        dtype=float
    )

    baseline_probability = model.predict_proba(
        [features]
    )[0][1]

    contributions = []

    for index in range(len(features)):

        modified = features.copy()

        modified[index] = 0

        modified_probability = model.predict_proba(
            [modified]
        )[0][1]

        contribution = (
            baseline_probability -
            modified_probability
        )

        contributions.append(contribution)

    results = []

    for index, contribution in enumerate(
        contributions
    ):

        results.append({
            "feature": FEATURE_NAMES[index],
            "index": index,
            "value": features[index],
            "contribution": float(contribution)
        })

    results.sort(
        key=lambda x: abs(
            x["contribution"]
        ),
        reverse=True
    )

    return results


def get_top_explanations(
    model,
    features,
    limit=5
):

    contributions = calculate_feature_contribution(
        model,
        features
    )

    explanations = []

    for item in contributions[:limit]:

        contribution = item["contribution"]

        if contribution > 0:
            direction = "increases phishing risk"
        else:
            direction = "reduces phishing risk"

        explanations.append({
            "feature": item["feature"],
            "value": item["value"],
            "contribution": round(
                contribution * 100,
                2
            ),
            "direction": direction
        })

    return explanations