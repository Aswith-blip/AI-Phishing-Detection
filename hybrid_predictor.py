import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from feature_extraction import extract_features


print("Loading hybrid model...")

model = joblib.load(
    "model/phishing_hybrid_model.pkl"
)

vectorizer = joblib.load(
    "model/hybrid_tfidf_vectorizer.pkl"
)

scaler = joblib.load(
    "model/hybrid_scaler.pkl"
)

print("Hybrid model loaded successfully!")


def predict_url(url):

    url = str(url).strip()

    # Extract 30 structural URL features
    structural = np.array(
        extract_features(url),
        dtype=float
    ).reshape(1, -1)

    # Scale structural features
    structural_scaled = scaler.transform(
        structural
    )

    # Extract character-level TF-IDF features
    text_features = vectorizer.transform(
        [url]
    )

    # Convert structural features to sparse format
    structural_sparse = csr_matrix(
        structural_scaled
    )

    # Combine 60,000 TF-IDF + 30 structural features
    combined_features = hstack([
        text_features,
        structural_sparse
    ])

    # Predict
    prediction = model.predict(
        combined_features
    )[0]

    probabilities = model.predict_proba(
        combined_features
    )[0]

    legitimate_probability = (
        probabilities[0] * 100
    )

    phishing_probability = (
        probabilities[1] * 100
    )

    if prediction == 1:
        result = "PHISHING WEBSITE"
        result_type = "danger"
    else:
        result = "LEGITIMATE WEBSITE"
        result_type = "safe"

    return {
        "prediction": int(prediction),
        "result": result,
        "result_type": result_type,
        "phishing_probability": round(
            phishing_probability,
            2
        ),
        "legitimate_probability": round(
            legitimate_probability,
            2
        )
    }


if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("HYBRID PHISHING DETECTOR TEST")
    print("=" * 70)

    test_urls = [
        "https://www.google.com",
        "https://www.microsoft.com",
        "http://secure-login-example.com/verify-account"
    ]

    for url in test_urls:

        print("\nURL:")
        print(url)

        result = predict_url(url)

        print(
            "Prediction:",
            result["result"]
        )

        print(
            "Phishing Probability:",
            str(
                result["phishing_probability"]
            ) + "%"
        )

        print(
            "Legitimate Probability:",
            str(
                result["legitimate_probability"]
            ) + "%"
        )

        print("-" * 70)