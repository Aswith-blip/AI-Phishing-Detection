import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


MODEL_PATH = os.path.join("model", "phishing_cnn.keras")
TOKENIZER_PATH = os.path.join("model", "cnn_tokenizer.pkl")
INFO_PATH = os.path.join("model", "cnn_info.pkl")


print("Loading CNN phishing detection model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(INFO_PATH, "rb") as f:
    model_info = pickle.load(f)

MAX_LENGTH = model_info.get("max_length", 200)

print("CNN model loaded successfully!")


def get_confidence_level(probability):
    """
    Convert phishing probability into a simple confidence level.
    """
    if probability >= 90:
        return "VERY HIGH"
    elif probability >= 75:
        return "HIGH"
    elif probability >= 60:
        return "MODERATE"
    else:
        return "LOW"


def predict_url(url):
    """
    Predict whether a URL is phishing or legitimate.
    """

    sequence = tokenizer.texts_to_sequences([url])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    probability = float(model.predict(padded_sequence, verbose=0)[0][0])

    phishing_probability = probability * 100
    legitimate_probability = (1 - probability) * 100

    if probability >= 0.5:
        prediction = "PHISHING WEBSITE"
        result_type = "danger"
    else:
        prediction = "LEGITIMATE WEBSITE"
        result_type = "safe"

    confidence_probability = max(
        phishing_probability,
        legitimate_probability
    )

    confidence = get_confidence_level(confidence_probability)

    return {
        "prediction": prediction,
        "result": prediction,
        "result_type": result_type,
        "phishing_probability": round(phishing_probability, 2),
        "legitimate_probability": round(legitimate_probability, 2),
        "confidence": confidence,
        "confidence_probability": round(confidence_probability, 2)
    }


if __name__ == "__main__":

    test_urls = [
        "https://www.google.com",
        "https://www.microsoft.com",
        "http://secure-login-example.com/verify-account",
        "http://paypal-login-security-example.com/account/verify"
    ]

    for url in test_urls:

        result = predict_url(url)

        print()
        print("URL:", url)
        print("Prediction:", result["prediction"])
        print(
            "Phishing Probability:",
            result["phishing_probability"],
            "%"
        )
        print(
            "Legitimate Probability:",
            result["legitimate_probability"],
            "%"
        )
        print("Confidence:", result["confidence"])