import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# LOAD CNN MODEL
# =========================================================

print("Loading CNN phishing detection model...")

model = tf.keras.models.load_model(
    "model/phishing_cnn.keras"
)

with open(
    "model/cnn_tokenizer.pkl",
    "rb"
) as f:
    tokenizer = pickle.load(f)

with open(
    "model/cnn_info.pkl",
    "rb"
) as f:
    model_info = pickle.load(f)


MAX_LENGTH = model_info.get(
    "max_length",
    200
)


print("CNN model loaded successfully!")


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_url(url):

    url = str(url).strip()

    if not url:
        raise ValueError(
            "URL cannot be empty."
        )

    # Convert URL into character sequence
    sequence = tokenizer.texts_to_sequences(
        [url]
    )

    # Pad sequence
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    # CNN prediction
    probability = float(
        model.predict(
            padded,
            verbose=0
        )[0][0]
    )

    # IMPORTANT:
    # Training uses:
    # 0 = Legitimate
    # 1 = Phishing
    #
    # If your training script used the opposite mapping,
    # we will correct it after testing.

    phishing_probability = probability
    legitimate_probability = 1 - probability

    if phishing_probability >= 0.5:

        prediction = "PHISHING WEBSITE"
        result_type = "danger"

    else:

        prediction = "LEGITIMATE WEBSITE"
        result_type = "safe"

    return {
        "prediction": prediction,
        "result": prediction,
        "result_type": result_type,
        "phishing_probability": round(
            phishing_probability * 100,
            2
        ),
        "legitimate_probability": round(
            legitimate_probability * 100,
            2
        )
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("CNN PHISHING DETECTOR TEST")
    print("=" * 70)

    test_urls = [

        "https://www.google.com",

        "https://www.microsoft.com",

        "http://secure-login-example.com/verify-account",

        "http://paypal-login-security-example.com/account/verify"

    ]

    for url in test_urls:

        result = predict_url(url)

        print()
        print("URL:")
        print(url)

        print(
            "Prediction:",
            result["prediction"]
        )

        print(
            "Phishing Probability:",
            f'{result["phishing_probability"]}%'
        )

        print(
            "Legitimate Probability:",
            f'{result["legitimate_probability"]}%'
        )