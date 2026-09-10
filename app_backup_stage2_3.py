from flask import Flask, render_template, request

from cnn_predictor import predict_url
from feature_extraction import extract_features
from risk_analysis import calculate_risk, get_detection_reasons

from database import (
    save_scan,
    get_recent_scans,
    get_statistics
)

from url_validator import validate_url

import joblib


app = Flask(__name__)
@app.after_request
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



# Load CNN model information
try:
    model_info = joblib.load("model/cnn_info.pkl")
except Exception:
    model_info = {}


MODEL_NAME = model_info.get(
    "model_name",
    "Character-Level CNN"
)

ACCURACY = model_info.get(
    "accuracy",
    0.998
)


@app.route("/")
def home():

    return render_template(
        "index.html",
        model_name=MODEL_NAME,
        accuracy=round(ACCURACY * 100, 2)
    )


@app.route("/predict", methods=["POST"])
def predict():

    raw_url = request.form.get("url", "").strip()

    # Validate URL before prediction
    valid, url, validation_error = validate_url(raw_url)

    if not valid:

        return render_template(
            "index.html",
            url=raw_url,
            error=validation_error,
            model_name=MODEL_NAME,
            accuracy=round(ACCURACY * 100, 2)
        )

    try:

        # CNN prediction
        prediction_result = predict_url(url)

        phishing_probability = prediction_result[
            "phishing_probability"
        ]

        legitimate_probability = prediction_result[
            "legitimate_probability"
        ]

        # Extract URL features
        features = extract_features(url)

        # Calculate risk
        risk_score, risk_level = calculate_risk(
            features,
            phishing_probability
        )

        # Detection reasons
        reasons = get_detection_reasons(features)

        # Save scan
        save_scan(
            url=url,
            prediction=prediction_result["prediction"],
            phishing_probability=phishing_probability,
            legitimate_probability=legitimate_probability,
            risk_score=risk_score,
            risk_level=risk_level
        )

        # Display result
        return render_template(
            "index.html",
            url=url,
            result=prediction_result["result"],
            result_type=prediction_result["result_type"],
            probability=phishing_probability,
            legitimate_probability=legitimate_probability,
            risk_score=risk_score,
            risk_level=risk_level,
            reasons=reasons,
            model_name=MODEL_NAME,
            accuracy=round(ACCURACY * 100, 2)
        )

    except Exception as e:

        return render_template(
            "index.html",
            url=url,
            error=f"Prediction error: {str(e)}",
            model_name=MODEL_NAME,
            accuracy=round(ACCURACY * 100, 2)
        )


@app.route("/history")
def history():

    scans = get_recent_scans(20)

    statistics = get_statistics()

    return render_template(
        "history.html",
        scans=scans,
        statistics=statistics,
        model_name=MODEL_NAME
    )


if __name__ == "__main__":

    app.run(
        debug=False,
        use_reloader=False
    )

