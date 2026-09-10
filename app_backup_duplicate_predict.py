from database import (
    save_scan,
    get_recent_scans,
    get_statistics
)
from flask import Flask, render_template, request

from cnn_predictor import predict_url
from feature_extraction import extract_features
from risk_analysis import calculate_risk, get_detection_reasons

from database import (
    save_scan,
    get_recent_scans,
    get_statistics
)

import joblib


app = Flask(__name__)


# =========================================================
# MODEL INFORMATION
# =========================================================

try:

    model_info = joblib.load(
        "model/cnn_info.pkl"
    )

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


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        model_name=MODEL_NAME,
        accuracy=round(
            ACCURACY * 100,
            2
        )
    )


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    url = request.form.get(
        "url",
        ""
    ).strip()


    if not url:

        return render_template(
            "index.html",
            error="Please enter a website URL.",
            model_name=MODEL_NAME,
            accuracy=round(
                ACCURACY * 100,
                2
            )
        )


    try:

        # CNN prediction
        prediction_result = predict_url(
            url
        )


        phishing_probability = (
            prediction_result[
                "phishing_probability"
            ]
        )


        legitimate_probability = (
            prediction_result[
                "legitimate_probability"
            ]
        )


        # Extract URL features
        features = extract_features(
            url
        )


        # Risk analysis
        risk_score, risk_level = calculate_risk(
            features,
            phishing_probability
        )


        # Detection reasons
        reasons = get_detection_reasons(
            features
        )


        # Save scan
        save_scan(
            url=url,
            prediction=prediction_result[
                "prediction"
            ],
            phishing_probability=phishing_probability,
            legitimate_probability=legitimate_probability,
            risk_score=risk_score,
            risk_level=risk_level
        )


        return render_template(
            "index.html",
            url=url,
            result=prediction_result[
                "result"
            ],
            result_type=prediction_result[
                "result_type"
            ],
            probability=phishing_probability,
            legitimate_probability=legitimate_probability,
            risk_score=risk_score,
            risk_level=risk_level,
            reasons=reasons,
            model_name=MODEL_NAME,
            accuracy=round(
                ACCURACY * 100,
                2
            )
        )


    except Exception as e:

        return render_template(
            "index.html",
            error=f"Prediction error: {str(e)}",
            model_name=MODEL_NAME,
            accuracy=round(
                ACCURACY * 100,
                2
            )
        )


# =========================================================
# HISTORY / DASHBOARD
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    raw_url = request.form.get("url", "").strip()

    # Validate URL before sending it to the AI model
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

        phishing_probability = prediction_result["phishing_probability"]
        legitimate_probability = prediction_result["legitimate_probability"]

        # Extract structural URL features
        features = extract_features(url)

        # Calculate application-level risk score
        risk_score, risk_level = calculate_risk(
            features,
            phishing_probability
        )

        # Generate human-readable reasons
        reasons = get_detection_reasons(features)

        # Save scan to database
        save_scan(
            url=url,
            prediction=prediction_result["prediction"],
            phishing_probability=phishing_probability,
            legitimate_probability=legitimate_probability,
            risk_score=risk_score,
            risk_level=risk_level
        )

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

# =========================================================
# START FLASK
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=False,
        use_reloader=False
    )