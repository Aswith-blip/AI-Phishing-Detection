from flask import Flask, render_template, request
import joblib

from feature_extraction import extract_features
from risk_analysis import (
    calculate_risk,
    get_detection_reasons
)


app = Flask(__name__)


# ============================================================
# LOAD FINAL MODEL
# ============================================================

model = joblib.load(
    "model/phishing_model.pkl"
)

model_info = joblib.load(
    "model/model_info.pkl"
)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(

        "index.html",

        model_name=model_info["model_name"],

        accuracy=round(
            model_info["accuracy"] * 100,
            2
        ),

        features=model_info[
            "number_of_features"
        ]
    )


# ============================================================
# PREDICT URL
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    url = request.form.get(
        "url",
        ""
    ).strip()


    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not url:

        return render_template(
            "index.html",

            error="Please enter a website URL.",

            model_name=model_info[
                "model_name"
            ],

            accuracy=round(
                model_info["accuracy"] * 100,
                2
            ),

            features=model_info[
                "number_of_features"
            ]
        )


    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    features = extract_features(
        url
    )


    # --------------------------------------------------------
    # ML prediction
    # --------------------------------------------------------

    prediction = model.predict(
        [features]
    )[0]


    probabilities = model.predict_proba(
        [features]
    )[0]


    phishing_probability = (
        probabilities[1] * 100
    )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if prediction == 1:

        result = (
            "PHISHING WEBSITE"
        )

    else:

        result = (
            "LEGITIMATE WEBSITE"
        )


    # --------------------------------------------------------
    # Risk calculation
    # --------------------------------------------------------

    risk_score, risk_level = calculate_risk(

        features,

        phishing_probability
    )


    # --------------------------------------------------------
    # Detection reasons
    # --------------------------------------------------------

    reasons = get_detection_reasons(
        features
    )


    # --------------------------------------------------------
    # Result type
    # --------------------------------------------------------

    if prediction == 1:

        result_type = "danger"

    else:

        result_type = "safe"


    # --------------------------------------------------------
    # Render result
    # --------------------------------------------------------

    return render_template(

        "index.html",

        url=url,

        result=result,

        result_type=result_type,

        probability=round(
            phishing_probability,
            2
        ),

        risk_score=risk_score,

        risk_level=risk_level,

        reasons=reasons,

        model_name=model_info[
            "model_name"
        ],

        accuracy=round(
            model_info["accuracy"] * 100,
            2
        ),

        features=model_info[
            "number_of_features"
        ]
    )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )