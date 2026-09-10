from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask import Flask, render_template, request
from cnn_predictor import predict_url
from feature_extraction import extract_features
from risk_analysis import calculate_risk_score, get_detection_reasons
from database import save_scan, get_recent_scans, get_statistics
from url_validator import validate_url
import joblib
import os


app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


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


MODEL_INFO_PATH = os.path.join("model", "cnn_info.pkl")

try:
    model_info = joblib.load(MODEL_INFO_PATH)
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

    url = request.form.get("url", "").strip()

    valid, cleaned_url, error = validate_url(url)

    if not valid:
        return render_template(
            "index.html",
            url=url,
            error=error,
            model_name=MODEL_NAME,
            accuracy=round(ACCURACY * 100, 2)
        )

    url = cleaned_url

    try:

        result = predict_url(url)

        confidence = result.get(
            "confidence",
            "VERY HIGH"
        )

        features = extract_features(url)

        risk_score, risk_level = calculate_risk_score(
            features
        )

        reasons = get_detection_reasons(
            features
        )

        save_scan(
            url,
            result["prediction"],
            result["phishing_probability"],
            result["legitimate_probability"],
            risk_score,
            risk_level
        )

        return render_template(
            "index.html",
            url=url,
            result=result,
            confidence=confidence,
            risk_score=risk_score,
            risk_level=risk_level,
            reasons=reasons,
            model_name=MODEL_NAME,
            accuracy=round(ACCURACY * 100, 2)
        )

    except Exception:

        return render_template(
            "index.html",
            url=url,
            error="Unable to analyze this URL. Please check the URL and try again.",
            model_name=MODEL_NAME,
            accuracy=round(ACCURACY * 100, 2)
        )


@app.route("/history")
def history():

    scans = get_recent_scans()

    statistics = get_statistics()

    return render_template(
        "history.html",
        scans=scans,
        statistics=statistics,
        model_name=MODEL_NAME,
        accuracy=round(ACCURACY * 100, 2)
    )

if __name__ == "__main__":

    import os
from waitress import serve
import os
from waitress import serve

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
    )
import os
from waitress import serve

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)import os
from waitress import serve

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    serve(app, host="0.0.0.0", port=port)
