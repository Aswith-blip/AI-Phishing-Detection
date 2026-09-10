import unittest

import app
from feature_extraction import extract_features
from risk_analysis import calculate_risk
from cnn_predictor import predict_url
from url_validator import validate_url


class TestURLValidator(unittest.TestCase):

    def test_valid_https_url(self):
        valid, url, error = validate_url("https://www.google.com")
        self.assertTrue(valid)
        self.assertEqual(error, "")

    def test_domain_without_scheme(self):
        valid, url, error = validate_url("google.com")
        self.assertTrue(valid)
        self.assertTrue(url.startswith("https://"))

    def test_reject_spaces(self):
        valid, url, error = validate_url("hello world")
        self.assertFalse(valid)

    def test_reject_localhost(self):
        valid, url, error = validate_url("http://localhost")
        self.assertFalse(valid)

    def test_reject_loopback_ip(self):
        valid, url, error = validate_url("http://127.0.0.1")
        self.assertFalse(valid)

    def test_reject_invalid_domain(self):
        valid, url, error = validate_url("https://-example.com")
        self.assertFalse(valid)


class TestFeatureExtraction(unittest.TestCase):

    def test_feature_count(self):
        features = extract_features("https://www.google.com")
        self.assertEqual(len(features), 30)


class TestRiskAnalysis(unittest.TestCase):

    def test_risk_score_range(self):
        features = extract_features("https://www.google.com")
        result = calculate_risk(features, 0.019)

        if isinstance(result, tuple):
            risk_score = result[0]
        else:
            risk_score = result

        self.assertGreaterEqual(risk_score, 0)
        self.assertLessEqual(risk_score, 100)


class TestCNNPrediction(unittest.TestCase):

    def test_google_prediction(self):
        result = predict_url("https://www.google.com")

        self.assertIn("prediction", result)
        self.assertIn("phishing_probability", result)
        self.assertIn("legitimate_probability", result)

    def test_prediction_confidence(self):
        result = predict_url("https://www.google.com")

        self.assertIn("confidence", result)
        self.assertIn("confidence_probability", result)

        self.assertIn(
            result["confidence"],
            ["VERY HIGH", "HIGH", "MODERATE", "LOW"]
        )

        self.assertGreaterEqual(
            result["confidence_probability"],
            0
        )

        self.assertLessEqual(
            result["confidence_probability"],
            100
        )


class TestFlaskApplication(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.app.test_client()

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_history_page(self):
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)

    def test_predict_page(self):
        response = self.client.post(
            "/predict",
            data={"url": "https://www.google.com"}
        )

        self.assertEqual(response.status_code, 200)

    def test_invalid_url(self):
        response = self.client.post(
            "/predict",
            data={"url": "hello world"}
        )

        self.assertEqual(response.status_code, 200)

        self.assertIn(
            b"URL must not contain spaces.",
            response.data
        )

    def test_security_headers(self):
        response = self.client.get("/")

        self.assertEqual(
            response.headers.get("X-Content-Type-Options"),
            "nosniff"
        )

        self.assertEqual(
            response.headers.get("X-Frame-Options"),
            "SAMEORIGIN"
        )

        self.assertEqual(
            response.headers.get("Referrer-Policy"),
            "strict-origin-when-cross-origin"
        )

        self.assertIn(
            "default-src 'self'",
            response.headers.get(
                "Content-Security-Policy",
                ""
            )
        )

    def test_rate_limit_configured(self):
        self.assertTrue(hasattr(app, "limiter"))
        self.assertIsNotNone(app.limiter)


if __name__ == "__main__":
    unittest.main(verbosity=2)
