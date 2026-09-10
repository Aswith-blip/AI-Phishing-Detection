# AI-Powered Phishing Website Detection System

An AI-powered web application for detecting potentially phishing URLs using machine learning and deep learning techniques.

## Project Overview

Phishing attacks commonly use deceptive URLs to trick users into visiting malicious websites. This project analyzes URL characteristics and uses a Character-Level Convolutional Neural Network (CNN) to classify URLs as either legitimate or phishing.

The system provides:

- URL validation
- AI-based phishing prediction
- Phishing and legitimate probabilities
- AI confidence level
- Risk score and risk classification
- Human-readable detection reasons
- Scan history
- Security analytics dashboard
- Model performance information
- Security headers
- Request rate limiting

## Technologies Used

- Python
- Flask
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- SQLite
- HTML
- CSS
- JavaScript
- Chart.js

## Dataset

The project uses the PhiUSIIL Phishing URL Dataset from the UCI Machine Learning Repository.

Dataset statistics:

- Total cleaned URLs: 235,370
- Legitimate URLs: 134,850
- Phishing URLs: 100,520
- Original features: 54
- URL-only features used by the application: 30

Duplicate and missing URL records were removed during preprocessing.

## Machine Learning Models

The project evaluates multiple approaches:

1. Gradient Boosting
2. Soft Voting Ensemble
3. Hybrid Character TF-IDF + URL Features
4. Character-Level CNN
5. Character-Level Transformer

## Model Comparison

| Model | Accuracy | F1 Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| Gradient Boosting | 99.62% | 99.56% | 99.80% | 99.84% |
| Soft Voting Ensemble | 99.61% | 99.54% | 99.78% | 99.83% |
| Hybrid TF-IDF + URL | 99.76% | 99.72% | 99.92% | 99.94% |
| Character CNN | 99.80% | 99.77% | 99.88% | 99.91% |
| Transformer | 99.72% | 99.67% | 99.85% | 99.89% |

The Character-Level CNN was selected as the primary production model because it achieved the highest held-out test accuracy and F1 score.

The Hybrid model achieved the highest ROC-AUC and PR-AUC and is retained as an important benchmark.

## Character-Level CNN

The CNN processes URLs at the character level, allowing the model to learn patterns directly from URL strings.

Architecture:

Input URL
?
Character Tokenization
?
Embedding
?
Conv1D
?
Batch Normalization
?
Conv1D
?
Batch Normalization
?
Global Max Pooling
?
Dense Layers
?
Dropout
?
Sigmoid Output

## CNN Test Results

- Accuracy: 99.80%
- Precision: 100.00%
- Recall: 99.54%
- F1 Score: 99.77%
- ROC-AUC: 99.88%
- PR-AUC: 99.91%

Confusion matrix:

| | Predicted Phishing | Predicted Legitimate |
|---|---:|---:|
| Actual Phishing | 26,970 | 0 |
| Actual Legitimate | 93 | 20,011 |

## Domain-Aware Validation

A separate domain-aware validation experiment was performed using completely disjoint domains between training and testing data.

Results:

- Accuracy: 99.59%
- Precision: 99.93%
- Recall: 99.10%
- F1 Score: 99.51%
- ROC-AUC: 99.77%
- PR-AUC: 99.81%

There was zero overlap between training and testing domains.

This provides evidence of strong generalization to previously unseen domains, although it should not be interpreted as guaranteed real-world accuracy.

## URL Security Validation

The application validates URLs before prediction.

It rejects:

- URLs containing spaces
- localhost URLs
- loopback IP addresses
- private IP addresses
- link-local IP addresses
- malformed domain names
- invalid domain characters
- excessively long URLs

URLs without a scheme are normalized by adding HTTPS.

## Application Security

The Flask application implements:

- Content Security Policy
- X-Content-Type-Options
- X-Frame-Options
- Referrer-Policy
- HTTP-only session cookies
- SameSite cookies
- Request size limitation
- Flask-Limiter rate limiting
- URL validation

The application does not open or execute the submitted website. Prediction is performed from the URL itself.

## Database

SQLite is used to store scan history.

Stored information includes:

- URL
- Prediction
- Phishing probability
- Legitimate probability
- Risk score
- Risk level
- Scan timestamp

## Project Structure

AI-Phishing-Detection/
¦
+-- dataset/
+-- model/
+-- templates/
+-- static/
+-- tests/
¦
+-- app.py
+-- feature_extraction.py
+-- train_model.py
+-- train_advanced_model.py
+-- train_ensemble_model.py
+-- train_calibrated_model.py
+-- risk_analysis.py
+-- explainability.py
+-- feature_importance.py
+-- feature_ablation.py
+-- train_hybrid_model.py
+-- hybrid_predictor.py
+-- train_cnn_model.py
+-- cnn_predictor.py
+-- train_transformer_model.py
+-- domain_validation.py
+-- database.py
+-- url_validator.py
+-- prepare_dataset.py
+-- download_dataset.py
+-- requirements.txt
+-- README.md

## Installation

Clone or copy the project and open PowerShell in the project directory.

Install dependencies:

    pip install -r requirements.txt

## Running the Application

Start the Flask application:

    python app.py

Then open the local application in a browser.

## Running Tests

Run the automated test suite:

    python -m unittest discover -s tests -p "test_*.py" -v

The current test suite contains 16 automated tests.

## Example Predictions

Example legitimate URL:

    https://www.google.com

Example suspicious URL:

    http://secure-login-example.com/verify-account

The application displays:

- Prediction
- Phishing probability
- Legitimate probability
- AI confidence
- Risk level
- Detection reasons

## Limitations

The system is trained on a specific phishing URL dataset and therefore cannot guarantee detection of every newly created phishing website.

The model analyzes URL characteristics and does not independently verify the content or reputation of a live website.

Reported test-set performance should not be interpreted as guaranteed real-world performance.

## Future Enhancements

Possible future improvements include:

- Real-time threat intelligence integration
- DNS and domain-age analysis
- SSL certificate analysis
- WHOIS-based features
- External reputation services
- Browser extension integration
- Email phishing detection
- QR-code phishing detection
- Continuous model retraining
- Explainable AI visualizations
- Cloud deployment
- Adversarial robustness testing

## Conclusion

The project demonstrates how machine learning and deep learning can be applied to phishing URL detection. The Character-Level CNN provides strong classification performance while allowing the application to analyze URLs without visiting the target websites.

The combination of URL validation, AI prediction, risk analysis, scan history, analytics, and application security provides a complete phishing detection prototype suitable for academic demonstration and further development.
