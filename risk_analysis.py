# =========================================================
# RISK ANALYSIS AND EXPLANATION
# AI-POWERED PHISHING WEBSITE DETECTION SYSTEM
# =========================================================


def calculate_risk(features, phishing_probability):
    """
    Calculate an application-level risk score.

    phishing_probability is expected as a percentage
    between 0 and 100.
    """

    score = float(phishing_probability)

    # HTTPS
    if features[8] == 0:
        score += 8

    # IP address
    if features[9] == 1:
        score += 12

    # @ symbol
    if features[10] == 1:
        score += 8

    # Multiple hyphens
    if features[3] >= 3:
        score += 3

    # Multiple subdomains
    if features[7] >= 2:
        score += 4

    # Suspicious keywords
    if features[18] >= 1:
        score += min(features[18] * 3, 12)

    # Very long URL
    if features[0] > 100:
        score += 4

    # Percent encoding
    if features[14] >= 2:
        score += 3

    # Consecutive dots
    if features[20] == 1:
        score += 3

    # Consecutive hyphens
    if features[21] == 1:
        score += 3

    # Domain contains digits
    if features[22] == 1:
        score += 2

    # URL shortener
    if features[26] == 1:
        score += 5

    # HTTPS reduces risk
    if features[8] == 1:
        score -= 3

    score = max(
        0,
        min(100, score)
    )

    if score >= 75:
        risk_level = "HIGH RISK"

    elif score >= 40:
        risk_level = "MEDIUM RISK"

    else:
        risk_level = "LOW RISK"

    return round(score, 2), risk_level


def get_detection_reasons(features):
    """
    Generate human-readable explanations
    from URL structural features.
    """

    reasons = []

    # HTTPS
    if features[8] == 0:
        reasons.append(
            "Website does not use HTTPS"
        )

    else:
        reasons.append(
            "Website uses HTTPS"
        )

    # IP address
    if features[9] == 1:
        reasons.append(
            "URL uses an IP address instead of a normal domain"
        )

    # @ symbol
    if features[10] == 1:
        reasons.append(
            "URL contains an @ symbol"
        )

    # Multiple hyphens
    if features[3] >= 3:
        reasons.append(
            "URL contains multiple hyphens"
        )

    # Multiple subdomains
    if features[7] >= 2:
        reasons.append(
            "URL contains multiple subdomains"
        )

    # Suspicious keywords
    if features[18] >= 1:
        reasons.append(
            "Suspicious security-related keywords detected"
        )

    # Long URL
    if features[0] > 100:
        reasons.append(
            "URL is unusually long"
        )

    # Percent encoding
    if features[14] >= 2:
        reasons.append(
            "URL contains multiple encoded characters"
        )

    # Consecutive dots
    if features[20] == 1:
        reasons.append(
            "URL contains consecutive dots"
        )

    # Consecutive hyphens
    if features[21] == 1:
        reasons.append(
            "URL contains consecutive hyphens"
        )

    # Domain digits
    if features[22] == 1:
        reasons.append(
            "Domain contains numeric characters"
        )

    # URL shortener
    if features[26] == 1:
        reasons.append(
            "URL uses a URL shortening service"
        )

    # If no suspicious indicators
    if len(reasons) == 1 and features[8] == 1:

        reasons.append(
            "No major structural phishing indicators detected"
        )

    return reasons