import re
from urllib.parse import urlparse


SUSPICIOUS_KEYWORDS = [
    "login",
    "signin",
    "sign-in",
    "verify",
    "verification",
    "account",
    "secure",
    "security",
    "update",
    "confirm",
    "password",
    "bank",
    "paypal",
    "payment",
    "wallet",
    "recover",
    "authentication",
    "credential",
    "billing",
    "invoice",
    "unlock",
    "suspend",
    "alert",
]


SHORTENER_DOMAINS = [
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
]


def extract_features(url):

    if not isinstance(url, str):
        url = str(url)

    url = url.strip()

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)

    domain = parsed.netloc.split(":")[0].lower()
    path = parsed.path
    query = parsed.query
    fragment = parsed.fragment

    features = []

    # 1 URL length
    features.append(len(url))

    # 2 Domain length
    features.append(len(domain))

    # 3 Number of dots
    features.append(url.count("."))

    # 4 Number of hyphens
    features.append(url.count("-"))

    # 5 Number of digits
    features.append(sum(c.isdigit() for c in url))

    # 6 Number of letters
    features.append(sum(c.isalpha() for c in url))

    # 7 Special characters
    features.append(len(re.findall(r"[^a-zA-Z0-9]", url)))

    # 8 Subdomains
    parts = domain.split(".")
    features.append(max(len(parts) - 2, 0))

    # 9 HTTPS
    features.append(1 if parsed.scheme == "https" else 0)

    # 10 IP address
    ip_pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
    features.append(1 if re.match(ip_pattern, domain) else 0)

    # 11 @ symbol
    features.append(1 if "@" in url else 0)

    # 12 Question marks
    features.append(url.count("?"))

    # 13 Equals signs
    features.append(url.count("="))

    # 14 Ampersands
    features.append(url.count("&"))

    # 15 Percent encoding
    features.append(url.count("%"))

    # 16 Path length
    features.append(len(path))

    # 17 Query length
    features.append(len(query))

    # 18 Fragment
    features.append(1 if fragment else 0)

    # 19 Suspicious keywords
    url_lower = url.lower()

    keyword_count = sum(
        1 for keyword in SUSPICIOUS_KEYWORDS
        if keyword in url_lower
    )

    features.append(keyword_count)

    # 20 URL depth
    depth = len(
        [part for part in path.split("/") if part]
    )

    features.append(depth)

    # 21 Consecutive dots
    features.append(
        1 if ".." in url else 0
    )

    # 22 Consecutive hyphens
    features.append(
        1 if "--" in url else 0
    )

    # 23 Domain contains digits
    features.append(
        1 if any(c.isdigit() for c in domain)
        else 0
    )

    # 24 Domain hyphen count
    features.append(domain.count("-"))

    # 25 Path digit count
    features.append(
        sum(c.isdigit() for c in path)
    )

    # 26 Query parameter count
    if query:
        features.append(
            len(query.split("&"))
        )
    else:
        features.append(0)

    # 27 URL shortener
    features.append(
        1 if domain in SHORTENER_DOMAINS
        else 0
    )

    # 28 Longest numeric sequence
    numbers = re.findall(r"\d+", url)

    if numbers:
        longest_number = max(
            len(n) for n in numbers
        )
    else:
        longest_number = 0

    features.append(longest_number)

    # 29 Letter-to-digit ratio
    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)

    if digits > 0:
        ratio = letters / digits
    else:
        ratio = letters

    features.append(round(ratio, 4))

    # 30 Entropy-like character diversity
    unique_chars = len(set(url))

    features.append(unique_chars)

    return features