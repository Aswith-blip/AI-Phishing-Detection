
from urllib.parse import urlparse
import re
import ipaddress


MAX_URL_LENGTH = 2048


def validate_url(url):

    if not url:
        return False, "", "Please enter a URL."

    url = url.strip()

    if not url:
        return False, "", "Please enter a URL."

    if len(url) > MAX_URL_LENGTH:
        return False, "", "URL is too long. Maximum length is 2048 characters."

    if re.search(r"\s", url):
        return False, "", "URL must not contain spaces."

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "https://" + url

    try:
        parsed = urlparse(url)
    except Exception:
        return False, "", "Invalid URL format."

    if parsed.scheme.lower() not in ("http", "https"):
        return False, "", "Only HTTP and HTTPS URLs are supported."

    if not parsed.netloc:
        return False, "", "URL must contain a valid domain."

    hostname = parsed.hostname

    if not hostname:
        return False, "", "Invalid domain name."

    hostname = hostname.lower()

    if len(hostname) > 253:
        return False, "", "Domain name is too long."

    # Reject localhost
    if hostname in ("localhost", "localhost.localdomain"):
        return False, "", "Localhost URLs are not supported."

    # Reject IP addresses used for local/private networks
    try:
        ip = ipaddress.ip_address(hostname)

        if ip.is_loopback:
            return False, "", "Loopback IP addresses are not supported."

        if ip.is_private:
            return False, "", "Private IP addresses are not supported."

        if ip.is_link_local:
            return False, "", "Link-local IP addresses are not supported."

    except ValueError:
        # Hostname is a domain name, not an IP address.
        pass

    if hostname.startswith(".") or hostname.endswith("."):
        return False, "", "Invalid domain name."

    if ".." in hostname:
        return False, "", "Invalid domain name."

    if not re.match(r"^[a-zA-Z0-9.-]+$", hostname):
        return False, "", "Domain contains invalid characters."

    for label in hostname.split("."):

        if not label:
            return False, "", "Invalid domain name."

        if label.startswith("-") or label.endswith("-"):
            return False, "", "Invalid domain name."

    return True, url, ""


def normalize_url(url):

    valid, cleaned_url, error = validate_url(url)

    if not valid:
        return None

    return cleaned_url
