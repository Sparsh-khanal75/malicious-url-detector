"""URL normalisation and hand-crafted lexical features."""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from collections.abc import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)
MULTI_SUFFIX = {
    "co.uk",
    "com.au",
    "co.in",
    "co.jp",
    "com.br",
    "co.nz",
    "org.uk",
    "ac.uk",
    "gov.uk",
    "com.cn",
    "com.tw",
    "co.za",
    "com.mx",
    "com.np",
    "edu.np",
    "gov.np",
}
SUSPICIOUS_WORDS = [
    "login",
    "signin",
    "verify",
    "account",
    "update",
    "secure",
    "bank",
    "confirm",
    "password",
    "webscr",
    "ebayisapi",
    "paypal",
    "wp-admin",
    "wp-content",
    "admin",
    "free",
    "bonus",
    "lucky",
    ".exe",
    ".zip",
    ".php",
    "cmd=",
    "dropbox",
    "invoice",
]
RISKY_TLDS = {
    "tk",
    "ml",
    "ga",
    "cf",
    "gq",
    "xyz",
    "top",
    "info",
    "ru",
    "cn",
    "biz",
    "club",
    "online",
    "site",
    "work",
    "click",
    "link",
    "pw",
    "su",
    "cc",
}
SHORTENERS = {"bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd", "buff.ly", "adf.ly", "bitly.com"}


def normalise(url: str) -> str:
    """Canonical form used for training AND serving.

    Strips scheme, 'www.' and trailing slashes and lower-cases the host. In the raw
    dataset 'www.' appears in 3% of malicious URLs and 0% of benign ones - a label
    shortcut a model would happily learn and that would not hold in production.
    """
    u = str(url).strip()
    u = SCHEME_RE.sub("", u)
    host, sep, rest = u.partition("/")
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return (host + sep + rest).rstrip("/")


def host_of(url: str) -> str:
    host = normalise(url).split("/", 1)[0]
    host = host.split("@")[-1]  # user:pass@host
    return host.split(":")[0]  # strip port


def registered_domain(url: str) -> str:
    host = host_of(url)
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in MULTI_SUFFIX:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


FEATURE_NAMES = [
    "url_len",
    "host_len",
    "path_len",
    "query_len",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "digit_ratio",
    "num_special",
    "num_subdomains",
    "path_depth",
    "num_params",
    "has_ip_host",
    "has_at",
    "has_port",
    "has_double_slash_in_path",
    "host_entropy",
    "url_entropy",
    "suspicious_word_count",
    "risky_tld",
    "is_shortener",
    "longest_token",
    "num_upper",
    "hex_encoded_chars",
    "tld_len",
]


def lexical_features(url: str) -> list[float]:
    n = normalise(url)
    host = host_of(url)
    path = n.split("/", 1)[1] if "/" in n else ""
    query = path.split("?", 1)[1] if "?" in path else ""
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0
    tokens = re.split(r"[/?.=&_\-]", n)
    low = n.lower()
    digits = sum(ch.isdigit() for ch in n)
    return [
        len(n),
        len(host),
        len(path),
        len(query),
        n.count("."),
        n.count("-"),
        digits,
        digits / max(len(n), 1),
        sum(not ch.isalnum() for ch in n),
        max(host.count(".") - 1, 0),
        path.count("/"),
        query.count("&") + (1 if query else 0),
        is_ip,
        int("@" in n),
        int(":" in n.split("/", 1)[0]),
        int("//" in path),
        shannon_entropy(host),
        shannon_entropy(n),
        sum(w in low for w in SUSPICIOUS_WORDS),
        int(tld in RISKY_TLDS),
        int(registered_domain(url) in SHORTENERS),
        max((len(t) for t in tokens), default=0),
        sum(ch.isupper() for ch in n),
        n.count("%"),
        len(tld),
    ]


class LexicalFeatures(BaseEstimator, TransformerMixin):
    """scikit-learn transformer: list of URLs -> (n, 25) float matrix."""

    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        return np.array([lexical_features(u) for u in X], dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)


def explain_lexical(url: str) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, lexical_features(url)))
