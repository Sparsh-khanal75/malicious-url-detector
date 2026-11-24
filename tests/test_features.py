import pandas as pd

from urlguard.data import load_clean, split
from urlguard.features import FEATURE_NAMES, lexical_features, normalise, registered_domain


def test_normalise_removes_shortcut_tokens():
    assert normalise("HTTPS://WWW.Example.com/Path/") == "example.com/Path"
    assert normalise("http://example.com") == normalise("example.com")


def test_registered_domain():
    assert registered_domain("a.b.example.co.uk/x") == "example.co.uk"
    assert registered_domain("10.0.0.1:8080/admin") == "10.0.0.1"
    assert registered_domain("user@evil.com/x") == "evil.com"


def test_lexical_features_shape_and_signals():
    f = dict(zip(FEATURE_NAMES, lexical_features("192.168.1.1:8080/paypal/login.php?a=1&b=2")))
    assert len(FEATURE_NAMES) == 25
    assert f["has_ip_host"] == 1 and f["has_port"] == 1
    assert f["num_params"] == 2 and f["suspicious_word_count"] >= 3


def test_cleaning_and_domain_split(tmp_path):
    csv = tmp_path / "d.csv"
    pd.DataFrame(
        {
            "url": ["www.a.com/x", "a.com/x", "b.com/1", "b.com/2", "c.com", "c.com", "d.com/q", "e.com"],
            "label": ["good", "good", "bad", "bad", "good", "bad", "good", "bad"],
        }
    ).to_csv(csv, index=False)
    df, stats = load_clean(csv)
    assert stats["duplicates_removed"] == 1  # www.a.com/x == a.com/x after normalisation
    assert stats["conflicting_urls_removed"] == 1  # c.com labelled both ways
    tr, te = split(df, "domain", test_size=0.4)
    assert not set(tr["domain"]) & set(te["domain"])
