import numpy as np

from urlguard.train import threshold_for_fpr


def test_model_separates_toy_data(toy_model, toy_data):
    urls, y, _ = toy_data
    p = toy_model.predict_proba(urls)
    assert np.mean((p >= 0.5) == np.array(y)) > 0.95


def test_explanation_points_at_malicious_ngrams(toy_model):
    pred = toy_model.explain("paypal.com.secure-999.tk/webscr?cmd=_login")
    assert pred.malicious
    assert pred.top_ngrams and all(g["weight"] > 0 for g in pred.top_ngrams)
    assert pred.signals.get("risky_tld") == 1


def test_threshold_respects_target_fpr():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(200)]
    p = np.r_[rng.uniform(0, 0.7, 1000), rng.uniform(0.3, 1, 200)]
    thr = threshold_for_fpr(y, p, 0.01)
    assert np.mean(p[y == 0] >= thr) <= 0.01
