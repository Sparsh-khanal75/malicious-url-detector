import pytest
from fastapi.testclient import TestClient

from urlguard.api import app


@pytest.fixture(scope="module")
def client(model_path):
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_PATH", str(model_path))
    with TestClient(app) as c:
        yield c
    mp.undo()


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_version"]


def test_predict_malicious_and_benign(client):
    bad = client.post("/predict", json={"url": "http://www.paypal.com.secure-42.tk/webscr?cmd=_login"}).json()
    good = client.post("/predict", json={"url": "https://github.com/page_123"}).json()
    assert bad["verdict"] == "malicious" and bad["normalised"].startswith("paypal.com")
    assert good["verdict"] == "benign"
    assert 0 <= good["score"] <= 1


def test_validation(client):
    assert client.post("/predict", json={"url": ""}).status_code == 422
    assert client.post("/predict", json={"url": "a" * 3000}).status_code == 422
    assert client.post("/predict/batch", json={"urls": []}).status_code == 422


def test_batch_and_metrics(client):
    r = client.post("/predict/batch", json={"urls": ["github.com/x", "free-bonus-7.ml/claim.exe"]}).json()
    assert r["count"] == 2 and [x["verdict"] for x in r["results"]] == ["benign", "malicious"]
    m = client.get("/metrics").text
    assert "urlguard_predictions_total" in m and "urlguard_request_seconds" in m


def test_web_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "urlguard" in r.text and "/predict" in r.text
