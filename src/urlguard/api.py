"""
REST API for the URL classifier.

  uvicorn urlguard.api:app --host 0.0.0.0 --port 8000
  open http://localhost:8000/ for the web UI, /docs for the API reference
  MODEL_PATH=models/urlguard.joblib (default, read at startup)
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

MAX_BATCH = 500
UI = Path(__file__).parent / "static" / "index.html"

PREDICTIONS = Counter("urlguard_predictions_total", "URLs scored", ["verdict"])
LATENCY = Histogram(
    "urlguard_request_seconds",
    "Request latency",
    ["endpoint"],
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
SCORES = Histogram(
    "urlguard_score", "Distribution of model scores (watch for drift)", buckets=[i / 10 for i in range(1, 11)]
)

state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only load model files you built yourself: joblib/pickle can execute code on load.
    state["model"] = joblib.load(os.getenv("MODEL_PATH", "models/urlguard.joblib"))
    yield
    state.clear()


app = FastAPI(
    title="urlguard",
    version="1.0.0",
    lifespan=lifespan,
    description="Malicious URL detection - char n-gram + lexical stacked model",
)


class UrlIn(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048, examples=["paypal.com.secure-login.tk/webscr"])


class BatchIn(BaseModel):
    urls: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)


class NGram(BaseModel):
    ngram: str
    weight: float


class Verdict(BaseModel):
    url: str
    normalised: str
    score: float
    verdict: str
    threshold: float
    top_ngrams: list[NGram]
    signals: dict[str, float]


def _model():
    m = state.get("model")
    if m is None:
        raise HTTPException(503, "model not loaded")
    return m


@app.get("/", include_in_schema=False)
def ui():
    """Single-page web UI for trying the model."""
    return FileResponse(UI)


@app.get("/health")
def health():
    m = state.get("model")
    return {
        "status": "ok" if m else "loading",
        "model_version": getattr(m, "version", None),
        "threshold": getattr(m, "threshold", None),
    }


@app.post("/predict", response_model=Verdict)
def predict(item: UrlIn):
    with LATENCY.labels("predict").time():
        m = _model()
        p = m.explain(item.url)
        verdict = "malicious" if p.malicious else "benign"
        PREDICTIONS.labels(verdict).inc()
        SCORES.observe(p.score)
        return Verdict(
            url=p.url,
            normalised=p.normalised,
            score=p.score,
            verdict=verdict,
            threshold=round(m.threshold, 4),
            top_ngrams=p.top_ngrams,
            signals=p.signals,
        )


@app.post("/predict/batch")
def predict_batch(batch: BatchIn):
    if any(len(u) > 2048 or not u.strip() for u in batch.urls):
        raise HTTPException(422, "each URL must be 1-2048 characters")
    with LATENCY.labels("batch").time():
        m = _model()
        t = time.perf_counter()
        scores = m.predict_proba(batch.urls)
        out = []
        for u, s in zip(batch.urls, scores):
            verdict = "malicious" if s >= m.threshold else "benign"
            PREDICTIONS.labels(verdict).inc()
            SCORES.observe(float(s))
            out.append({"url": u, "score": round(float(s), 4), "verdict": verdict})
        return {"count": len(out), "latency_ms": round((time.perf_counter() - t) * 1000, 1), "results": out}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
