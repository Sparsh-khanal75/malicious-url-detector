"""Dataset download, cleaning and leakage-aware splitting."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .features import normalise, registered_domain

DATA_URL = (
    "https://raw.githubusercontent.com/faizann24/Using-machine-learning-to-detect-malicious-URLs/master/data/data.csv"
)
DATA_SHA256 = "3243f054f5205e352f7a9adc2d7b993d516acd4ad17cb032b91c989138e34808"
EXPECTED_ROWS = 420_464


def download(dest: str | Path = "data/urls.csv") -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading dataset -> {dest}")
        urllib.request.urlretrieve(DATA_URL, dest)
    digest = sha256(dest)
    if digest != DATA_SHA256:
        raise RuntimeError(f"Checksum mismatch for {dest}: {digest} (dataset changed upstream?)")
    return dest


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_clean(path: str | Path, sample: int | None = None, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Load, normalise, de-duplicate and drop URLs with conflicting labels."""
    raw = pd.read_csv(path)
    stats = {"raw_rows": len(raw)}
    df = raw.dropna(subset=["url", "label"]).copy()
    df["url"] = df["url"].astype(str).map(normalise)
    df = df[df["url"].str.len() > 0]
    df["y"] = (df["label"].str.strip().str.lower() == "bad").astype(int)
    conflicts = df.groupby("url")["y"].nunique()
    conflicting = set(conflicts[conflicts > 1].index)
    df = df[~df["url"].isin(conflicting)]
    before = len(df)
    df = df.drop_duplicates("url")
    stats.update(conflicting_urls_removed=len(conflicting), duplicates_removed=before - len(df))
    if sample and sample < len(df):
        df = df.groupby("y", group_keys=False).sample(frac=sample / len(df), random_state=seed)
    df["domain"] = df["url"].map(registered_domain)
    df = df[["url", "y", "domain"]].reset_index(drop=True)
    stats.update(
        clean_rows=len(df),
        malicious=int(df["y"].sum()),
        benign=int((df["y"] == 0).sum()),
        unique_domains=int(df["domain"].nunique()),
    )
    return df, stats


def split(df: pd.DataFrame, strategy: str = "domain", test_size: float = 0.2, seed: int = 42):
    """'random' = stratified row split; 'domain' = no registered domain in both train and test.

    A random split lets the model see e.g. 3,000 en.wikipedia.org URLs in training and
    1,000 more in test - it then scores well by memorising domains. The domain split
    measures what matters: performance on sites the model has never seen.
    """
    if strategy == "random":
        return train_test_split(df, test_size=test_size, stratify=df["y"], random_state=seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, df["y"], groups=df["domain"]))
    return df.iloc[tr], df.iloc[te]
