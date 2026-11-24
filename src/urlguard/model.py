"""Two-stage URL classifier: character n-gram text model stacked with lexical features."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from . import __version__
from .features import FEATURE_NAMES, LexicalFeatures, normalise


@dataclass
class Prediction:
    url: str
    normalised: str
    score: float
    malicious: bool
    top_ngrams: list[dict[str, float]]
    signals: dict[str, float]


class UrlGuardModel:
    """
    Stage 1 - TF-IDF over character 3-5-grams + logistic regression. Learns sub-word
              patterns ('paypa', '.php?', 'secur') without hand-written rules.
    Stage 2 - gradient boosting over 25 lexical features + the stage-1 score.
              Stage-1 scores for training come from out-of-fold predictions grouped by
              domain, so stage 2 never learns from leaked, over-confident scores.
    """

    def __init__(self, max_features: int = 200_000, C: float = 4.0, n_folds: int = 3, seed: int = 42):
        self.max_features, self.C, self.n_folds, self.seed = max_features, C, n_folds, seed
        self.threshold = 0.5
        self.version = __version__

    # ------------------------------------------------------------------ #
    def _text_model(self):
        vec = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=3, max_features=self.max_features, sublinear_tf=True
        )
        clf = LogisticRegression(C=self.C, max_iter=300, solver="liblinear")
        return vec, clf

    def fit(self, urls: Sequence[str], y: Sequence[int], groups: Sequence[str]):
        urls = [normalise(u) for u in urls]
        y = np.asarray(y)
        groups = np.asarray(groups)
        self.lex_ = LexicalFeatures()
        L = self.lex_.transform(urls)

        oof = np.zeros(len(urls))
        for tr, va in GroupKFold(n_splits=self.n_folds).split(urls, y, groups):
            vec, clf = self._text_model()
            Xtr = vec.fit_transform([urls[i] for i in tr])
            clf.fit(Xtr, y[tr])
            oof[va] = clf.predict_proba(vec.transform([urls[i] for i in va]))[:, 1]

        self.vec_, self.text_clf_ = self._text_model()
        self.text_clf_.fit(self.vec_.fit_transform(urls), y)
        self.meta_ = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, random_state=self.seed)
        self.meta_.fit(np.column_stack([L, oof]), y)
        self.vocab_ = np.array(self.vec_.get_feature_names_out())
        return self

    # ------------------------------------------------------------------ #
    def text_score(self, urls: Sequence[str]) -> np.ndarray:
        return self.text_clf_.predict_proba(self.vec_.transform([normalise(u) for u in urls]))[:, 1]

    def predict_proba(self, urls: Sequence[str]) -> np.ndarray:
        norm = [normalise(u) for u in urls]
        s1 = self.text_clf_.predict_proba(self.vec_.transform(norm))[:, 1]
        return self.meta_.predict_proba(np.column_stack([self.lex_.transform(norm), s1]))[:, 1]

    def predict(self, urls: Sequence[str]) -> np.ndarray:
        return (self.predict_proba(urls) >= self.threshold).astype(int)

    def explain(self, url: str, k: int = 5) -> Prediction:
        """Score + the character n-grams and lexical signals that pushed it toward 'malicious'."""
        norm = normalise(url)
        x = self.vec_.transform([norm])
        contrib = x.multiply(self.text_clf_.coef_[0]).tocsr()
        idx = contrib.indices[np.argsort(-contrib.data)][:k]
        top = [
            {"ngram": str(self.vocab_[i]), "weight": round(float(contrib[0, i]), 3)} for i in idx if contrib[0, i] > 0
        ]
        lex = dict(zip(FEATURE_NAMES, self.lex_.transform([norm])[0].tolist()))
        signals = {
            k2: v
            for k2, v in lex.items()
            if k2 in {"has_ip_host", "has_at", "risky_tld", "is_shortener", "has_port"} and v
        }
        if lex["suspicious_word_count"]:
            signals["suspicious_word_count"] = lex["suspicious_word_count"]
        score = float(self.predict_proba([url])[0])
        return Prediction(
            url=url,
            normalised=norm,
            score=round(score, 4),
            malicious=score >= self.threshold,
            top_ngrams=top,
            signals=signals,
        )
