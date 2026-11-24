"""
Train, evaluate and package the URL classifier.

  python -m urlguard.train                       # full dataset (~6 min on 2 CPU cores)
  python -m urlguard.train --sample 40000        # quick run used by CI
  python -m urlguard.train --min-pr-auc 0.95     # fail (exit 1) if the model is worse -> CI quality gate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .data import download, load_clean, split
from .model import UrlGuardModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def metrics(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "threshold": thr,
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "false_positive_rate": fp / (fp + tn),
        "recall_at_1pct_fpr": float(np.interp(0.01, fpr, tpr)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_for_fpr(y, p, target_fpr: float) -> float:
    """Lowest threshold whose false-positive rate on validation data stays <= target."""
    neg = np.sort(p[np.asarray(y) == 0])
    k = int(np.floor(len(neg) * (1 - target_fpr)))
    return float(neg[min(k, len(neg) - 1)]) + 1e-9


def threshold_max_f1(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[int(np.argmax(f1[:-1]))])


def text_only_baseline(tr, te):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, max_features=200_000, sublinear_tf=True)
    clf = LogisticRegression(C=4.0, max_iter=300, solver="liblinear").fit(vec.fit_transform(tr["url"]), tr["y"])
    return clf.predict_proba(vec.transform(te["url"]))[:, 1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/urls.csv")
    ap.add_argument("--sample", type=int, default=None, help="train on a stratified sample of N rows")
    ap.add_argument(
        "--threshold-policy",
        choices=["f1", "fpr"],
        default="f1",
        help="f1 = maximise F1 on validation; fpr = highest recall with FPR <= --target-fpr",
    )
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--out", default="models/urlguard.joblib")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--min-pr-auc", type=float, default=None, help="quality gate for CI")
    ap.add_argument("--skip-random-split", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    rep = Path(args.reports)
    rep.mkdir(parents=True, exist_ok=True)

    df, stats = load_clean(download(args.data), sample=args.sample)
    print("Data:", stats)

    # Leakage check: how optimistic is a random split?
    leakage = {}
    if not args.skip_random_split:
        for strategy in ("random", "domain"):
            tr, te = split(df, strategy)
            p = text_only_baseline(tr, te)
            leakage[strategy] = {
                "test_rows_with_domain_seen_in_training": float(te["domain"].isin(set(tr["domain"])).mean()),
                "pr_auc": average_precision_score(te["y"], p),
                "f1@0.5": f1_score(te["y"], p >= 0.5),
            }
            print(
                f"  text-only LR, {strategy:<6} split: PR-AUC={leakage[strategy]['pr_auc']:.4f}  "
                f"F1={leakage[strategy]['f1@0.5']:.4f}"
            )

    # Final model: domain-grouped train / validation / test
    train_full, test = split(df, "domain", test_size=0.2)
    train, val = split(train_full, "domain", test_size=0.1, seed=7)
    model = UrlGuardModel().fit(train["url"].tolist(), train["y"].values, train["domain"].values)
    p_val = model.predict_proba(val["url"].tolist())
    if args.threshold_policy == "fpr":
        model.threshold = threshold_for_fpr(val["y"].values, p_val, args.target_fpr)
    else:
        model.threshold = threshold_max_f1(val["y"].values, p_val)
    val_fpr = float(np.mean(p_val[val["y"].values == 0] >= model.threshold))

    p_test = model.predict_proba(test["url"].tolist())
    p_text = model.text_score(test["url"].tolist())
    results = {
        "stage1_text_model@0.5": metrics(test["y"].values, p_text, 0.5),
        "stacked_model@0.5": metrics(test["y"].values, p_test, 0.5),
        "stacked_model@tuned": metrics(test["y"].values, p_test, model.threshold),
    }
    for k, v in results.items():
        print(
            f"  {k:<24} PR-AUC={v['pr_auc']:.4f} F1={v['f1']:.4f} precision={v['precision']:.4f} "
            f"recall={v['recall']:.4f} FPR={v['false_positive_rate']:.4f}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out, compress=3)
    summary = {
        "model_version": model.version,
        "data": stats,
        "split": "domain-grouped (80/20, val 10% of train)",
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "threshold_policy": args.threshold_policy,
        "threshold": model.threshold,
        "validation_fpr_at_threshold": val_fpr,
        "leakage_check": leakage,
        "test_results": results,
        "train_seconds": round(time.time() - t0, 1),
        "model_size_mb": round(Path(args.out).stat().st_size / 1e6, 1),
    }
    (rep / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    plot_reports(test, p_test, p_text, model, rep)
    print(f"Saved {args.out} ({summary['model_size_mb']} MB) and reports in {rep}/ in {summary['train_seconds']}s")

    if args.min_pr_auc is not None:
        got = results["stacked_model@tuned"]["pr_auc"]
        if got < args.min_pr_auc:
            print(f"QUALITY GATE FAILED: PR-AUC {got:.4f} < {args.min_pr_auc}")
            return 1
        print(f"Quality gate passed: PR-AUC {got:.4f} >= {args.min_pr_auc}")
    return 0


def plot_reports(test, p_test, p_text, model, rep: Path):
    y = test["y"].values
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for p, name, col in [(p_text, "Stage 1: char n-gram LR", "#c05621"), (p_test, "Stacked model", "#2b6cb0")]:
        pr, rc, _ = precision_recall_curve(y, p)
        axes[0].plot(rc, pr, label=f"{name} (AP={average_precision_score(y, p):.3f})", color=col)
    axes[0].set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-recall, unseen-domain test set",
        xlim=(0, 1),
        ylim=(0.5, 1.01),
    )
    axes[0].legend(loc="lower left")
    bins = np.linspace(0, 1, 41)
    axes[1].hist(p_test[y == 0], bins=bins, alpha=0.7, label="benign", color="#2f855a", log=True)
    axes[1].hist(p_test[y == 1], bins=bins, alpha=0.7, label="malicious", color="#c53030", log=True)
    axes[1].axvline(model.threshold, color="black", ls="--", label=f"threshold {model.threshold:.2f}")
    axes[1].set(xlabel="Model score", ylabel="URLs (log scale)", title="Score distribution")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(rep / "evaluation.png", dpi=140)
    plt.close(fig)

    coef = model.text_clf_.coef_[0]
    top_bad, top_good = np.argsort(-coef)[:15], np.argsort(coef)[:15]
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [repr(model.vocab_[i]) for i in top_good[::-1]] + [repr(model.vocab_[i]) for i in top_bad[::-1]]
    vals = list(coef[top_good[::-1]]) + list(coef[top_bad[::-1]])
    ax.barh(labels, vals, color=["#2f855a"] * 15 + ["#c53030"] * 15)
    ax.set(title="Most influential character n-grams (stage 1)", xlabel="Logistic-regression weight")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(rep / "top_ngrams.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
