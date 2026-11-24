# Model Card: urlguard 1.0.0

## Overview
Binary classifier that scores a URL string from 0 (benign) to 1 (malicious). It is a two-stage stack: a character 3-5-gram TF-IDF + logistic regression, whose score feeds a histogram gradient boosting model together with 25 hand-crafted lexical features. Decision threshold: **0.464** (F1-optimal on validation).

## Intended use
- A fast first-pass filter for URLs in email, chat or proxy logs, to prioritise analyst review or trigger deeper checks (sandboxing, reputation lookup).
- **Not** a sole blocking control. It sees only the URL string, never the page.

## Training data
- Public dataset: 420,464 labelled URLs (good/bad) from [faizann24/Using-machine-learning-to-detect-malicious-URLs](https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs). SHA-256 `3243f054…8e34808`.
- Cleaning: normalisation (scheme, `www.`, trailing `/` removed), 13,580 duplicates removed, 2 URLs with conflicting labels removed → **406,880 URLs (62,082 malicious, 15.3%) from 115,321 registered domains**.
- Split by registered domain: train 298,232 / validation 38,429 / test 70,219. No domain appears in more than one split.

## Metrics (test set, unseen domains)
| Metric | Value |
|---|---|
| PR-AUC | 0.978 |
| ROC-AUC | 0.993 |
| F1 @ 0.464 | 0.934 |
| Precision / recall | 0.957 / 0.912 |
| False positive rate | 0.91% (523 of 57,384 benign) |
| Recall at 1% FPR | 0.916 |

Full numbers are in [`reports/metrics.json`](reports/metrics.json).

## Known limitations and risks
- **Dataset age and source bias.** The benign URLs skew toward well-known sites, and the malicious ones toward older phishing and malware campaigns. Expect lower performance on modern phishing kits and on non-English sites.
- **Unstable FPR estimates.** A few large benign domains dominate the false positives, so FPR measured on one sample of domains can differ by 1.5–2× on another. Monitor the live score distribution.
- **Evasion.** An attacker who knows the features can craft URLs that look benign (short, `.com`, no keywords). It should be paired with content-based and reputation-based detection.
- **Compromised legitimate sites** host malicious content on clean domains, and the model will usually score them as benign.

## Ethical considerations
False positives can block legitimate small businesses whose URLs look unusual (long, numeric, non-English). Use the score to rank items for review, not to block silently.

## Retraining
`python -m urlguard.train` (about 5 minutes on 2 CPU cores). CI blocks any change that drops sample PR-AUC below 0.90.
