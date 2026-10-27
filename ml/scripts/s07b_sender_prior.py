"""s07b: set the base head's sender-feature weights from domain knowledge, validated on the modern dev split.

    uv run python -I scripts/s07b_sender_prior.py

Features 21-26 are zero for every corpus message, even one with a sender (sender information reaches the trained
weights only through the sender × content vector), so their weights aren't learned there. The signal is strong in
practice: US short codes are carrier-vetted and almost never send smishing (7 of ~19.5k NCSU messages), while
email-to-SMS gateways and international numbers are scam favorites. This step grid-searches those weights on
eval/modern_v1.jsonl's dev split only (the test split stays untouched for s09), maximizing
recall - 5·FPR at the Balanced threshold, preferring smaller weights on ties. It writes them into head_base.bin
(full and fallback heads); on-device learning never changes them.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import replace

import numpy as np

from smsml.decide import SPAM, Thresholds, decide
from smsml.features import META_NAMES
from smsml.head import Head
from smsml.pipeline import BUNDLE_WORK, featurize, load_modern_eval, senders_of, work, write_json

IDX = {n: META_NAMES.index(n) for n in META_NAMES}
OTP, URLS = IDX["otp_pattern"], IDX["url_count"]
FIRST_CONTACT_PRIOR = 0.5  # mild; the eval set can't validate it (each message is standalone)


def sender_meta(meta: np.ndarray, rows) -> np.ndarray:
    """Sender features as the device computes them; first contact is unknown here, so left at 0."""
    out = meta.copy()
    for i, (_, r) in enumerate(rows.iterrows()):
        kind, s = r["sender_type"], str(r["sender"])
        out[i, IDX["snd_short_code"]] = float(kind == "SHORT_CODE")
        out[i, IDX["snd_alphanumeric"]] = float(kind == "ALPHANUMERIC")
        out[i, IDX["snd_email"]] = float(kind == "EMAIL")
        out[i, IDX["snd_long_code_nanp"]] = float(kind == "PHONE" and s.startswith("+1"))
        out[i, IDX["snd_international"]] = float(kind == "PHONE" and s.startswith("+") and not s.startswith("+1"))
    return out


def evaluate(head: Head, emb, X, S, meta, y, thr: Thresholds) -> tuple[float, float]:
    z = head.bias + (emb.astype(np.float64) - head.mu_embed) @ head.w_embed.astype(np.float64) \
        + (meta.astype(np.float64) - head.mu_meta) @ head.w_meta.astype(np.float64) + X @ head.w_hash.astype(np.float64) \
        + S @ head.w_snd.astype(np.float64)
    p = 1 / (1 + np.exp(-z))
    flagged = np.array([decide(memory_match=None, p=float(pi),
                               otp_no_url=m[OTP] == 1.0 and m[URLS] == 0.0, thresholds=thr)[0] == SPAM
                        for pi, m in zip(p, meta)])
    return float(flagged[y == 1].mean()), float(flagged[y == 0].mean())


def with_sender(head: Head, w: dict[str, float]) -> Head:
    wm, wf = head.w_meta.copy(), head.w_meta_fb.copy()
    for name, v in w.items():
        wm[IDX[name]] = v
        wf[IDX[name]] = v
    return replace(head, w_meta=wm, w_meta_fb=wf)


def main() -> int:
    head = Head.load(BUNDLE_WORK / "head_base.bin")
    cal = json.loads(work("calibration.json").read_text())
    thr = Thresholds(**cal["thresholds"]["full"])
    modern = load_modern_eval()
    em = np.load(work("emb_modern.npy"))
    dev = (modern["split"] == "dev").to_numpy()
    X, S, meta = featurize(modern.loc[dev, "text"].tolist(), head.idf, senders_of(modern[dev]))
    meta = sender_meta(meta, modern[dev])
    y = modern.loc[dev, "label"].to_numpy()
    X, S = X.astype(np.float64), S.astype(np.float64)

    base_recall, base_fpr = evaluate(head, em[dev], X, S, meta, y, thr)
    best = None
    grid = {"snd_short_code": [0, -1, -2, -3, -4, -5], "snd_email": [0, 1, 2, 3],
            "snd_international": [0, 0.5, 1, 1.5], "snd_alphanumeric": [0, 0.5]}
    for values in itertools.product(*grid.values()):
        w = dict(zip(grid.keys(), values))
        r, f = evaluate(with_sender(head, w), em[dev], X, S, meta, y, thr)
        score = r - 5 * f
        size = sum(abs(v) for v in values)
        if best is None or score > best[0] + 1e-12 or (abs(score - best[0]) <= 1e-12 and size < best[1]):
            best = (score, size, w, r, f)
    _, _, w, r, f = best
    w = {**w, "snd_first_contact": FIRST_CONTACT_PRIOR, "snd_long_code_nanp": 0.0}
    tuned = with_sender(head, w)
    tuned.save(BUNDLE_WORK / "head_base.bin")
    out = {"weights": w, "dev_before": {"recall": base_recall, "fpr": base_fpr}, "dev_after": {"recall": r, "fpr": f}}
    write_json(work("sender_prior.json"), out)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
