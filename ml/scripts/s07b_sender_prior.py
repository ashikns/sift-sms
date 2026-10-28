"""s07b: set the base head's sender-feature weights, validated on held-out messages scored the way the phone does.

    uv run python -I scripts/s07b_sender_prior.py

Features 21-30 are zero for every corpus message, even one with a sender (sender information reaches the trained
weights only through the sender × content vector), so their weights aren't learned there. Two grid searches set them,
each with the sender features filled in as the phone computes them (pipeline.sender_meta), maximizing
recall - 5·FPR at the Balanced threshold and preferring smaller weights on ties:
- Sender type, on eval/modern_v1.jsonl's dev split (its test split stays untouched for s09). The signal is strong in
  practice: US short codes are carrier-vetted and almost never send smishing (7 of ~19.5k NCSU messages), while
  email-to-SMS gateways and international numbers are scam favorites.
- The Indian DLT category suffix (-P promotional, -S service, -T transactional, -G government), on the inbox's older
  half with s06's out-of-fold logits, so each message is scored by a model that didn't learn its template (the newest
  half, inbox_eval, stays untouched for s09). Many senders don't follow the categories (one bank sends card alerts and
  its own upsells alike under -S, and some legitimate notices arrive under -P), so the suffix only contributes: each
  weight keeps the sign its category stands for and stays within 2 on the log-odds scale. It can tip a borderline
  message but not overrule what a message says, and the content model never sees it.
It writes them into head_base.bin (full and fallback heads); on-device learning never changes them.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from smsml.decide import SPAM, Thresholds, decide
from smsml.features import DLT_CATEGORIES, SENDER_SLICE, dlt_category
from smsml.head import Head
from smsml.pipeline import (BUNDLE_WORK, META_INDEX, featurize, load_modern_eval, sender_key, sender_meta, senders_of,
                            work, write_json)

OTP, URLS = META_INDEX["otp_pattern"], META_INDEX["url_count"]
FIRST_CONTACT_PRIOR = 0.5  # mild; the eval set can't validate it (each message is standalone)
KIND_GRID = {"snd_short_code": [0, -1, -2, -3, -4, -5], "snd_email": [0, 1, 2, 3],
             "snd_international": [0, 0.5, 1, 1.5], "snd_alphanumeric": [0, 0.5]}
# The suffix's bounds: promotional only counts toward spam, service and transactional only toward legitimate, and
# government either way (TRAI awareness broadcasts are spam, weather and disaster alerts aren't). Service stays small
# because the user's bank sends its upsells under -S.
DLT_GRID = {"dlt_promotional": [0, 0.5, 1, 1.5, 2], "dlt_service": [0, -0.5, -1],
            "dlt_transactional": [0, -0.5, -1, -1.5, -2], "dlt_government": [-1, -0.5, 0, 0.5, 1]}


def flagged(z: np.ndarray, meta: np.ndarray, thr: Thresholds) -> np.ndarray:
    p = 1 / (1 + np.exp(-z))
    return np.array([decide(memory_match=None, p=float(pi),
                            otp_no_url=m[OTP] == 1.0 and m[URLS] == 0.0, thresholds=thr)[0] == SPAM
                     for pi, m in zip(p, meta)])


def full_z(head: Head, emb, X, S, meta) -> np.ndarray:
    return head.bias + (emb.astype(np.float64) - head.mu_embed) @ head.w_embed.astype(np.float64) \
        + (meta.astype(np.float64) - head.mu_meta) @ head.w_meta.astype(np.float64) + X @ head.w_hash.astype(np.float64) \
        + S @ head.w_snd.astype(np.float64)


def sender_z(head: Head, z_content: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """Content logits (sender features excluded, as s06's out-of-fold ones) plus the head's sender-feature terms."""
    m = meta[:, SENDER_SLICE].astype(np.float64) - head.mu_meta[SENDER_SLICE]
    return z_content + m @ head.w_meta[SENDER_SLICE].astype(np.float64)


def recall_fpr(f: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    return float(f[y == 1].mean()), float(f[y == 0].mean())


def with_sender(head: Head, w: dict[str, float]) -> Head:
    wm, wf = head.w_meta.copy(), head.w_meta_fb.copy()
    for name, v in w.items():
        wm[META_INDEX[name]] = v
        wf[META_INDEX[name]] = v
    return replace(head, w_meta=wm, w_meta_fb=wf)


def search(head: Head, grid: dict[str, list[float]], z_of, meta, y, thr: Thresholds):
    """The grid point with the best recall - 5·FPR, the smallest Σ|w| among ties: (weights, recall, fpr). `z_of` maps a
    head to the logits."""
    best = None
    for values in itertools.product(*grid.values()):
        w = dict(zip(grid.keys(), values))
        r, f = recall_fpr(flagged(z_of(with_sender(head, w)), meta, thr), y)
        score = r - 5 * f
        size = sum(abs(v) for v in values)
        if best is None or score > best[0] + 1e-12 or (abs(score - best[0]) <= 1e-12 and size < best[1]):
            best = (score, size, w, r, f)
    return best[2], best[3], best[4]


def by_category(without: np.ndarray, with_: np.ndarray, cats: list[str | None], y: np.ndarray) -> dict:
    """Per DLT category (and "none"): messages flagged without and with the suffix weights."""
    out = {}
    for c in [*DLT_CATEGORIES, None]:
        m = np.array([k == c for k in cats], dtype=bool)
        if not m.any():
            continue
        ham, spam = m & (y == 0), m & (y == 1)
        out[c or "none"] = {"n_ham": int(ham.sum()), "n_spam": int(spam.sum()),
                            "flagged_ham_without": int(without[ham].sum()), "flagged_ham_with": int(with_[ham].sum()),
                            "caught_spam_without": int(without[spam].sum()), "caught_spam_with": int(with_[spam].sum())}
    return out


def main() -> int:
    head = Head.load(BUNDLE_WORK / "head_base.bin")
    cal = json.loads(work("calibration.json").read_text())
    thr = Thresholds(**cal["thresholds"]["full"])

    # Sender type, on the modern dev split.
    modern = load_modern_eval()
    em = np.load(work("emb_modern.npy"))
    dev = (modern["split"] == "dev").to_numpy()
    X, S, meta = featurize(modern.loc[dev, "text"].tolist(), head.idf, senders_of(modern[dev]))
    meta = sender_meta(meta, modern[dev])
    y = modern.loc[dev, "label"].to_numpy()
    X, S = X.astype(np.float64), S.astype(np.float64)
    kind_before = recall_fpr(flagged(full_z(head, em[dev], X, S, meta), meta, thr), y)
    w, r, f = search(head, KIND_GRID, lambda h: full_z(h, em[dev], X, S, meta), meta, y, thr)
    kind = {**w, "snd_first_contact": FIRST_CONTACT_PRIOR, "snd_long_code_nanp": 0.0}
    head = with_sender(head, kind)

    # DLT category, on the inbox's older half scored out of fold (with the sender-type weights just chosen).
    df = pd.read_parquet(work("corpus_split.parquet"))
    sub = df.iloc[np.load(work("emb_corpus_rows.npy"))].reset_index(drop=True)
    oof = np.load(work("inbox_oof.npz"))
    rows, z_content = oof["rows"], oof["z"]
    older = sub.iloc[rows]
    mi = sender_meta(np.load(work("features_meta.npy"))[rows], older)
    yi = older["label"].to_numpy()
    cats = [dlt_category(sender_key(s)) for s in senders_of(older)]
    before = flagged(sender_z(head, z_content, mi), mi, thr)
    dlt, d_recall, d_fpr = search(head, DLT_GRID, lambda h: sender_z(h, z_content, mi), mi, yi, thr)
    head = with_sender(head, dlt)
    after = flagged(sender_z(head, z_content, mi), mi, thr)

    head.save(BUNDLE_WORK / "head_base.bin")
    out = {"weights": {**kind, **dlt},
           "kind": {"dev_before": {"recall": kind_before[0], "fpr": kind_before[1]},
                    "dev_after": {"recall": r, "fpr": f}},
           "dlt": {"inbox_older_half_out_of_fold": {"n": len(rows), "without": dict(zip(("recall", "fpr"), recall_fpr(before, yi))),
                                                    "with": {"recall": d_recall, "fpr": d_fpr}},
                   "by_category": by_category(before, after, cats, yi)}}
    write_json(work("sender_prior.json"), out)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
