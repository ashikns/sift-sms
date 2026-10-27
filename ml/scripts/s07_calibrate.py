"""s07: calibrate decision thresholds by ham false-positive rate, and check near-duplicate cutoffs.

    uv run python -I scripts/s07_calibrate.py

Thresholds are set on val_a + val_b hams so that at most floor(target · n_ham) hams score at or above them
(FPR is independent of the spam/ham mix). Separate sets for the full head and the lexical-only fallback.
The near-duplicate check measures, for validation messages and their nearest training neighbour by n-gram
cosine, how often the labels agree above each cutoff, and picks the ham-match cutoff (s08 tunes the spam one).
Writes data/work/calibration.json.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

from smsml.head import Head
from smsml.pipeline import BUNDLE_WORK, config, work, write_json


def scores(head: Head, emb, X, meta, fallback: bool) -> np.ndarray:
    hash_part = X @ head.w_hash.astype(np.float64)
    meta_c = meta.astype(np.float64) - head.mu_meta.astype(np.float64)
    if fallback:
        z = head.fb_bias + head.fb_scale * hash_part + meta_c @ head.w_meta_fb.astype(np.float64)
    else:
        z = head.bias + (emb.astype(np.float64) - head.mu_embed.astype(np.float64)) @ head.w_embed.astype(np.float64) \
            + meta_c @ head.w_meta.astype(np.float64) + hash_part
    return 1.0 / (1.0 + np.exp(-z))


def threshold_at(ham_scores: np.ndarray, target: float) -> float:
    hs = np.sort(ham_scores)[::-1]
    k = int(np.floor(target * len(hs)))
    if k >= len(hs):
        return 0.0
    return float(np.nextafter(hs[k], 1.0))


def row_normalize(X: sp.csr_matrix) -> sp.csr_matrix:
    n = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    n[n == 0] = 1.0
    return sp.diags(1.0 / n) @ X


def main() -> int:
    cfg = config()
    head = Head.load(BUNDLE_WORK / "head_base.bin")
    df = pd.read_parquet(work("corpus_split.parquet"))
    rows = np.load(work("emb_corpus_rows.npy"))
    emb = np.load(work("emb_corpus.npy"))
    X = sp.load_npz(work("features_X.npz")).astype(np.float64)
    meta = np.load(work("features_meta.npy"))
    sub = df.iloc[rows].reset_index(drop=True)
    split, y = sub["split"].to_numpy(), sub["label"].to_numpy()
    cal = np.isin(split, ["val_a", "val_b"])

    out: dict = {"thresholds": {}, "recall": {}, "n_ham": int((y[cal] == 0).sum()), "n_spam": int((y[cal] == 1).sum())}
    targets = cfg["calibrate"]["fpr"]
    for kind, fb in (("full", False), ("fallback", True)):
        p = scores(head, emb[cal], X[cal], meta[cal], fb)
        ham, spam = p[y[cal] == 0], p[y[cal] == 1]
        t = {name: threshold_at(ham, float(fpr)) for name, fpr in targets.items()}
        out["thresholds"][kind] = t
        out["recall"][kind] = {name: float((spam >= v).mean()) for name, v in t.items()}
        out["recall"][kind]["fpr_check"] = {name: float((ham >= v).mean()) for name, v in t.items()}
    print(out)

    # Near-duplicate cutoffs: nearest training neighbour by n-gram cosine.
    rng = np.random.default_rng(int(cfg["seed"]))
    tr = np.flatnonzero(split == "train")
    q = rng.choice(np.flatnonzero(cal), size=min(3000, int(cal.sum())), replace=False)
    Xn = row_normalize(X)
    Xt = Xn[tr].T.tocsc()
    best_cos, best_lab = np.zeros(len(q)), np.zeros(len(q), dtype=int)
    for s in range(0, len(q), 250):
        sims = (Xn[q[s:s + 250]] @ Xt).toarray()
        j = sims.argmax(axis=1)
        best_cos[s:s + 250] = sims[np.arange(len(j)), j]
        best_lab[s:s + 250] = y[tr][j]
    agree = best_lab == y[q]
    table = []
    for tau in np.arange(0.60, 0.99, 0.05):
        for lab in (0, 1):
            m = (best_cos >= tau) & (best_lab == lab)
            table.append({"tau": round(float(tau), 2), "neighbor_label": lab, "n": int(m.sum()),
                          "agreement": float(agree[m].mean()) if m.any() else None})

    def pick(lab: int, need: float, default: float) -> float:
        for row in table:
            if row["neighbor_label"] == lab and row["n"] >= 30 and row["agreement"] is not None \
                    and row["agreement"] >= need and 0.75 <= row["tau"] <= 0.95:
                return row["tau"]
        return default

    out["knn"] = {
        "tau_ham": max(pick(0, 0.999, cfg["knn"]["tau_ham"]), cfg["knn"]["tau_ham"]),
        "table": table,
    }
    write_json(work("calibration.json"), out)
    print({k: v for k, v in out["knn"].items() if k != "table"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
