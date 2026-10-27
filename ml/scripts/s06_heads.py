"""s06: train the base head (score_v1) from int8 embeddings + hashed n-grams + metadata.

    uv run python -I scripts/s06_heads.py

- IDF over the train split's feature buckets.
- Lexical LR on [hashed n-grams | centered metadata]; embedding LR on centered embeddings (class-balanced).
- A 3-parameter stacker z = a·z_embed + c·z_lexical + d (a, c ≥ 0) fit on val_a, folded into one linear model.
- Fallback head (no encoder) = the lexical LR alone: fb_bias = b_l, fb_scale = 1/c, w_meta_fb = w_l,meta.
Writes data/work/bundle/head_base.bin and data/work/features.npz (sparse/meta for every non-capped row).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from smsml.features import N_META
from smsml.head import Head
from smsml.hashing import BUCKETS
from smsml.pipeline import BUNDLE_WORK, config, document_frequencies, featurize, work, write_json
from smsml.features import idf_from_df


def balanced_weights(y: np.ndarray) -> np.ndarray:
    w = np.ones(len(y))
    for lab in (0, 1):
        n = max(1, int((y == lab).sum()))
        w[y == lab] = len(y) / (2.0 * n)
    return w


def fit_stacker(ze: np.ndarray, zl: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    w = balanced_weights(y)

    def f(t):
        a, c, d = t
        z = a * ze + c * zl + d
        loss = np.sum(w * (np.logaddexp(0, z) - y * z))
        r = w * (1 / (1 + np.exp(-z)) - y)
        return loss, np.array([np.sum(r * ze), np.sum(r * zl), np.sum(r)])

    res = minimize(f, np.array([0.5, 0.5, 0.0]), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, None), (0.05, None), (None, None)])
    return tuple(float(x) for x in res.x)


def main() -> int:
    cfg = config()
    hc = cfg["heads"]
    df = pd.read_parquet(work("corpus_split.parquet"))
    rows = np.load(work("emb_corpus_rows.npy"))
    emb = np.load(work("emb_corpus.npy"))
    sub = df.iloc[rows].reset_index(drop=True)
    split = sub["split"].to_numpy()
    y = sub["label"].to_numpy().astype(np.float64)
    tr = split == "train"

    dfreq = document_frequencies(sub.loc[tr, "text"].tolist())
    idf = idf_from_df(dfreq, int(tr.sum()))
    X, meta = featurize(sub["text"].tolist(), idf)
    sp.save_npz(work("features_X.npz"), X)
    np.save(work("features_meta.npy"), meta)
    np.save(work("idf.npy"), idf)

    mu_e = emb[tr].mean(axis=0).astype(np.float32)
    mu_m = meta[tr].mean(axis=0).astype(np.float32)
    Xl = sp.hstack([X, sp.csr_matrix(meta - mu_m)], format="csr")
    lex = LogisticRegression(C=hc["lexical_C"], class_weight="balanced", solver="lbfgs", max_iter=3000)
    lex.fit(Xl[tr], y[tr])
    w_l = lex.coef_[0]
    w_h, w_lm, b_l = w_l[:BUCKETS], w_l[BUCKETS:], float(lex.intercept_[0])
    embr = LogisticRegression(C=hc["embed_C"], class_weight="balanced", solver="lbfgs", max_iter=3000)
    embr.fit(emb[tr] - mu_e, y[tr])
    w_e, b_e = embr.coef_[0], float(embr.intercept_[0])

    zl = Xl @ w_l + b_l
    ze = (emb - mu_e) @ w_e + b_e
    va = split == "val_a"
    a, c, d = fit_stacker(ze[va], zl[va], y[va])
    head = Head(
        bias=a * b_e + c * b_l + d,
        w_embed=(a * w_e).astype(np.float32), mu_embed=mu_e,
        w_meta=(c * w_lm).astype(np.float32), mu_meta=mu_m,
        w_hash=(c * w_h).astype(np.float32), idf=idf,
        fb_bias=b_l, fb_scale=1.0 / c, w_meta_fb=w_lm.astype(np.float32),
    )
    BUNDLE_WORK.mkdir(parents=True, exist_ok=True)
    head.save(BUNDLE_WORK / "head_base.bin")

    # Folded head == stacker, up to float32 rounding.
    z_full = np.array([head.z(emb[i], meta[i], X[i].indices, X[i].data) for i in np.flatnonzero(va)[:500]])
    z_stack = (a * ze + c * zl + d)[va][:500]
    fold_err = float(np.abs(z_full - z_stack).max())
    report = {"stacker": {"a": a, "c": c, "d": d}, "fold_max_abs": fold_err, "n_meta": N_META}
    for name, mask in [("val_a", va), ("val_b", split == "val_b"), ("test", split == "test")]:
        report[name] = {
            "pr_auc_embed": float(average_precision_score(y[mask], ze[mask])),
            "pr_auc_lexical": float(average_precision_score(y[mask], zl[mask])),
            "pr_auc_stacked": float(average_precision_score(y[mask], (a * ze + c * zl + d)[mask])),
        }
    write_json(work("heads_report.json"), report)
    print(report)
    if fold_err > 1e-3:
        print("ERROR: folded head disagrees with the stacker", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
