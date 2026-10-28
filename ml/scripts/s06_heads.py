"""s06: train the base head (score_v2) from int8 embeddings + hashed n-grams + sender × content pairs + metadata.

    uv run python -I scripts/s06_heads.py

- IDF over the train split's feature buckets.
- Lexical LR on [hashed n-grams | SND vectors | centered metadata]; embedding LR on centered embeddings
  (class-balanced). The SND block (rows with a sender; the metadata's sender-type features stay 0) gets its own L2
  strength: its columns are scaled by sqrt(snd_C / lexical_C), which makes its effective C snd_C. snd_C is picked from
  heads.snd_C by the lexical model's PR-AUC on the val rows that have a sender (ties: the stronger regularization).
- A 3-parameter stacker z = a·z_embed + c·z_lexical + d (a, c ≥ 0) fit on val_a, folded into one linear model.
- Fallback head (no encoder) = the lexical LR alone: fb_bias = b_l, fb_scale = 1/c, w_meta_fb = w_l,meta. It adds the
  full head's SND term unscaled (SPEC.md), i.e. c times the lexical LR's.
- Out-of-fold logits for the inbox's older half, for s07b's DLT-category weights: the inbox's training rows are split
  into 5 folds by near-duplicate cluster, and each fold is scored by both LRs refit without it (same C values, the same
  stacker), so every message reads like one from a template the model hasn't learned; the validation rows keep the
  full fit's logits (they never train).
Writes data/work/bundle/head_base.bin, data/work/features_X.npz + features_S.npz + features_meta.npy (n-gram, SND and
metadata features for every non-capped row), data/work/inbox_oof.npz (`rows` into those rows, content logits `z`
without the sender features) and data/work/heads_report.json.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from smsml.features import N_META, idf_from_df
from smsml.head import Head
from smsml.hashing import BUCKETS
from smsml.pipeline import BUNDLE_WORK, config, document_frequencies, featurize, senders_of, work, write_json

OOF_FOLDS = 5


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


def fit_lexical(X: sp.csr_matrix, S: sp.csr_matrix, meta_c: np.ndarray, y: np.ndarray, rows: np.ndarray, C: float,
                snd_C: float, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(w_hash, w_snd, w_meta, bias) of the lexical LR on `rows`, the SND block regularized with snd_C."""
    scale = float(np.sqrt(snd_C / C))
    Xl = sp.hstack([X, S * scale, sp.csr_matrix(meta_c)], format="csr")
    lex = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=3000)
    lex.fit(Xl[rows], y[rows], sample_weight=weight[rows])
    w = lex.coef_[0]
    return w[:BUCKETS], w[BUCKETS:2 * BUCKETS] * scale, w[2 * BUCKETS:], float(lex.intercept_[0])


def fit_embed(emb_c: np.ndarray, y: np.ndarray, rows: np.ndarray, C: float, weight: np.ndarray) -> tuple[np.ndarray, float]:
    embr = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=3000)
    embr.fit(emb_c[rows], y[rows], sample_weight=weight[rows])
    return embr.coef_[0], float(embr.intercept_[0])


def inbox_oof(X, S, meta_c, emb_c, y, sub: pd.DataFrame, weight, lexical_C: float, snd_C: float, embed_C: float,
              stacker: tuple[float, float, float], z_stacked: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(rows, z) for the inbox's older half: out-of-fold stacked logits for its training rows (folds by near-duplicate
    cluster), the full fit's for its validation rows."""
    a, c, d = stacker
    split = sub["split"].to_numpy()
    inbox = (sub["source"] == "inbox").to_numpy()
    tr = split == "train"
    clusters = sub["cluster_id"].astype(str).to_numpy()
    order = np.random.default_rng(seed).permutation(np.unique(clusters[inbox & tr]))
    fold_of = {cl: i % OOF_FOLDS for i, cl in enumerate(order)}
    fold = np.array([fold_of.get(cl, -1) if ib and t else -1 for cl, ib, t in zip(clusters, inbox, tr)])
    z = z_stacked.copy()
    for k in range(OOF_FOLDS):
        held = np.flatnonzero(fold == k)
        keep = np.flatnonzero(tr & (fold != k))
        w_h, w_s, w_lm, b_l = fit_lexical(X, S, meta_c, y, keep, lexical_C, snd_C, weight)
        w_e, b_e = fit_embed(emb_c, y, keep, embed_C, weight)
        zl = X[held] @ w_h + S[held] @ w_s + meta_c[held] @ w_lm + b_l
        z[held] = a * (emb_c[held] @ w_e + b_e) + c * zl + d
    rows = np.flatnonzero(inbox & np.isin(split, ["train", "val_a", "val_b"]))
    return rows, z[rows]


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
    # Per-source sample weights (the user's own inbox is a small share of the rows but the distribution that matters).
    weight = sub["source"].map(lambda s: float(hc["source_weight"].get(s, 1.0))).to_numpy()

    dfreq = document_frequencies(sub.loc[tr, "text"].tolist())
    idf = idf_from_df(dfreq, int(tr.sum()))
    X, S, meta = featurize(sub["text"].tolist(), idf, senders_of(sub))
    sp.save_npz(work("features_X.npz"), X)
    sp.save_npz(work("features_S.npz"), S)
    np.save(work("features_meta.npy"), meta)

    mu_e = emb[tr].mean(axis=0).astype(np.float32)
    mu_m = meta[tr].mean(axis=0).astype(np.float32)
    meta_c = meta - mu_m
    lexical_C = float(hc["lexical_C"])
    snd_grid = sorted(float(v) for v in hc["snd_C"])
    has_snd = np.diff(S.indptr) > 0
    val_snd = np.isin(split, ["val_a", "val_b"]) & has_snd
    # Tuning needs train rows with a sender and both labels among the val rows with one; otherwise the strongest
    # regularization (an empty SND block when no row has a sender).
    tunable = bool((tr & has_snd).any()) and len(np.unique(y[val_snd])) == 2
    fits, snd_pr_auc = {}, {}
    for snd_C in snd_grid if tunable else snd_grid[:1]:
        fits[snd_C] = fit_lexical(X, S, meta_c, y, np.flatnonzero(tr), lexical_C, snd_C, weight)
        if tunable:
            w_h, w_s, w_lm, b_l = fits[snd_C]
            zl_val = X[val_snd] @ w_h + S[val_snd] @ w_s + meta_c[val_snd] @ w_lm + b_l
            snd_pr_auc[snd_C] = float(average_precision_score(y[val_snd], zl_val))
    snd_C = max(fits, key=lambda v: (snd_pr_auc.get(v, 0.0), -v))
    w_h, w_s, w_lm, b_l = fits[snd_C]
    w_e, b_e = fit_embed(emb - mu_e, y, np.flatnonzero(tr), hc["embed_C"], weight)

    zl = X @ w_h + S @ w_s + meta_c @ w_lm + b_l
    ze = (emb - mu_e) @ w_e + b_e
    va = split == "val_a"
    a, c, d = fit_stacker(ze[va], zl[va], y[va])
    head = Head(
        bias=a * b_e + c * b_l + d,
        w_embed=(a * w_e).astype(np.float32), mu_embed=mu_e,
        w_meta=(c * w_lm).astype(np.float32), mu_meta=mu_m,
        w_hash=(c * w_h).astype(np.float32), w_snd=(c * w_s).astype(np.float32), idf=idf,
        fb_bias=b_l, fb_scale=1.0 / c, w_meta_fb=w_lm.astype(np.float32),
    )
    BUNDLE_WORK.mkdir(parents=True, exist_ok=True)
    head.save(BUNDLE_WORK / "head_base.bin")
    z_stacked = a * ze + c * zl + d
    oof_rows, oof_z = inbox_oof(X, S, meta_c, emb - mu_e, y, sub, weight, lexical_C, snd_C, hc["embed_C"], (a, c, d),
                                z_stacked, int(cfg["seed"]))
    np.savez(work("inbox_oof.npz"), rows=oof_rows, z=oof_z)

    # Folded head == stacker, up to float32 rounding (rows with a sender first, so the SND term is checked too).
    check = np.concatenate([np.flatnonzero(va & has_snd)[:250], np.flatnonzero(va & ~has_snd)])[:500]
    z_full = np.array([head.z(emb[i], meta[i], X[i].indices, X[i].data, S[i].indices, S[i].data) for i in check])
    z_stack = (a * ze + c * zl + d)[check]
    fold_err = float(np.abs(z_full - z_stack).max())
    report = {"stacker": {"a": a, "c": c, "d": d}, "fold_max_abs": fold_err, "n_meta": N_META,
              "snd": {"C": snd_C, "val_pr_auc_by_C": {f"{v:g}": auc for v, auc in snd_pr_auc.items()},
                      "train_rows": int((tr & has_snd).sum()), "val_rows": int(val_snd.sum())},
              "inbox_oof": {"folds": OOF_FOLDS, "rows": len(oof_rows),
                            "pr_auc_in_sample": float(average_precision_score(y[oof_rows], z_stacked[oof_rows])),
                            "pr_auc_out_of_fold": float(average_precision_score(y[oof_rows], oof_z))}}
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
