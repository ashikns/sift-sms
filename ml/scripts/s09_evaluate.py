"""s09: evaluate the bundle against the acceptance gates; exits non-zero if any gate fails.

    uv run python -I scripts/s09_evaluate.py

Writes data/work/metrics.json. Gates:
  test        PR-AUC ≥ 0.99; Balanced FPR ≤ 0.5% (cluster-bootstrap 95% upper bound ≤ 1.0%), recall ≥ 95%
  future      recall ≥ 90% at Balanced (campaigns that first appeared after the training cutoff)
  modern      recall ≥ 85%, ham FPR ≤ 2%, OTP false positives: 0 at Conservative, ≤ 1 at Balanced
  robustness  invisible/homoglyph/fullwidth/math-letter variants normalize identically (100%);
              visible tricks cost ≤ 5 recall points
  int8        mean embedding cosine vs fp32 ≥ 0.99 (s04) and PR-AUC drop ≤ 0.002
  learning    (the user's marks are the only signal; operating point = "light retraining", chosen by the user)
              corrections carry over to repeats (missed-scam variants caught ≥ 60% and ≥ 3x the base model; flagged
              legit variants kept ≥ 70%); cost (5·FP + FN) not worse than no learning (diligent ≤ 100%,
              spam-only/noisy ≤ 105%); a spam-only marker newly flags ≤ 2 pt of held-out legit messages
"""

from __future__ import annotations

import json
import random
import sys

import numpy as np
import onnxruntime as ort
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import average_precision_score

from smsml.augment import augment
from smsml.decide import SPAM, Thresholds, decide
from smsml.features import META_NAMES, metadata, sparse_features
from smsml.head import Head
from smsml.normalize import normalize
from smsml.pipeline import BUNDLE_WORK, config, featurize, load_modern_eval, snapshot_dir, work, write_json
from smsml.resources import default_resources
from smsml.tokenizer import WordPiece

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from s07b_sender_prior import sender_meta  # noqa: E402

OTP, URLS = META_NAMES.index("otp_pattern"), META_NAMES.index("url_count")


def probs(head: Head, emb, X, meta) -> np.ndarray:
    z = head.bias + (emb.astype(np.float64) - head.mu_embed) @ head.w_embed.astype(np.float64) \
        + (meta.astype(np.float64) - head.mu_meta) @ head.w_meta.astype(np.float64) + X @ head.w_hash.astype(np.float64)
    return 1 / (1 + np.exp(-z))


def filtered(p: np.ndarray, meta: np.ndarray, thr: Thresholds, sensitivity: str) -> np.ndarray:
    out = []
    for pi, m in zip(p, meta):
        label, _ = decide(memory_match=None, p=float(pi),
                          otp_no_url=m[OTP] == 1.0 and m[URLS] == 0.0, thresholds=thr, sensitivity=sensitivity)
        out.append(label == SPAM)
    return np.array(out)


def bootstrap_fpr_upper(flags: np.ndarray, clusters: np.ndarray, n: int = 1000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    by = {c: flags[clusters == c] for c in uniq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        f = np.concatenate([by[c] for c in pick])
        vals.append(f.mean())
    return float(np.quantile(vals, 0.975))


def encoding_attacks(text: str, r: random.Random) -> list[str]:
    res = default_resources()
    inverse: dict[str, list[int]] = {}
    for cp, ascii_ in res.confusables.items():
        if len(ascii_) == 1:
            inverse.setdefault(ascii_, []).append(cp)
    zw = "".join(c + ("​" if r.random() < 0.3 else "") for c in text)
    homo = "".join(chr(r.choice(inverse[c])) if c in inverse and r.random() < 0.4 else c for c in text)
    full = "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" and r.random() < 0.5 else c for c in text)
    math = "".join(chr(0x1D400 + ord(c) - 65) if "A" <= c <= "Z" and r.random() < 0.5 else
                   (chr(0x1D41A + ord(c) - 97) if "a" <= c <= "z" and r.random() < 0.5 else c) for c in text)
    return [zw, homo, full, math]


def main() -> int:
    cfg = config()
    head = Head.load(BUNDLE_WORK / "head_base.bin")
    cal = json.loads(work("calibration.json").read_text())
    thr = Thresholds(**cal["thresholds"]["full"])
    df = pd.read_parquet(work("corpus_split.parquet"))
    rows = np.load(work("emb_corpus_rows.npy"))
    emb = np.load(work("emb_corpus.npy"))
    X = sp.load_npz(work("features_X.npz")).astype(np.float64)
    meta = np.load(work("features_meta.npy"))
    sub = df.iloc[rows].reset_index(drop=True)
    split, y = sub["split"].to_numpy(), sub["label"].to_numpy()
    p = probs(head, emb, X, meta)
    m: dict = {}
    gates: dict[str, bool] = {}

    # Test split.
    te = split == "test"
    f_bal = filtered(p[te], meta[te], thr, "balanced")
    ham, spam = y[te] == 0, y[te] == 1
    m["test"] = {
        "n_ham": int(ham.sum()), "n_spam": int(spam.sum()),
        "pr_auc": float(average_precision_score(y[te], p[te])),
        "balanced": {"fpr": float(f_bal[ham].mean()), "recall": float(f_bal[spam].mean()),
                     "fpr_upper95": bootstrap_fpr_upper(f_bal[ham], sub["cluster_id"].to_numpy()[te][ham])},
    }
    for s in ("conservative", "aggressive"):
        f = filtered(p[te], meta[te], thr, s)
        m["test"][s] = {"fpr": float(f[ham].mean()), "recall": float(f[spam].mean())}
    by_src = {}
    for src in sorted(set(sub["source"][te])):
        mk = (sub["source"].to_numpy()[te] == src) & spam
        if mk.any():
            by_src[src] = float(f_bal[mk].mean())
    m["test"]["recall_by_source"] = by_src
    gates["test_pr_auc"] = m["test"]["pr_auc"] >= 0.99
    gates["test_fpr"] = m["test"]["balanced"]["fpr"] <= 0.005 and m["test"]["balanced"]["fpr_upper95"] <= 0.01
    gates["test_recall"] = m["test"]["balanced"]["recall"] >= 0.95

    # Future campaigns.
    fu = split == "future"
    if fu.any():
        f = filtered(p[fu], meta[fu], thr, "balanced")
        m["future"] = {"n": int(fu.sum()), "recall": float(f.mean())}
        gates["future_recall"] = m["future"]["recall"] >= 0.90

    # Modern hand-written test set.
    modern = load_modern_eval()
    em = np.load(work("emb_modern.npy"))
    mt = (modern["split"] == "test").to_numpy()
    Xm, mm = featurize(modern.loc[mt, "text"].tolist(), head.idf)
    mm = sender_meta(mm, modern[mt])  # sender features as the phone computes them (s07b)
    pm = probs(head, em[mt], Xm.astype(np.float64), mm)
    ym = modern.loc[mt, "label"].to_numpy()
    cats = modern.loc[mt, "category"].to_numpy()
    fb = filtered(pm, mm, thr, "balanced")
    fc = filtered(pm, mm, thr, "conservative")
    otp_ham = (ym == 0) & (cats == "otp")
    m["modern"] = {
        "n": int(mt.sum()), "recall": float(fb[ym == 1].mean()), "ham_fpr": float(fb[ym == 0].mean()),
        "otp_fp_conservative": int(fc[otp_ham].sum()), "otp_fp_balanced": int(fb[otp_ham].sum()),
        "recall_by_category": {c: float(fb[(ym == 1) & (cats == c)].mean()) for c in sorted(set(cats[ym == 1]))},
        "false_positives": modern.loc[mt].loc[(fb & (ym == 0))[:]].assign(p=pm[fb & (ym == 0)])[["category", "text", "p"]]
        .to_dict("records") if (fb & (ym == 0)).any() else [],
        "missed_by_category": {c: int(((~fb) & (ym == 1) & (cats == c)).sum()) for c in sorted(set(cats[ym == 1]))},
    }
    gates["modern_recall"] = m["modern"]["recall"] >= 0.85
    gates["modern_fpr"] = m["modern"]["ham_fpr"] <= 0.02
    gates["modern_otp"] = m["modern"]["otp_fp_conservative"] == 0 and m["modern"]["otp_fp_balanced"] <= 1
    for s_ in ("conservative", "aggressive"):
        f_ = filtered(pm, mm, thr, s_)
        m["modern"][s_] = {"recall": float(f_[ym == 1].mean()), "ham_fpr": float(f_[ym == 0].mean())}

    # Robustness.
    r = random.Random(int(cfg["seed"]))
    test_spam = sub.loc[te & (y == 1), "text"].sample(n=min(400, int((te & (y == 1)).sum())), random_state=1).tolist()
    invariant = total = 0
    for t in test_spam:
        ref = normalize(t).enc
        for a in encoding_attacks(t, r):
            total += 1
            invariant += normalize(a).enc == ref
    m["robustness"] = {"encoding_invariance": invariant / total}
    wp = WordPiece.from_file(snapshot_dir(cfg) / "vocab.txt")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(str(BUNDLE_WORK / "encoder.int8.ort"), so, providers=["CPUExecutionProvider"])

    def score_texts(texts):
        out = []
        for t in texts:
            n = normalize(t)
            ids = np.asarray([wp.encode(n.enc)], dtype=np.int64)
            e = sess.run(["embedding"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]
            idx, val = sparse_features(n, head.idf)
            mt_ = metadata(n)
            z = head.z(e, mt_, idx, val)
            out.append((1 / (1 + np.exp(-z)), mt_))
        return out

    base_scores = score_texts(test_spam)
    attacked = score_texts([augment(t, r) for t in test_spam])
    rec0 = np.mean(filtered(np.array([s for s, _ in base_scores]), np.stack([mm_ for _, mm_ in base_scores]), thr, "balanced"))
    rec1 = np.mean(filtered(np.array([s for s, _ in attacked]), np.stack([mm_ for _, mm_ in attacked]), thr, "balanced"))
    m["robustness"]["visible_recall"] = {"clean": float(rec0), "attacked": float(rec1), "drop": float(rec0 - rec1)}
    gates["robust_encoding"] = m["robustness"]["encoding_invariance"] == 1.0
    gates["robust_visible"] = m["robustness"]["visible_recall"]["drop"] <= 0.05

    # int8 vs fp32.
    rep = json.loads(work("export", "report.json").read_text())
    so32 = ort.SessionOptions()
    so32.intra_op_num_threads = 8
    s32 = ort.InferenceSession(str(work("export", "encoder.fp32.onnx")), so32, providers=["CPUExecutionProvider"])
    idx_te = np.flatnonzero(te)
    pick = np.random.default_rng(3).choice(idx_te, size=min(1500, len(idx_te)), replace=False)
    e32 = []
    for i in pick:
        ids = np.asarray([wp.encode(sub["enc"][i])], dtype=np.int64)
        e32.append(s32.run(["embedding"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0])
    p32 = probs(head, np.stack(e32), X[pick], meta[pick])
    drop = float(average_precision_score(y[pick], p[pick]) - average_precision_score(y[pick], p32))
    m["int8"] = {"cosine_mean": rep["int8_vs_fp32_cosine"]["mean"], "cosine_min": rep["int8_vs_fp32_cosine"]["min"],
                 "pr_auc_int8_minus_fp32": drop}
    gates["int8"] = rep["int8_vs_fp32_cosine"]["mean"] >= 0.99 and drop <= 0.002

    # Learning (s08 --confirm: the chosen settings on 8 fresh streams). Classification is always per message, so
    # learning is judged on what a correction should do: carry over to the same message coming back (a scam resent
    # with new names, amounts and links; a legit sender's recurring notice), without making mixed streams worse.
    pers = json.loads(work("personalization.json").read_text())
    c = pers["confirm"]
    v = c["variants"]
    m["learning"] = {
        "params": c["params"],
        "missed_scam_variants_caught": v["missed_spam"]["right_with_learning"],
        "missed_scam_variants_caught_base": v["missed_spam"]["right_with_base_model"],
        "flagged_legit_variants_kept": v["flagged_ham"]["right_with_learning"],
        "flagged_legit_variants_kept_base": v["flagged_ham"]["right_with_base_model"],
        "diligent_cost_ratio": c["diligent"]["cost_ratio"],
        "spam_only_cost_ratio": c["spam_only"]["cost_ratio"],
        "noisy_cost_ratio": c["noisy"]["cost_ratio"],
        "diligent_repeat_campaign_error_ratio": c["diligent"]["repeat_error_ratio"],
        "spam_only_probe_fpr_delta": c["spam_only"]["probe_fpr_delta"],
    }
    # The operating point is the user's choice ("light retraining"): repeat handling vs. drift for a user who only
    # ever marks spam (held-out legit messages it newly flags). These gates hold that choice in place.
    gates["learning_repeats"] = (v["missed_spam"]["right_with_learning"] >= 0.6
                                 and v["missed_spam"]["right_with_learning"] >= 3 * v["missed_spam"]["right_with_base_model"]
                                 and v["flagged_ham"]["right_with_learning"] >= 0.7)
    gates["learning_no_harm"] = (c["diligent"]["cost_ratio"] <= 1.0 and c["spam_only"]["cost_ratio"] <= 1.05
                                 and c["noisy"]["cost_ratio"] <= 1.05)
    gates["learning_drift"] = c["spam_only"]["probe_fpr_delta"] <= 0.02

    m["gates"] = gates
    write_json(work("metrics.json"), m)
    printable = {k: v for k, v in m.items() if k != "modern"}
    print(json.dumps(printable, indent=2))
    print(json.dumps({k: v for k, v in m["modern"].items() if k != "false_positives"}, indent=2))
    for fp in m["modern"]["false_positives"]:
        print("  modern FP:", fp)
    failed = [k for k, v in gates.items() if not v]
    print("GATES:", "all passed" if not failed else f"FAILED {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
