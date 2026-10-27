"""s02: group-stratified train/val/test split, a "future" holdout, and training caps.

    uv run python -I scripts/s02_split.py

Writes data/work/corpus_split.parquet with a `split` column:
  train | val_a | val_b | test | future | capped
- future: clusters containing any timestamped message at or after the newest `future_frac` quantile.
- templates are split by template_id; everything else by cluster with StratifiedGroupKFold over (label, source).
- val is halved by cluster: val_a fits the stacker (s06), val_b scores it on clusters it didn't see. Everywhere else
  (s03 checkpoint selection, s04 export checks, s07 thresholds, the s08 simulation pool) both halves are used together.
- train keeps at most max(`max_per_cluster_train`, √n) rows per cluster of n and at most `max_spam_ratio` spam per ham
  (excess rows become `capped`, used nowhere).
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from smsml.pipeline import config, work


def stable_unit(key: str, salt: str) -> float:
    h = hashlib.sha256((salt + key).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") / 2 ** 64


def main() -> int:
    cfg = config()
    sc = cfg["split"]
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(work("corpus.parquet"))
    df["split"] = ""

    # Future holdout: campaigns (clusters) that first appear after the cutoff, i.e. unseen in training.
    ts = df["timestamp"].dropna()
    if len(ts):
        cutoff = ts.quantile(1 - float(sc["future_frac"]))
        start = df.dropna(subset=["timestamp"]).groupby("cluster_id")["timestamp"].min()
        future_clusters = set(start[start >= cutoff].index)
        df.loc[df["cluster_id"].isin(future_clusters), "split"] = "future"
        print(f"future cutoff {cutoff}: {len(future_clusters)} new campaigns, {int((df['split'] == 'future').sum())} rows")

    # Templates by template id.
    tmask = (df["source"] == "templates") & (df["split"] == "")
    for tid in df.loc[tmask, "template_id"].dropna().unique():
        u = stable_unit(str(tid), "tpl")
        part = "train" if u < sc["train"] else ("val" if u < sc["train"] + sc["val"] else "test")
        df.loc[tmask & (df["template_id"] == tid), "split"] = part

    # Everything else by cluster.
    rest = df.index[df["split"] == ""]
    strata = (df.loc[rest, "label"].astype(str) + "|" + df.loc[rest, "source"].astype(str)).values
    groups = df.loc[rest, "cluster_id"].values
    folds = np.full(len(rest), -1)
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(rest)), strata, groups)):
        folds[test_idx] = k
    n_train, n_val = round(20 * sc["train"]), round(20 * sc["val"])
    df.loc[rest, "split"] = np.where(folds < n_train, "train", np.where(folds < n_train + n_val, "val", "test"))

    # Halve val by cluster.
    val = df["split"] == "val"
    df.loc[val, "split"] = ["val_a" if stable_unit(c, "val") < 0.5 else "val_b" for c in df.loc[val, "cluster_id"]]

    # Training caps.
    train = df.index[df["split"] == "train"]
    keep = []
    for _, idx in df.loc[train].groupby("cluster_id").groups.items():
        idx = list(idx)
        # Big clusters are transitive near-duplicate chains; keep √n of them so their variety isn't lost.
        cap = max(int(sc["max_per_cluster_train"]), int(np.ceil(np.sqrt(len(idx)))))
        if len(idx) > cap:
            idx = list(rng.choice(idx, size=cap, replace=False))
        keep.extend(idx)
    capped = set(train) - set(keep)
    t = df.loc[keep]
    n_ham, n_spam = int((t["label"] == 0).sum()), int((t["label"] == 1).sum())
    max_spam = int(sc["max_spam_ratio"] * n_ham)
    if n_spam > max_spam:
        spam_clusters = t.loc[t["label"] == 1, "cluster_id"].unique()
        rng.shuffle(spam_clusters)
        counts = t.loc[t["label"] == 1].groupby("cluster_id").size()
        kept, total = set(), 0
        for c in spam_clusters:
            if total + counts[c] > max_spam:
                continue
            kept.add(c)
            total += counts[c]
        drop = t.index[(t["label"] == 1) & ~t["cluster_id"].isin(kept)]
        capped |= set(drop)
    df.loc[list(capped), "split"] = "capped"

    out = work("corpus_split.parquet")
    df.to_parquet(out, index=False)
    summary = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(summary.to_string())
    print(df[df["split"].isin(["train", "test"])].groupby(["split", "source", "label"]).size().unstack(fill_value=0).to_string())
    # Sanity: no cluster in two evaluation-relevant splits.
    multi = df[df["split"] != "capped"].groupby("cluster_id")["split"].nunique()
    leaking = multi[multi > 1]
    leaking = [c for c in leaking.index if set(df.loc[df["cluster_id"] == c, "split"]) - {"capped", "train"}
               and "train" in set(df.loc[df["cluster_id"] == c, "split"])]
    if leaking:
        print(f"ERROR: {len(leaking)} clusters span train and an evaluation split", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
