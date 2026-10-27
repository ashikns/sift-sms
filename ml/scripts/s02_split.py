"""s02: the inbox's time split, group-stratified train/val/test for the rest, a "future" holdout, and training caps.

    uv run python -I scripts/s02_split.py

Writes data/work/corpus_split.parquet with a `split` column:
  train | val_a | val_b | test | future | capped | inbox_eval
- inbox_eval: rows a source holds out for evaluation (schema `holdout`), i.e. the inbox's newest half. Nothing trains,
  selects, calibrates or tunes on them; s05/s06 only embed and featurize them for evaluation.
- The inbox's other rows (its oldest half) mainly train, split by time: the newest `time_val` of them validate (val_a
  and val_b balanced by cluster), the rest train. They're never tested, future or capped, and their templates recur
  across the cut by design.
- Synthetic sources (registry `train_only`) only train. Rows that s01 clustered with any of these training rows train
  too (templates move whole), so no test or future cluster holds a near-duplicate of one.
- future: clusters containing any timestamped message at or after the newest `future_frac` quantile of the rest.
- templates are split by template_id, and rows of other sources that s01 clustered with a template's messages follow
  it; everything else by cluster with StratifiedGroupKFold over (label, source).
- val is halved by cluster: val_a fits the stacker (s06), val_b scores it on clusters it didn't see. Everywhere else
  (s03 checkpoint selection, s04 export checks, s07 thresholds, the s08 simulation pool) both halves are used together.
- train keeps at most max(`max_per_cluster_train`, √n) rows per cluster of n and at most `max_spam_ratio` spam per ham
  (excess rows become `capped`, used nowhere). The inbox's rows count toward the ratio but are never capped.
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from smsml.pipeline import config, work
from smsml.sources import SOURCES

TRAIN_ONLY = frozenset(s.name for s in SOURCES if s.train_only)


def stable_unit(key: str, salt: str) -> float:
    h = hashlib.sha256((salt + key).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") / 2 ** 64


def assign_splits(df: pd.DataFrame, sc: dict, seed: int, train_only: frozenset[str] = TRAIN_ONLY) -> pd.DataFrame:
    """`df` with its `split` column (see the module docstring); `sc` is pipeline.yaml's split section."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = ""

    # Source holdouts: evaluation only.
    held = df["holdout"].notna()
    df.loc[held, "split"] = df.loc[held, "holdout"]

    # The holdout's source (the inbox) splits its other rows by time. The newest validate, cluster by cluster (largest
    # first, onto the smaller half); the older ones train.
    own = df["source"].isin(set(df.loc[held, "source"])) & ~held
    order = df[own].sort_values(["timestamp", "source_id"]).index
    late = order[len(order) - int(round(float(sc["time_val"]) * len(order))):]
    df.loc[order.difference(late), "split"] = "train"
    size = {"val_a": 0, "val_b": 0}
    for cid, n in df.loc[late].groupby("cluster_id").size().sort_values(ascending=False, kind="stable").items():
        half = min(size, key=size.get)
        size[half] += n
        df.loc[late[(df.loc[late, "cluster_id"] == cid).to_numpy()], "split"] = half

    # Synthetic sources only train, and so does whatever s01 clustered with these training rows (templates below).
    synthetic = df["source"].isin(train_only)
    df.loc[synthetic, "split"] = "train"
    trained = set(df.loc[own | synthetic, "cluster_id"])
    df.loc[(df["split"] == "") & df["cluster_id"].isin(trained) & (df["source"] != "templates"), "split"] = "train"

    # Future holdout: campaigns (clusters) that first appear after the cutoff, i.e. unseen in training.
    timed = df["split"] == ""
    stamped = df[timed].dropna(subset=["timestamp"])
    if len(stamped):
        cutoff = stamped["timestamp"].quantile(1 - float(sc["future_frac"]))
        start = stamped.groupby("cluster_id")["timestamp"].min()
        future_clusters = set(start[start >= cutoff].index)
        df.loc[timed & df["cluster_id"].isin(future_clusters), "split"] = "future"
        print(f"future cutoff {cutoff}: {len(future_clusters)} new campaigns, {int((df['split'] == 'future').sum())} rows")

    # Templates by template id (one clustered with a training row trains); their near-duplicates from other sources
    # follow them.
    tmask = (df["source"] == "templates") & (df["split"] == "")
    hot = set(df.loc[tmask & df["cluster_id"].isin(trained), "template_id"])
    for tid in df.loc[tmask, "template_id"].dropna().unique():
        u = stable_unit(str(tid), "tpl")
        part = "train" if tid in hot or u < sc["train"] else ("val" if u < sc["train"] + sc["val"] else "test")
        df.loc[tmask & (df["template_id"] == tid), "split"] = part
    of_template = df[tmask].groupby("cluster_id")["split"].first()
    follow = (df["split"] == "") & df["cluster_id"].isin(of_template.index)
    df.loc[follow, "split"] = df.loc[follow, "cluster_id"].map(of_template)

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

    # Training caps (not on the inbox's rows).
    train = df.index[(df["split"] == "train") & ~own]
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
    exempt = df.loc[own & (df["split"] == "train"), "label"]
    n_ham = int((t["label"] == 0).sum() + (exempt == 0).sum())
    n_spam = int((t["label"] == 1).sum() + (exempt == 1).sum())
    max_spam = int(sc["max_spam_ratio"] * n_ham)
    if n_spam > max_spam:
        budget = max_spam - int((exempt == 1).sum())
        spam_clusters = t.loc[t["label"] == 1, "cluster_id"].unique().tolist()
        rng.shuffle(spam_clusters)
        counts = t.loc[t["label"] == 1].groupby("cluster_id").size()
        kept, total = set(), 0
        for c in spam_clusters:
            if total + counts[c] > budget:
                continue
            kept.add(c)
            total += counts[c]
        drop = t.index[(t["label"] == 1) & ~t["cluster_id"].isin(kept)]
        capped |= set(drop)
    df.loc[list(capped), "split"] = "capped"
    return df


def leaking_clusters(df: pd.DataFrame) -> list[str]:
    """Clusters with rows in train and in an evaluation split. The inbox's time-split rows share templates across the
    cut by design, so a cluster holding them may mix train and val, but nothing else."""
    live = df[df["split"] != "capped"]
    by = live.groupby("cluster_id")["split"]
    own = df["source"].isin(set(df.loc[df["holdout"].notna(), "source"])) & df["holdout"].isna()
    timed = by.size().index.isin(set(df.loc[own, "cluster_id"]))
    has_train = live["split"].eq("train").groupby(live["cluster_id"]).any()
    beyond_val = (~live["split"].isin(["train", "val_a", "val_b"])).groupby(live["cluster_id"]).any()
    return by.size().index[np.where(timed, beyond_val, has_train & (by.nunique() > 1))].tolist()


def main() -> int:
    cfg = config()
    df = assign_splits(pd.read_parquet(work("corpus.parquet")), cfg["split"], int(cfg["seed"]))
    out = work("corpus_split.parquet")
    df.to_parquet(out, index=False)
    summary = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(summary.to_string())
    print(df.groupby(["source", "split", "label"]).size().unstack(fill_value=0).to_string())
    # Sanity: no training row's cluster in an evaluation split.
    leaking = leaking_clusters(df)
    if leaking:
        print(f"ERROR: {len(leaking)} clusters span train and an evaluation split", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
