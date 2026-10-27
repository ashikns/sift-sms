"""s01: unify the sources, normalize, drop exact duplicates, and cluster near-duplicates.

    uv run python -I scripts/s01_corpus.py

Writes data/work/corpus.parquet with norm_v1 columns (clean/enc/ng) and a cluster_id. Clusters come from MinHash
LSH over character 5-gram shingles of `ng` (Jaccard ≥ 0.7), merged with campaign IDs (NCSU campaigns, inbox
templates), so near-duplicates and campaign siblings never straddle splits. A cluster holding both labels is split
into one cluster per label. Holdout rows (schema `holdout`: the inbox's newest half) are kept as received, repeats
included, and clustered only among themselves (`h` cluster ids), so they never drop, merge or regroup another row.
Fails when the private inbox export is missing.
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

from smsml.pipeline import WORKERS, config, normalize_frame, work
from smsml.sources import load_all

_PERM = 128
_K = 5
_SEED = 1


def shingles(ng: str, k: int) -> list[bytes]:
    s = " " + ng + " "
    if len(s) <= k:
        return [s.encode("utf-8")]
    return list({s[i:i + k].encode("utf-8") for i in range(len(s) - k + 1)})


def _minhash(ng: str) -> tuple[np.ndarray, str]:
    m = MinHash(num_perm=_PERM, seed=_SEED)
    m.update_batch(shingles(ng, _K))
    return m.hashvalues, m.scheme


class UnionFind:
    def __init__(self, n: int):
        self.p = np.arange(n)

    def find(self, a: int) -> int:
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Exact duplicates on the n-gram view, keeping the first; a view seen with both labels loses every row."""
    key = pd.Series([hashlib.sha1(s.encode("utf-8")).hexdigest() for s in df["ng"]], index=df.index)
    labels_per_key = df["label"].groupby(key).nunique()
    conflicting = key.isin(labels_per_key.index[labels_per_key > 1])
    repeat = key.duplicated() & ~conflicting
    print(f"exact duplicates removed: {int(repeat.sum())}; rows with conflicting labels dropped: {int(conflicting.sum())}")
    return df[~conflicting & ~repeat].reset_index(drop=True)


def clusters(df: pd.DataFrame, jac: float, prefix: str) -> list[str]:
    """Cluster ids `<prefix><root>-<label>`: near-duplicates merged with campaign siblings, one cluster per label."""
    if df.empty:
        return []
    with ProcessPoolExecutor(WORKERS) as ex:
        hv = list(ex.map(_minhash, df["ng"].tolist(), chunksize=512))
    lsh = MinHashLSH(threshold=jac, num_perm=_PERM)
    mhs = []
    for i, (h, scheme) in enumerate(hv):
        m = MinHash(num_perm=_PERM, seed=_SEED, hashvalues=h, scheme=scheme)
        mhs.append(m)
        lsh.insert(i, m)
    uf = UnionFind(len(df))
    pairs = 0
    for i, m in enumerate(mhs):
        for j in lsh.query(m):
            if j > i and m.jaccard(mhs[j]) >= jac:
                uf.union(i, j)
                pairs += 1
    first_of: dict[str, int] = {}
    for i, c in enumerate(df["campaign_id"].fillna("")):
        if c:
            if c in first_of:
                uf.union(first_of[c], i)
            else:
                first_of[c] = i
    roots = np.array([uf.find(i) for i in range(len(df))])
    ids = [f"{prefix}{r}-{lab}" for r, lab in zip(roots, df["label"])]
    mixed = df.groupby(roots)["label"].nunique()
    print(f"[{prefix}] near-duplicate pairs: {pairs}; clusters: {len(set(ids))} "
          f"(label-mixed roots split: {int((mixed > 1).sum())})")
    return ids


def build(df: pd.DataFrame, jac: float) -> pd.DataFrame:
    """Normalized rows → the corpus: empty texts dropped, the rest deduplicated and clustered, holdout rows apart."""
    empty = df["clean"].str.len() == 0
    print(f"dropping {int(empty.sum())} rows that normalize to empty text")
    df = df[~empty]
    held = df["holdout"].notna()
    pool, hold = dedupe(df[~held]), df[held]
    out = pd.concat([pool, hold], ignore_index=True)
    out["cluster_id"] = clusters(pool, jac, "c") + clusters(hold, jac, "h")
    return out


def main() -> int:
    global _PERM, _K
    cfg = config()
    _PERM, _K = int(cfg["dedupe"]["num_perm"]), int(cfg["dedupe"]["shingle"])
    try:
        df = load_all()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print("loaded:", df.groupby(["source", "label"]).size().to_dict())
    df = build(normalize_frame(df), float(cfg["dedupe"]["jaccard"]))
    print("largest clusters:", Counter(df["cluster_id"]).most_common(5))
    out = work("corpus.parquet")
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows):", df.groupby(["source", "label"]).size().to_dict())
    print("of which held out:", df.groupby(["holdout", "label"]).size().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
