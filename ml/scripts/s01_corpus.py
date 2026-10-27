"""s01: unify the sources, normalize, drop exact duplicates, and cluster near-duplicates.

    uv run python -I scripts/s01_corpus.py

Writes data/work/corpus.parquet with norm_v1 columns (clean/enc/ng) and a cluster_id. Clusters come from MinHash
LSH over character 5-gram shingles of `ng` (Jaccard ≥ 0.7), merged with NCSU campaign IDs, so near-duplicates and
campaign siblings never straddle splits. A cluster holding both labels is split into one cluster per label.
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
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


def main() -> int:
    global _PERM, _K
    cfg = config()
    _PERM, _K = int(cfg["dedupe"]["num_perm"]), int(cfg["dedupe"]["shingle"])
    jac = float(cfg["dedupe"]["jaccard"])

    df = load_all()
    print("loaded:", df.groupby(["source", "label"]).size().to_dict())
    df = normalize_frame(df)
    empty = df["clean"].str.len() == 0
    print(f"dropping {int(empty.sum())} rows that normalize to empty text")
    df = df[~empty].reset_index(drop=True)

    # Exact duplicates on the n-gram view; conflicting labels are dropped entirely.
    df["dedupe_key"] = [hashlib.sha1(s.encode("utf-8")).hexdigest() for s in df["ng"]]
    labels_per_key = df.groupby("dedupe_key")["label"].nunique()
    conflicting = set(labels_per_key[labels_per_key > 1].index)
    n_conf = int(df["dedupe_key"].isin(conflicting).sum())
    df = df[~df["dedupe_key"].isin(conflicting)]
    before = len(df)
    df = df.drop_duplicates("dedupe_key", keep="first").reset_index(drop=True)
    print(f"exact duplicates removed: {before - len(df)}; rows with conflicting labels dropped: {n_conf}")

    # Near-duplicate clusters.
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
    camp = df["campaign_id"].fillna("")
    first_of: dict[str, int] = {}
    for i, c in enumerate(camp):
        if c:
            if c in first_of:
                uf.union(first_of[c], i)
            else:
                first_of[c] = i
    roots = np.array([uf.find(i) for i in range(len(df))])
    # Split label-mixed clusters per label.
    df["cluster_id"] = [f"c{r}-{lab}" for r, lab in zip(roots, df["label"])]
    mixed = df.groupby(roots)["label"].nunique()
    print(f"near-duplicate pairs: {pairs}; clusters: {df['cluster_id'].nunique()} "
          f"(label-mixed roots split: {int((mixed > 1).sum())})")
    sizes = Counter(df["cluster_id"])
    print("largest clusters:", sizes.most_common(5))
    out = work("corpus.parquet")
    df.drop(columns=["dedupe_key"]).to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows):", df.groupby(["source", "label"]).size().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
