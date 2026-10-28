"""MurmurHash3 x86_32 feature hashing (SPEC.md feat_v3). `murmur3_32` is a readable reference; `mmh3` is used for speed."""

from __future__ import annotations

import mmh3

SEED = 0x53504D31
BITS = 18
BUCKETS = 1 << BITS
MASK = BUCKETS - 1


def _rotl32(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF


def murmur3_32(data: bytes, seed: int = 0) -> int:
    c1, c2 = 0xCC9E2D51, 0x1B873593
    h = seed & 0xFFFFFFFF
    n = len(data)
    nblocks = n // 4
    for i in range(nblocks):
        k = int.from_bytes(data[4 * i: 4 * i + 4], "little")
        k = (k * c1) & 0xFFFFFFFF
        k = _rotl32(k, 15)
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
        h = _rotl32(h, 13)
        h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
    tail = data[4 * nblocks:]
    k = 0
    if len(tail) == 3:
        k ^= tail[2] << 16
    if len(tail) >= 2:
        k ^= tail[1] << 8
    if len(tail) >= 1:
        k ^= tail[0]
        k = (k * c1) & 0xFFFFFFFF
        k = _rotl32(k, 15)
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
    h ^= n
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def hash_feature(feature: str) -> tuple[int, int]:
    """Returns (bucket index, sign)."""
    h = mmh3.hash(feature.encode("utf-8"), SEED, signed=False)
    return h & MASK, (1 if (h >> 31) == 0 else -1)
