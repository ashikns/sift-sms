"""The linear head (score_v2) and the on-device personalization objective, mirrored by spamcore.

Personalization is MAP logistic regression anchored to the base weights:
    minimize  Σ_i c_i · logloss(z_i, y_i)  +  ½ Σ_g λ_g ‖w_g − w_base,g‖²
over the embedding block, the content metadata, the hashed n-gram weights and the sender × content (SND) weights
that occur in the examples (each sparse block's "active set"). The bias and the sender-type features stay frozen, so
marks only teach the model about message content, and a sender only through its pairs with the message's words and
link domains. The problem is strictly convex, so any converged solver (scipy here, a hand-written L-BFGS in Kotlin)
reaches the same optimum.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

from . import bundle
from .features import N_META, SENDER_SLICE, sparse_cosine
from .hashing import BUCKETS

EMBED_DIM = 384


@dataclass(frozen=True)
class Head:
    bias: float
    w_embed: np.ndarray
    mu_embed: np.ndarray
    w_meta: np.ndarray
    mu_meta: np.ndarray
    w_hash: np.ndarray
    w_snd: np.ndarray
    idf: np.ndarray
    fb_bias: float
    fb_scale: float
    w_meta_fb: np.ndarray

    def to_blocks(self) -> dict[str, np.ndarray]:
        f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
        return {
            "bias": np.asarray([self.bias], dtype=np.float64),
            "w_embed": f32(self.w_embed),
            "mu_embed": f32(self.mu_embed),
            "w_meta": f32(self.w_meta),
            "mu_meta": f32(self.mu_meta),
            "w_hash": f32(self.w_hash),
            "w_snd": f32(self.w_snd),
            "idf": f32(self.idf),
            "fb_bias": np.asarray([self.fb_bias], dtype=np.float64),
            "fb_scale": np.asarray([self.fb_scale], dtype=np.float64),
            "w_meta_fb": f32(self.w_meta_fb),
        }

    @staticmethod
    def from_blocks(b: dict[str, np.ndarray]) -> "Head":
        return Head(
            bias=float(b["bias"][0]), w_embed=b["w_embed"], mu_embed=b["mu_embed"], w_meta=b["w_meta"],
            mu_meta=b["mu_meta"], w_hash=b["w_hash"], w_snd=b["w_snd"], idf=b["idf"], fb_bias=float(b["fb_bias"][0]),
            fb_scale=float(b["fb_scale"][0]), w_meta_fb=b["w_meta_fb"],
        )

    def save(self, path) -> None:
        bundle.write(path, self.to_blocks())

    @staticmethod
    def load(path) -> "Head":
        return Head.from_blocks(bundle.read(path))

    def z(self, emb: np.ndarray | None, meta: np.ndarray, idx: np.ndarray, val: np.ndarray, snd_idx: np.ndarray,
          snd_val: np.ndarray) -> float:
        """score_v2 logit, accumulated in double in SPEC order. A missing embedding contributes 0 (its mean)."""
        z = self.bias
        if emb is not None:
            z += float(np.dot(self.w_embed.astype(np.float64), emb.astype(np.float64) - self.mu_embed.astype(np.float64)))
        z += float(np.dot(self.w_meta.astype(np.float64), meta.astype(np.float64) - self.mu_meta.astype(np.float64)))
        z += float(np.dot(self.w_hash[idx].astype(np.float64), val.astype(np.float64)))
        z += self.snd_term(snd_idx, snd_val)
        return z

    def z_fallback(self, meta: np.ndarray, idx: np.ndarray, val: np.ndarray, snd_idx: np.ndarray,
                   snd_val: np.ndarray) -> float:
        z = self.fb_bias + self.fb_scale * float(np.dot(self.w_hash[idx].astype(np.float64), val.astype(np.float64)))
        z += float(np.dot(self.w_meta_fb.astype(np.float64), meta.astype(np.float64) - self.mu_meta.astype(np.float64)))
        z += self.snd_term(snd_idx, snd_val)  # unscaled: the same SND term as the full head
        return z

    def snd_term(self, snd_idx: np.ndarray, snd_val: np.ndarray) -> float:
        return float(np.dot(self.w_snd[snd_idx].astype(np.float64), snd_val.astype(np.float64)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class Example:
    emb: np.ndarray | None
    meta: np.ndarray
    idx: np.ndarray
    val: np.ndarray
    snd_idx: np.ndarray
    snd_val: np.ndarray
    label: int
    key: int = 0


FROZEN = 1e5  # a block whose prior is at least this stays at its base weights


@dataclass(frozen=True)
class Priors:
    embed: float = 1.0
    meta: float = 4.0
    hash: float = 0.5
    snd: float = 0.5


@dataclass(frozen=True)
class Weighting:
    spam: float = 1.0
    ham: float = 1.0
    cluster_cap: float = 2.0
    cluster_cos: float = 0.8
    max_examples: int = 5000


def example_weights(examples: list[Example], w: Weighting) -> np.ndarray:
    """Base weight per example, then cap each near-duplicate cluster's total weight (scaled proportionally).

    Examples are the user's marks, oldest-first; only the newest max_examples are kept. Clusters use leader
    clustering in list order: an example joins the first earlier cluster (same label) whose leader has sparse cosine
    ≥ cluster_cos with it, otherwise it starts a new cluster. Mirrored exactly by spamcore's ExampleWeighting.
    """
    n = len(examples)
    base = np.array([w.spam if e.label == 1 else w.ham for e in examples], dtype=np.float64)
    keep = np.arange(n) >= n - w.max_examples
    base[~keep] = 0.0
    leaders: list[int] = []
    cluster_of = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if not keep[i]:
            continue
        e = examples[i]
        for c, lead in enumerate(leaders):
            le = examples[lead]
            if le.label == e.label and sparse_cosine(le.idx, le.val, e.idx, e.val) >= w.cluster_cos:
                cluster_of[i] = c
                break
        else:
            cluster_of[i] = len(leaders)
            leaders.append(i)
    totals = np.zeros(len(leaders), dtype=np.float64)
    for i in range(n):
        if keep[i]:
            totals[cluster_of[i]] += base[i]
    out = base.copy()
    for i in range(n):
        if keep[i] and totals[cluster_of[i]] > w.cluster_cap:
            out[i] = base[i] * (w.cluster_cap / totals[cluster_of[i]])
    return out


def personalize(base: Head, examples: list[Example], weights: np.ndarray, priors: Priors,
                tol: float = 1e-10, max_iter: int = 500) -> Head:
    """Exact MAP optimum over the active sets (scipy L-BFGS-B), used as the reference for the Kotlin trainer.

    Parameters, in order: the embedding block, the metadata, the hashed n-gram weights of the examples' feature
    vectors and the SND weights of their SND vectors. The two sparse blocks are separate even where they share a
    bucket index.
    """
    use = [i for i in range(len(examples)) if weights[i] > 0]
    if not use:
        return base
    active = np.unique(np.concatenate([examples[i].idx for i in use]).astype(np.int64))
    snd_active = np.unique(np.concatenate([examples[i].snd_idx for i in use]).astype(np.int64))
    pos = {int(k): j for j, k in enumerate(active)}
    snd_pos = {int(k): j for j, k in enumerate(snd_active)}
    d_e, d_m, d_h = EMBED_DIM, N_META, len(active)
    o_h, o_s = d_e + d_m, d_e + d_m + d_h
    w0 = np.concatenate([
        base.w_embed.astype(np.float64), base.w_meta.astype(np.float64), base.w_hash[active].astype(np.float64),
        base.w_snd[snd_active].astype(np.float64),
    ])
    lam = np.empty_like(w0)
    lam[:d_e] = priors.embed
    lam[d_e:o_h] = priors.meta
    lam[o_h:o_s] = priors.hash
    lam[o_s:] = priors.snd
    frozen = np.zeros_like(w0, dtype=bool)
    frozen[d_e + SENDER_SLICE.start:d_e + SENDER_SLICE.stop] = True
    frozen[:d_e] |= priors.embed >= FROZEN
    frozen[d_e:o_h] |= priors.meta >= FROZEN
    frozen[o_h:o_s] |= priors.hash >= FROZEN
    frozen[o_s:] |= priors.snd >= FROZEN
    mu_e = base.mu_embed.astype(np.float64)
    mu_m = base.mu_meta.astype(np.float64)
    rows_e = np.stack([
        (examples[i].emb.astype(np.float64) - mu_e) if examples[i].emb is not None else np.zeros(d_e) for i in use
    ])
    rows_m = np.stack([examples[i].meta.astype(np.float64) - mu_m for i in use])
    hash_rows = [(np.array([pos[int(k)] for k in examples[i].idx], dtype=np.int64),
                  examples[i].val.astype(np.float64)) for i in use]
    snd_rows = [(np.array([snd_pos[int(k)] for k in examples[i].snd_idx], dtype=np.int64),
                 examples[i].snd_val.astype(np.float64)) for i in use]
    y = np.array([examples[i].label for i in use], dtype=np.float64)
    c = np.array([weights[i] for i in use], dtype=np.float64)

    def f(theta: np.ndarray):
        we, wm, wh, ws = theta[:d_e], theta[d_e:o_h], theta[o_h:o_s], theta[o_s:]
        z = base.bias + rows_e @ we + rows_m @ wm + np.array([wh[p] @ v for p, v in hash_rows]) \
            + np.array([ws[p] @ v for p, v in snd_rows])
        # logloss = log(1 + e^z) - y z, computed stably
        loss = float(np.sum(c * (np.logaddexp(0.0, z) - y * z)))
        diff = theta - w0
        loss += 0.5 * float(np.sum(lam * diff * diff))
        r = c * (sigmoid(z) - y)
        g = np.empty_like(theta)
        g[:d_e] = rows_e.T @ r
        g[d_e:o_h] = rows_m.T @ r
        gh = np.zeros(d_h)
        for (p, v), ri in zip(hash_rows, r):
            np.add.at(gh, p, ri * v)
        g[o_h:o_s] = gh
        gs = np.zeros(len(snd_active))
        for (p, v), ri in zip(snd_rows, r):
            np.add.at(gs, p, ri * v)
        g[o_s:] = gs
        g += lam * diff
        g[frozen] = 0.0
        return loss, g

    res = minimize(f, w0.copy(), jac=True, method="L-BFGS-B", options={"maxiter": max_iter, "gtol": tol, "ftol": 1e-15})
    theta = res.x
    w_hash = base.w_hash.copy()
    w_hash[active] = theta[o_h:o_s].astype(np.float32)
    w_snd = base.w_snd.copy()
    w_snd[snd_active] = theta[o_s:].astype(np.float32)
    return replace(
        base,
        w_embed=theta[:d_e].astype(np.float32),
        w_meta=theta[d_e:o_h].astype(np.float32),
        w_hash=w_hash,
        w_snd=w_snd,
    )


def empty_head(idf: np.ndarray | None = None) -> Head:
    z = lambda n: np.zeros(n, dtype=np.float32)  # noqa: E731
    return Head(0.0, z(EMBED_DIM), z(EMBED_DIM), z(N_META), z(N_META), z(BUCKETS), z(BUCKETS),
                idf if idf is not None else np.ones(BUCKETS, dtype=np.float32), 0.0, 1.0, z(N_META))
