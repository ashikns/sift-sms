from dataclasses import replace

import numpy as np
import pytest

from smsml.features import N_META, SENDER_SLICE
from smsml.head import EMBED_DIM, FROZEN, Example, Head, Priors, Weighting, empty_head, example_weights, personalize

EMPTY_IDX = np.zeros(0, dtype=np.int32)
EMPTY_VAL = np.zeros(0, dtype=np.float32)


def small_head(seed: int = 3) -> Head:
    rng = np.random.default_rng(seed)
    head = empty_head()
    w_hash = head.w_hash.copy()
    w_hash[[5, 9, 40]] = [0.5, -1.0, 2.0]
    w_snd = head.w_snd.copy()
    w_snd[[5, 7, 300]] = [1.5, -0.25, 3.0]
    return replace(head, bias=-0.5, w_embed=rng.normal(0, 0.5, EMBED_DIM).astype(np.float32),
                   w_meta=rng.normal(0, 0.5, N_META).astype(np.float32), mu_meta=np.full(N_META, 0.1, np.float32),
                   w_hash=w_hash, w_snd=w_snd, fb_bias=-1.0, fb_scale=2.0,
                   w_meta_fb=np.full(N_META, 0.2, np.float32))


def test_z_adds_the_snd_term_and_the_fallback_adds_it_unscaled():
    head = small_head()
    rng = np.random.default_rng(1)
    emb = rng.normal(0, 1, EMBED_DIM).astype(np.float32)
    meta = rng.uniform(0, 1, N_META).astype(np.float32)
    idx, val = np.array([5, 9], np.int32), np.array([0.6, 0.8], np.float32)
    s_idx, s_val = np.array([5, 300], np.int32), np.array([0.6, -0.8], np.float32)
    snd = 1.5 * float(np.float32(0.6)) + 3.0 * float(np.float32(-0.8))
    assert head.z(emb, meta, idx, val, s_idx, s_val) == pytest.approx(head.z(emb, meta, idx, val, EMPTY_IDX, EMPTY_VAL) + snd)
    assert head.z_fallback(meta, idx, val, s_idx, s_val) == pytest.approx(
        head.z_fallback(meta, idx, val, EMPTY_IDX, EMPTY_VAL) + snd)
    # The same bucket index means different weights in the two sparse vectors.
    assert head.z(None, meta, s_idx, s_val, EMPTY_IDX, EMPTY_VAL) != head.z(None, meta, EMPTY_IDX, EMPTY_VAL, s_idx, s_val)


def test_head_round_trips_w_snd(tmp_path):
    head = small_head()
    head.save(tmp_path / "head.bin")
    back = Head.load(tmp_path / "head.bin")
    assert np.array_equal(back.w_snd, head.w_snd) and np.array_equal(back.w_hash, head.w_hash)


def example(rng, label, snd_idx, snd_val, key, hash_idx=None):
    hash_idx = np.array([5, 9, 40] if hash_idx is None else hash_idx, np.int32)
    v = rng.uniform(0.2, 1.0, hash_idx.size)
    emb = rng.normal(0, 1, EMBED_DIM)
    return Example((emb / np.linalg.norm(emb)).astype(np.float32), rng.uniform(0, 1, N_META).astype(np.float32),
                   hash_idx, (v / np.linalg.norm(v)).astype(np.float32), np.asarray(snd_idx, np.int32),
                   np.asarray(snd_val, np.float32), label, key=key)


def test_personalize_learns_the_snd_block_only_where_examples_have_a_sender():
    head = small_head()
    rng = np.random.default_rng(2)
    examples = [example(rng, 0, [5, 300], [0.6, 0.8], 1), example(rng, 1, [], [], 2), example(rng, 0, [7], [1.0], 3)]
    weights = example_weights(examples, Weighting(cluster_cos=2.0))
    out = personalize(head, examples, weights, Priors(embed=FROZEN, meta=FROZEN, hash=1.0, snd=0.25))
    changed = np.flatnonzero(out.w_snd != head.w_snd)
    assert set(changed.tolist()) <= {5, 7, 300} and changed.size > 0
    assert out.w_snd[300] < head.w_snd[300]  # a ham example pulls its SND weights down
    assert np.array_equal(out.w_embed, head.w_embed) and np.array_equal(out.w_meta, head.w_meta)
    z0 = [head.z(e.emb, e.meta, e.idx, e.val, e.snd_idx, e.snd_val) for e in examples]
    z1 = [out.z(e.emb, e.meta, e.idx, e.val, e.snd_idx, e.snd_val) for e in examples]
    assert z1[0] < z0[0] and z1[2] < z0[2]


def test_snd_and_hash_blocks_are_separate_with_their_own_priors():
    head = replace(empty_head(), bias=0.0)
    rng = np.random.default_rng(4)
    # One example whose n-gram and SND vectors are the same single bucket: at the optimum λ·Δw equals the same
    # gradient for both blocks, so the steps are inversely proportional to the priors.
    e = example(rng, 1, [123], [1.0], 1, hash_idx=[123])
    out = personalize(head, [e], np.ones(1), Priors(embed=FROZEN, meta=FROZEN, hash=1.0, snd=4.0))
    dh, ds = float(out.w_hash[123]), float(out.w_snd[123])
    assert dh > 0 and ds > 0
    assert dh * 1.0 == pytest.approx(ds * 4.0, rel=1e-5)


def test_frozen_snd_prior_pins_the_block_and_sender_types_never_move():
    head = small_head()
    rng = np.random.default_rng(5)
    examples = [example(rng, i % 2, [5, 7, 300], [0.5, 0.5, 0.7071], i) for i in range(6)]
    weights = example_weights(examples, Weighting(cluster_cos=2.0))
    out = personalize(head, examples, weights, Priors(snd=FROZEN))
    assert np.array_equal(out.w_snd, head.w_snd)
    assert np.array_equal(out.w_meta[SENDER_SLICE], head.w_meta[SENDER_SLICE])
    assert not np.array_equal(out.w_hash, head.w_hash)
