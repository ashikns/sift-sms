import os
import random

import mmh3
import numpy as np
import pytest

from smsml.features import META_NAMES, feature_strings, metadata, sparse_cosine, sparse_features, word_tokens
from smsml.hashing import murmur3_32
from smsml.normalize import normalize


def meta(text):
    return dict(zip(META_NAMES, metadata(normalize(text)).tolist()))


def test_murmur_reference_vectors():
    assert murmur3_32(b"foo", 0) == 4138058784
    assert murmur3_32(b"foo", 42) == (-1322301282) & 0xFFFFFFFF
    assert murmur3_32(b"", 0) == 0
    assert murmur3_32(b"", 1) == 0x514E28B7


def test_murmur_matches_mmh3_on_random_bytes():
    rng = random.Random(7)
    for _ in range(2000):
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(40)))
        seed = rng.randrange(1 << 32)
        assert murmur3_32(data, seed) == mmh3.hash(data, seed, signed=False)


def test_word_tokens_split_currency():
    assert word_tokens("win $000 now, £00!") == ["win", "$", "000", "now", "£", "00"]


def test_feature_families():
    chars, words, urls = feature_strings(normalize("Hi bit.ly/x"))
    assert "c: hi" in chars and "c:hi " in chars and "c: hi b" in chars
    assert "w:hi" in words and "b:hi bit" in words
    assert urls == ["h:bit.ly", "t:ly", "u:bit.ly"]
    assert chars == sorted(chars)


def test_metadata_examples():
    m = meta("Call (555) 010-4477 now")
    assert m["phone_number"] == 1.0
    assert meta("Text 55501 for info")["phone_number"] == 0.0
    assert meta("Email jane.doe@example.com")["email_address"] == 1.0
    assert meta("You won $1,000 today")["currency_amount"] == 1.0
    assert meta("worth 500 dollars")["currency_amount"] == 1.0
    assert meta("Reply STOP to opt out")["optout_stop"] == 1.0
    assert meta("Your verification code is 482913")["otp_pattern"] == 1.0
    assert meta("Your 2FA: 4829")["otp_pattern"] == 1.0
    assert meta("Meet at 2026 Main St")["otp_pattern"] == 0.0
    assert meta("WIN NOW!!!")["upper_ratio"] == 1.0
    assert meta("WIN NOW!!!")["excl_count"] == pytest.approx(0.6)
    assert meta("x" * 1000)["len_log"] == pytest.approx(1.0, abs=2e-4)
    assert all(meta("anything")[k] == 0.0 for k in META_NAMES[21:])
    assert meta("Delivered: https://www.amazon.com/gp/css")["url_official"] == 1.0
    assert meta("Delivered: https://amazon-track.top/x")["url_official"] == 0.0


def test_sparse_features_are_family_normalized():
    idf = np.ones(1 << 18, dtype=np.float32)
    idx, val = sparse_features(normalize("Claim your prize at https://prize-win.top now"), idf)
    assert np.all(np.diff(idx) > 0)
    assert 1.0 < float(np.linalg.norm(val.astype(np.float64))) <= 3 ** 0.5 + 1e-6
    assert sparse_cosine(idx, val, idx, val) == pytest.approx(1.0)


def test_near_duplicates_score_high_and_unrelated_low():
    idf = np.ones(1 << 18, dtype=np.float32)
    a = sparse_features(normalize("USPS: your package is on hold. Update address: https://usps-hold.top/a1"), idf)
    b = sparse_features(normalize("USPS: your parcel is on hold. Update address: https://usps-hold.top/b7"), idf)
    c = sparse_features(normalize("Are we still on for dinner at 7?"), idf)
    assert sparse_cosine(*a, *b) > 0.8
    assert sparse_cosine(*a, *c) < 0.2


@pytest.mark.skipif(not os.environ.get("SIFT_SLOW"), reason="set SIFT_SLOW=1")
def test_sparse_cosine_symmetry_random():
    idf = np.ones(1 << 18, dtype=np.float32)
    a = sparse_features(normalize("alpha beta gamma"), idf)
    b = sparse_features(normalize("beta gamma delta"), idf)
    assert sparse_cosine(*a, *b) == pytest.approx(sparse_cosine(*b, *a))
