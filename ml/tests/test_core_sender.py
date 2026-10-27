import math

import numpy as np
import pytest

from smsml.features import feature_strings, sender_id, snd_features, snd_strings, sparse_cosine, sparse_features
from smsml.hashing import hash_feature
from smsml.normalize import normalize
from smsml.pipeline import featurize, sender_key, senders_of

ALERT = "INR 2,500.00 spent using ICICI Bank Card XX1234 at AMAZON. If not you, call 18002662. icici.com/x"


@pytest.mark.parametrize("key,sid", [
    ("AX-ICICIT-S", "ICICIT"),
    ("JD-ICICIT-T", "ICICIT"),
    ("VM-HDFCBK", "HDFCBK"),
    ("AX-ICI-S", "ICI"),
    ("AX-ICICIBNK-S", "ICICIBNK"),
    ("AX-123456-S", "123456"),
    ("ICICIT", "ICICIT"),
    ("+915550100123", "+915550100123"),
    ("57575", "57575"),
    ("jane.doe@example.com", "jane.doe@example.com"),
    ("AMAZON ALERTS", "AMAZON ALERTS"),
    ("AX-ICICIBANK-S", "AX-ICICIBANK-S"),  # ID longer than 8
    ("AX-IC-S", "AX-IC-S"),  # ID shorter than 3
    ("A1-ICICIT-S", "A1-ICICIT-S"),
    ("AXX-ICICIT-S", "AXX-ICICIT-S"),
    ("AX-ICICIT-SS", "AX-ICICIT-SS"),
    ("AX-ICICIT-1", "AX-ICICIT-1"),
    ("AX-ICICIT-", "AX-ICICIT-"),
    ("AX--S", "AX--S"),
    ("ax-icicit-s", "ax-icicit-s"),
    ("AX-ICICIT-S-T", "AX-ICICIT-S-T"),
    ("AX-ICİCIT-S", "AX-ICİCIT-S"),
    ("AX-ICICIT-S ", "AX-ICICIT-S "),
    ("AMAZ\ud800N \U0001F4E6", "AMAZ\ufffdN \U0001F4E6"),
    ("unknown", None),
    ("", None),
    (None, None),
])
def test_sender_id(key, sid):
    assert sender_id(key) == sid


def test_snd_strings_pair_the_sid_with_words_and_link_domains():
    n = normalize("Card spent at icici.com/x")
    _, words, urls = feature_strings(n)
    strings = snd_strings("ICICIT", n)
    assert strings == ["s:ICICIT|" + f for f in words + [u for u in urls if u.startswith("u:")]]
    assert "s:ICICIT|w:spent" in strings and "s:ICICIT|b:card spent" in strings and "s:ICICIT|u:icici.com" in strings
    assert not any(f.startswith(("s:ICICIT|c:", "s:ICICIT|h:", "s:ICICIT|t:")) for f in strings)
    assert snd_strings(None, n) == []


def test_snd_vector_is_binary_and_normalized():
    n = normalize(ALERT)
    idx, val = snd_features("ICICIT", n)
    assert idx.dtype == np.int32 and val.dtype == np.float32
    assert np.all(np.diff(idx) > 0)
    assert float(np.linalg.norm(val.astype(np.float64))) == pytest.approx(1.0, abs=1e-6)
    sums: dict[int, int] = {}
    for f in snd_strings("ICICIT", n):
        k, sign = hash_feature(f)
        sums[k] = sums.get(k, 0) + sign
    nonzero = {k: v for k, v in sums.items() if v != 0}
    norm = math.sqrt(sum(v * v for v in nonzero.values()))
    assert idx.tolist() == sorted(nonzero)
    assert val.tolist() == [np.float32(nonzero[k] / norm) for k in sorted(nonzero)]
    empty = snd_features(None, n)
    assert empty[0].size == 0 and empty[1].size == 0


def test_snd_vector_depends_on_the_sid_and_not_the_route():
    n = normalize(ALERT)
    a = snd_features(sender_id("AX-ICICIT-S"), n)
    b = snd_features(sender_id("JD-ICICIT-T"), n)
    c = snd_features(sender_id("VM-HDFCBK-S"), n)
    assert a[0].tolist() == b[0].tolist() and a[1].tolist() == b[1].tolist()
    assert sparse_cosine(*a, *c) < 0.2


def test_sender_key_mirrors_the_app_on_an_indian_sim():
    assert sender_key("ax-icicit-s ") == "AX-ICICIT-S"
    assert sender_key("  Amazon \t Alerts ") == "AMAZON ALERTS"
    assert sender_key("Jane.Doe@Example.COM") == "jane.doe@example.com"
    assert sender_key("+91 55501 00123") == "+915550100123"
    assert sender_key("+1 (555) 010-0199") == "+15550100199"
    assert sender_key("9555010012") == "+919555010012"
    assert sender_key("09555010012") == "+919555010012"
    assert sender_key("919555010012") == "+919555010012"
    assert sender_key("57575") == "57575"
    assert sender_key("123") == "123"
    assert sender_key("5550100123") == "5550100123"  # 10 digits not starting 6-9: kept, as when the formatter gives up
    assert sender_key("1234567") == "1234567"
    assert sender_key("") == sender_key(None) == sender_key("   ") == "unknown"
    assert sender_id(sender_key(None)) is None


def test_senders_of_reads_the_column_when_present():
    import pandas as pd

    assert senders_of(pd.DataFrame({"text": ["a", "b"]})) == [None, None]
    df = pd.DataFrame({"text": ["a", "b"], "sender": pd.array(["AX-ICICIT-S", pd.NA], dtype="string")})
    assert senders_of(df) == ["AX-ICICIT-S", None]


def test_featurize_with_senders():
    idf = np.ones(1 << 18, dtype=np.float32)
    texts = [ALERT, ALERT, "Are we still on for dinner?"]
    X, S, meta = featurize(texts, idf, ["AX-ICICIT-S", "jd-icicit-t", None])
    X0, S0, meta0 = featurize(texts, idf)
    assert (X != X0).nnz == 0 and np.array_equal(meta, meta0)  # n-grams and metadata ignore the sender
    assert not meta[:, 21:].any()  # the sender-type features stay 0 offline, even with a sender
    assert S0.nnz == 0 and S[2].nnz == 0
    n = normalize(ALERT)
    want_idx, want_val = snd_features("ICICIT", n)
    for i in (0, 1):  # both routes of one DLT sender give the same SND vector
        assert S[i].indices.tolist() == want_idx.tolist()
        assert S[i].data.tolist() == want_val.tolist()
    x_idx, x_val = sparse_features(n, idf)
    assert X[0].indices.tolist() == x_idx.tolist()
    with pytest.raises(ValueError):
        featurize(texts, idf, [None])
