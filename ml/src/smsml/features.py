"""feat_v1 — hashed n-gram features and the 27 metadata values (SPEC.md). Mirrored by spamcore's features package."""

from __future__ import annotations

import math

import numpy as np

from .hashing import BUCKETS, hash_feature
from .normalize import CURRENCY, Normalized

META_NAMES = (
    "len_log", "url_count", "url_shortener", "url_ip_host", "url_risky_tld", "url_punycode_or_at",
    "url_brand_lookalike", "phone_number", "email_address", "currency_amount", "optout_stop", "otp_pattern",
    "upper_ratio", "digit_ratio", "excl_count", "emoji_count", "invisible", "lookalike", "dropped_ratio",
    "non_gsm7", "url_official", "snd_short_code", "snd_long_code_nanp", "snd_alphanumeric", "snd_email",
    "snd_international", "snd_first_contact",
)
N_META = len(META_NAMES)
SENDER_SLICE = slice(21, 27)

CURRENCY_TOKENS = frozenset(chr(c) for c in CURRENCY) | {"$"}
PHONE_CHARS = frozenset("0123456789+-.() ")
PHONE_START = frozenset("0123456789+(")
OPTOUT_VERBS = ("reply", "text", "txt", "send", "opt")
OTP_WORDS = ("code", "otp", "passcode", "verification", "pin", "0fa")  # "2FA" reads "0fa" after digit folding
LN_1001 = math.log(1001.0)


def word_tokens(ng: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for c in ng:
        if "a" <= c <= "z" or "0" <= c <= "9":
            cur.append(c)
            continue
        if cur:
            out.append("".join(cur))
            cur = []
        if c in CURRENCY_TOKENS:
            out.append(c)
    if cur:
        out.append("".join(cur))
    return out


def feature_strings(n: Normalized) -> tuple[list[str], list[str], list[str]]:
    """(char, word, url) families, each deduplicated and sorted by code point."""
    s = " " + n.ng + " "
    chars = {"c:" + s[i:i + k] for k in (3, 4, 5) for i in range(len(s) - k + 1)}
    toks = word_tokens(n.ng)
    words = {"w:" + t for t in toks}
    words.update("b:" + toks[i] + " " + toks[i + 1] for i in range(len(toks) - 1))
    urls: set[str] = set()
    for u in n.urls:
        urls.add("u:" + u.domain)
        urls.add("h:" + u.host)
        if u.tld:
            urls.add("t:" + u.tld)
    return sorted(chars), sorted(words), sorted(urls)


def _family_vector(strings: list[str], idf: np.ndarray) -> dict[int, float]:
    v: dict[int, float] = {}
    for f in strings:
        idx, sign = hash_feature(f)
        v[idx] = v.get(idx, 0.0) + sign * float(idf[idx])
    v = {k: x for k, x in v.items() if x != 0.0}
    norm = math.sqrt(sum(x * x for _, x in sorted(v.items())))
    if norm > 0.0:
        v = {k: x / norm for k, x in v.items()}
    return v


def sparse_features(n: Normalized, idf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted int32 indices and float32 values (sum of the three L2-normalized family vectors)."""
    total: dict[int, float] = {}
    for family in feature_strings(n):
        for k, x in sorted(_family_vector(family, idf).items()):
            total[k] = total.get(k, 0.0) + x
    items = sorted((k, x) for k, x in total.items() if x != 0.0)
    idx = np.fromiter((k for k, _ in items), dtype=np.int32, count=len(items))
    val = np.fromiter((x for _, x in items), dtype=np.float64, count=len(items)).astype(np.float32)
    return idx, val


def bucket_set(n: Normalized) -> set[int]:
    """Buckets touched by any feature string, for document frequencies."""
    out: set[int] = set()
    for family in feature_strings(n):
        for f in family:
            out.add(hash_feature(f)[0])
    return out


def idf_from_df(df: np.ndarray, n_docs: int) -> np.ndarray:
    return (np.log((1.0 + n_docs) / (1.0 + df.astype(np.float64))) + 1.0).astype(np.float32)


def phone_number(clean: str) -> bool:
    i, n = 0, len(clean)
    while i < n:
        if clean[i] in PHONE_START:
            j = i
            digits = 0
            while j < n and clean[j] in PHONE_CHARS:
                if "0" <= clean[j] <= "9":
                    digits += 1
                j += 1
            if 10 <= digits <= 15:
                return True
            i = j
        else:
            i += 1
    return False


def email_address(clean: str, url_tokens: frozenset[int]) -> bool:
    for idx, tok in enumerate(clean.split(" ") if clean else []):
        if idx in url_tokens:
            continue
        a = tok.find("@")
        if a >= 1 and "." in tok[a + 1:]:
            return True
    return False


def currency_amount(ng: str, toks: list[str]) -> bool:
    n = len(ng)
    for i, c in enumerate(ng):
        if c not in CURRENCY_TOKENS:
            continue
        if i + 1 < n and ng[i + 1] == "0":
            return True
        if i + 2 < n and ng[i + 1] == " " and ng[i + 2] == "0":
            return True
        if i >= 1 and ng[i - 1] == "0":
            return True
        if i >= 2 and ng[i - 1] == " " and ng[i - 2] == "0":
            return True
    for i in range(1, len(toks)):
        if toks[i] in ("usd", "dollars", "bucks") and all(ch == "0" for ch in toks[i - 1]):
            return True
    return False


def metadata(n: Normalized) -> np.ndarray:
    """The metadata values; the sender features (21-26) stay 0 (public datasets have no sender info)."""
    toks = word_tokens(n.ng)
    words = set(toks)
    m = [0.0] * N_META
    m[0] = min(math.log(1.0 + n.cp_count) / LN_1001, 1.0)
    m[1] = min(len(n.urls), 3) / 3.0
    m[2] = float(any(u.shortener for u in n.urls))
    m[3] = float(any(u.ip_host for u in n.urls))
    m[4] = float(any(u.risky_tld for u in n.urls))
    m[5] = float(any(u.punycode or u.at_sign for u in n.urls))
    m[6] = float(any(u.brand_lookalike for u in n.urls))
    m[7] = float(phone_number(n.clean))
    m[8] = float(email_address(n.clean, n.url_tokens))
    m[9] = float(currency_amount(n.ng, toks))
    m[10] = float("stop" in words and any(v in words for v in OPTOUT_VERBS))
    has_code = any(4 <= len(t) <= 8 and all(ch == "0" for ch in t) for t in toks)
    has_word = any(w in words for w in OTP_WORDS) or any(
        toks[i] == "one" and toks[i + 1] == "time" for i in range(len(toks) - 1))
    m[11] = float(has_code and has_word)
    letters = sum(1 for c in n.clean if "a" <= c <= "z" or "A" <= c <= "Z")
    upper = sum(1 for c in n.clean if "A" <= c <= "Z")
    m[12] = upper / max(1, letters)
    non_space = sum(1 for c in n.clean if c != " ")
    m[13] = sum(1 for c in n.clean if "0" <= c <= "9") / max(1, non_space)
    m[14] = min(n.clean.count("!"), 5) / 5.0
    m[15] = min(n.emoji_count, 5) / 5.0
    m[16] = min(n.invisible_count, 3) / 3.0
    m[17] = min(n.lookalike_count, 4) / 4.0
    m[18] = min(n.dropped_count / max(1, n.cp_count), 1.0)
    m[19] = float(n.non_gsm7)
    m[20] = float(any(u.official for u in n.urls))
    return np.asarray(m, dtype=np.float64).astype(np.float32)


def sparse_cosine(ai: np.ndarray, av: np.ndarray, bi: np.ndarray, bv: np.ndarray) -> float:
    """Cosine of two sorted sparse vectors, accumulated in double (SPEC near-duplicate similarity)."""
    na = math.sqrt(float(np.dot(av.astype(np.float64), av.astype(np.float64))))
    nb = math.sqrt(float(np.dot(bv.astype(np.float64), bv.astype(np.float64))))
    if na == 0.0 or nb == 0.0:
        return 0.0
    i = j = 0
    dot = 0.0
    while i < len(ai) and j < len(bi):
        if ai[i] == bi[j]:
            dot += float(av[i]) * float(bv[j])
            i += 1
            j += 1
        elif ai[i] < bi[j]:
            i += 1
        else:
            j += 1
    return dot / (na * nb)


__all__ = [
    "META_NAMES", "N_META", "SENDER_SLICE", "BUCKETS", "word_tokens",
    "feature_strings", "sparse_features", "bucket_set", "idf_from_df", "metadata", "sparse_cosine",
]
