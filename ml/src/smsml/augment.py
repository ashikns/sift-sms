"""Training-time text augmentation with visible obfuscation tricks.

Invisible characters, homoglyphs and fullwidth/math letters are canonicalized by norm_v1, so augmentation only
covers what survives normalization: spaced letters, inserted punctuation, leetspeak, typos, case games, URL and
amount variations.
"""

from __future__ import annotations

import random
import re

LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "b": "8", "g": "9"}
TLDS = ["top", "xyz", "icu", "live", "shop", "info", "cc", "vip", "click", "site", "online", "com", "net"]
SHORTENERS = ["bit.ly", "tinyurl.com", "rb.gy", "t.ly", "is.gd", "cutt.ly"]
URL_RE = re.compile(r"(https?://)?([a-z0-9-]+\.)+[a-z]{2,}(/\S*)?", re.I)
MONEY_RE = re.compile(r"([$£€])\s?\d[\d,]*(\.\d{2})?")


def _spaced(w: str, r: random.Random) -> str:
    return " ".join(w)


def _punct(w: str, r: random.Random) -> str:
    sep = r.choice([".", "-", "*", "_"])
    return sep.join(w)


def _leet(w: str, r: random.Random) -> str:
    return "".join(LEET.get(c.lower(), c) if r.random() < 0.5 else c for c in w)


def _typo(w: str, r: random.Random) -> str:
    if len(w) < 4:
        return w
    i = r.randrange(1, len(w) - 1)
    op = r.randrange(3)
    if op == 0:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op == 1:
        return w[:i] + w[i + 1:]
    return w[:i] + w[i] + w[i:]


def _case(w: str, r: random.Random) -> str:
    return "".join(c.upper() if r.random() < 0.5 else c.lower() for c in w)


WORD_OPS = [_spaced, _punct, _leet, _typo, _case]


def _url_variant(m: re.Match, r: random.Random) -> str:
    choice = r.randrange(4)
    if choice == 0:
        return f"https://{r.choice(SHORTENERS)}/{r.randrange(16 ** 5):05x}"
    host = m.group(0).split("/")[2] if m.group(0).lower().startswith("http") else m.group(0).split("/")[0]
    labels = host.split(".")
    if choice == 1 and len(labels) >= 2:
        labels[-1] = r.choice(TLDS)
        return "https://" + ".".join(labels) + f"/{r.randrange(10 ** 4)}"
    if choice == 2:
        return host + "/" + "".join(r.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(r.randrange(3, 9)))
    return m.group(0)


def _money(m: re.Match, r: random.Random) -> str:
    amount = r.choice([r.randrange(1, 10), r.randrange(10, 100), r.randrange(100, 5000)])
    cents = f".{r.randrange(100):02d}" if r.random() < 0.4 else ""
    return f"{m.group(1)}{amount}{cents}"


def augment(text: str, r: random.Random, max_ops: int = 2) -> str:
    out = text
    if r.random() < 0.3:
        out = URL_RE.sub(lambda m: _url_variant(m, r), out, count=1)
    if r.random() < 0.3:
        out = MONEY_RE.sub(lambda m: _money(m, r), out, count=1)
    words = out.split(" ")
    eligible = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    for _ in range(r.randrange(1, max_ops + 1)):
        if not eligible:
            break
        i = r.choice(eligible)
        words[i] = r.choice(WORD_OPS)(words[i], r)
    return " ".join(words)
