"""norm_v1 — the reference normalizer and URL scanner (SPEC.md). Mirrored by spamcore's TextNormalizer.

Every character class is an explicit code-point list; nothing here uses regular expressions.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .resources import Resources, default_resources

MAX_CPS = 2000

EMOJI = ((0x1F000, 0x1FAFF), (0x2600, 0x27BF), (0x2300, 0x23FF), (0x2B00, 0x2BFF))
INVISIBLE = (
    (0x00AD, 0x00AD), (0x034F, 0x034F), (0x061C, 0x061C), (0x115F, 0x1160), (0x17B4, 0x17B5), (0x180B, 0x180F),
    (0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x206F), (0x3164, 0x3164), (0xFE00, 0xFE0F), (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0), (0x1BCA0, 0x1BCA3), (0xE0000, 0xE007F), (0xE0100, 0xE01EF),
)
# Removed like the rest, but not counted: they appear in ordinary emoji sequences.
INVISIBLE_UNCOUNTED = ((0x200D, 0x200D), (0xFE0E, 0xFE0F), (0xE0020, 0xE007F))
MARKS = ((0x0300, 0x036F), (0x1AB0, 0x1AFF), (0x1DC0, 0x1DFF), (0x20D0, 0x20FF), (0xFE20, 0xFE2F))
PUNCT = {
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'", 0x2032: "'", 0x02BC: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"', 0x2033: '"',
    0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-", 0x2015: "-", 0x2212: "-",
    0x2022: "*", 0x2043: "*",
}
CURRENCY = frozenset((0x00A2, 0x00A3, 0x00A5, 0x20AC, 0x20B9))

LEAD_STRIP = frozenset("([{<\"'")
TRAIL_STRIP = frozenset(".,;:!?)]}>\"'")
HOST_STOP = "/?#"
LABEL_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")


def in_ranges(cp: int, ranges) -> bool:
    for lo, hi in ranges:
        if lo <= cp <= hi:
            return True
    return False


def is_ascii_alnum_str(s: str) -> bool:
    if not s:
        return False
    for c in s:
        o = ord(c)
        if not (0x30 <= o <= 0x39 or 0x41 <= o <= 0x5A or 0x61 <= o <= 0x7A):
            return False
    return True


def ascii_lower(s: str) -> str:
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in s)


def is_digits(s: str) -> bool:
    return bool(s) and all("0" <= c <= "9" for c in s)


def is_letters(s: str) -> bool:
    return bool(s) and all("a" <= c <= "z" for c in s)


@dataclass(frozen=True)
class Url:
    host: str
    tld: str
    domain: str
    ip_host: bool
    punycode: bool
    at_sign: bool
    shortener: bool
    risky_tld: bool
    brand_lookalike: bool
    official: bool


@dataclass(frozen=True)
class Normalized:
    clean: str
    enc: str
    ng: str
    urls: tuple[Url, ...]
    url_tokens: frozenset[int]  # indices of `clean` space-tokens that are URLs
    cp_count: int
    emoji_count: int
    invisible_count: int
    lookalike_count: int
    dropped_count: int
    non_gsm7: bool


def _is_ipv4(host: str) -> bool:
    parts = host.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not (1 <= len(p) <= 3 and is_digits(p)) or int(p) > 255:
            return False
    return True


def _authority_end(s: str) -> int:
    end = len(s)
    for ch in HOST_STOP:
        k = s.find(ch)
        if k != -1 and k < end:
            end = k
    return end


def parse_url(core: str, res: Resources) -> Url | None:
    low = ascii_lower(core)
    if low.startswith("https://"):
        rest, schemed = low[8:], True
    elif low.startswith("http://"):
        rest, schemed = low[7:], True
    else:
        rest, schemed = low, low.startswith("www.")
    end = _authority_end(rest)
    authority = rest[:end]
    at_sign = False
    if "@" in authority:
        if not schemed:
            return None  # a bare "name@host" is an email address, not a URL
        authority = authority[authority.rfind("@") + 1:]
        at_sign = True
    colon = authority.find(":")
    host = authority if colon == -1 else authority[:colon]
    labels = host.split(".")
    if len(labels) < 2 or any(not lab for lab in labels):
        return None
    ip = _is_ipv4(host)
    if not schemed and not ip:
        if any(c not in LABEL_CHARS for lab in labels for c in lab):
            return None
        tld = labels[-1]
        if not is_letters(tld) or tld not in res.tlds or is_digits(labels[0]):
            return None
        if tld in res.tld_stoplist:
            second = labels[-2]
            evidence = (
                "/" in rest[end:]
                or len(labels) >= 3
                or "-" in second
                or any("0" <= c <= "9" for c in second)
            )
            if not evidence:
                return None
    if host.startswith("www."):
        host = host[4:]
    hl = host.split(".")
    if ip:
        tld, domain = "", host
    else:
        tld = hl[-1]
        if len(hl) >= 3 and (hl[-2] + "." + hl[-1]) in res.psl:
            domain = ".".join(hl[-3:])
        else:
            domain = ".".join(hl[-2:]) if len(hl) >= 2 else host
    words = set(host.replace("-", ".").split("."))
    lookalike = False
    for keyword, officials, substring in res.brands:
        matches = (substring and keyword in host) or keyword in words
        if matches and domain not in officials:
            lookalike = True
            break
    return Url(
        host=host,
        tld=tld,
        domain=domain,
        ip_host=ip,
        punycode=any(lab.startswith("xn--") for lab in hl),
        at_sign=at_sign,
        shortener=domain in res.shorteners,
        risky_tld=bool(tld) and tld in res.risky_tlds,
        brand_lookalike=lookalike,
        official=domain in res.official_domains,
    )


def _code_points(raw: str) -> list[int]:
    cps = [0xFFFD if 0xD800 <= ord(c) <= 0xDFFF else ord(c) for c in raw]
    return cps[:MAX_CPS]


def normalize(raw: str, res: Resources | None = None) -> Normalized:
    res = res or default_resources()
    # 1-2. code points and raw stats
    cps = _code_points(raw)
    cp_count = len(cps)
    non_gsm7 = any(cp not in res.gsm7 for cp in cps)
    emoji_count = sum(1 for cp in cps if in_ranges(cp, EMOJI))
    # 3. invisible characters
    kept, invisible_count = [], 0
    for cp in cps:
        if in_ranges(cp, INVISIBLE):
            if not in_ranges(cp, INVISIBLE_UNCOUNTED):
                invisible_count += 1
        else:
            kept.append(cp)
    # 4. lookalike census (before NFKC)
    lookalike_count = 0
    for cp in kept:
        if cp < 0x80:
            continue
        ch = chr(cp)
        f = unicodedata.normalize("NFKC", ch)
        if f != ch and is_ascii_alnum_str(f):
            lookalike_count += 1
        elif cp in res.confusables:
            lookalike_count += 1
    # 5. NFKC
    text = unicodedata.normalize("NFKC", "".join(chr(cp) for cp in kept))
    # 6. confusables
    text = "".join(res.confusables.get(ord(c), c) for c in text)
    # 7. strip marks
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not in_ranges(ord(c), MARKS))
    text = unicodedata.normalize("NFC", text)
    # 8. punctuation map
    text = "".join(PUNCT.get(ord(c), c) for c in text)
    # 9. restrict
    out, dropped = [], 0
    for c in text:
        o = ord(c)
        if o <= 0x1F or 0x7F <= o <= 0x9F:
            out.append(" ")
        elif in_ranges(o, EMOJI):
            out.append(" ")
        elif 0x20 <= o <= 0x7E or o in CURRENCY:
            out.append(c)
        else:
            out.append(" ")
            if o != 0xFFFD:  # replacement chars come from decoding damage, not from the sender
                dropped += 1
    # 10. collapse + trim
    clean = " ".join(t for t in "".join(out).split(" ") if t)
    # 11-12. URL scan, enc
    tokens = clean.split(" ") if clean else []
    urls, url_tokens, enc_tokens = [], set(), []
    for idx, tok in enumerate(tokens):
        i, j = 0, len(tok)
        while i < j and tok[i] in LEAD_STRIP:
            i += 1
        while j > i and tok[j - 1] in TRAIL_STRIP:
            j -= 1
        core = tok[i:j]
        url = parse_url(core, res) if core else None
        if url is not None:
            urls.append(url)
            url_tokens.add(idx)
            enc_tokens.append(tok[:i] + url.host + tok[j:])
        else:
            enc_tokens.append(tok)
    enc = " ".join(enc_tokens)
    # 13. n-gram view
    ng = "".join("0" if "0" <= c <= "9" else (chr(ord(c) + 32) if "A" <= c <= "Z" else c) for c in enc)
    return Normalized(
        clean=clean,
        enc=enc,
        ng=ng,
        urls=tuple(urls),
        url_tokens=frozenset(url_tokens),
        cp_count=cp_count,
        emoji_count=emoji_count,
        invisible_count=invisible_count,
        lookalike_count=lookalike_count,
        dropped_count=dropped,
        non_gsm7=non_gsm7,
    )
