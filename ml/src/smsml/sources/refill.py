"""Deterministic refill of anonymization placeholders with synthetic values.

* IMC'25 replaced PII with Presidio entity tags such as `<URL>`, `<PHONE_NUMBER>`, `<DATE_TIME>`,
  `<NAMED_ENTITY>`, `<LOCATION>`, `<US_DRIVER_LICENSE>` (in practice: reference/order codes), ...
* The UCI collection (and the copies of it inside Mishra & Soni) replaced numbers, times, URLs and emails with
  `&lt;#&gt;`, `&lt;DECIMAL&gt;`, `&lt;TIME&gt;`, `&lt;URL&gt;`, `&lt;EMAIL&gt;` — only in ham, so leaving them
  would hand the model a label-leaking artifact.

Refilled values are synthetic: common US first names, fictional 555-01xx numbers, TEST-NET IP addresses, and
hosts sampled from real NCSU smishing URLs (spam-side `<URL>`s). Each message's RNG is seeded from
(seed, original text), so identical messages get identical refills regardless of row order or source — that
keeps cross-source deduplication exact.
"""

from __future__ import annotations

import hashlib
import random
import string

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "David", "Elizabeth", "William",
    "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Chris", "Karen", "Daniel", "Lisa",
    "Matthew", "Nancy", "Anthony", "Betty", "Mark", "Sandra", "Steven", "Ashley", "Paul", "Emily", "Andrew",
    "Michelle", "Josh", "Amanda", "Kevin", "Melissa", "Brian", "Stephanie", "Maria", "Jose", "Luis", "Ana",
    "Priya", "Wei", "Aisha", "Omar", "Jordan", "Taylor",
]
BRANDS = [
    "Amazon", "PayPal", "USPS", "Netflix", "Apple", "Chase", "Wells Fargo", "Bank of America", "FedEx", "UPS",
    "DHL", "Walmart", "Costco", "Verizon", "AT&T", "T-Mobile", "Coinbase", "Venmo", "Cash App", "Microsoft",
]
CITIES = [
    "Houston", "Dallas", "Austin", "Denver", "Seattle", "Chicago", "Phoenix", "Atlanta", "Miami", "Boston",
    "Portland", "Nashville", "San Diego", "Columbus", "Charlotte", "Tampa", "Orlando", "Las Vegas", "New York",
    "Los Angeles",
]
NATIONALITIES = ["American", "Canadian", "British", "Indian", "Australian", "Mexican", "Filipino", "Nigerian"]
DATE_PHRASES = [
    "today", "tomorrow", "24 hours", "48 hours", "12 hours", "2 days", "this week", "tonight", "11:59 PM",
    "10/28/2023", "11/04/2023", "Oct 28", "Nov 4", "March 12, 2023", "2023-06-15", "01/12", "15-05-2023",
    "Monday", "Friday", "the next 24hrs", "30 minutes", "midnight",
]
AREA_CODES = ["212", "213", "312", "404", "415", "469", "512", "602", "617", "646", "702", "713", "720", "786",
              "813", "832", "917", "929"]
SAFE_FALLBACK_HOSTS = ["secure-update-center.com", "account-verify-portal.net", "parcel-redelivery.info",
                       "toll-payment-service.top", "billing-support-desk.xyz"]
BENIGN_HOSTS = ["www.friendsplace.co.uk", "www.funnyclips.net", "www.mytunes-music.com", "www.photo-share.org"]
FREE_MAIL = ["gmail.com", "outlook.com", "yahoo.com", "icloud.com", "aol.com"]

PERSON_CUES = {"dear", "hi", "hello", "hey", "mr", "mrs", "ms", "miss", "dr", "sir", "madam", "is", "am", "im",
               "i'm", "name", "regards", "thanks", "from"}


def rng_for(seed: int, text: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{text}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def _alnum(rng: random.Random, n: int, upper: bool = True) -> str:
    pool = (string.ascii_uppercase if upper else string.ascii_lowercase) + string.digits
    return "".join(rng.choice(pool) for _ in range(n))


def fictional_phone(rng: random.Random) -> str:
    area = rng.choice(AREA_CODES)
    tail = f"01{rng.randint(0, 99):02d}"
    return rng.choice([f"({area}) 555-{tail}", f"{area}-555-{tail}", f"+1{area}555{tail}", f"{area}555{tail}",
                       f"1-{area}-555-{tail}"])


def _preceding_word(text: str, idx: int) -> str:
    j = idx
    while j > 0 and not text[j - 1].isalnum():
        j -= 1
    k = j
    while k > 0 and (text[k - 1].isalnum() or text[k - 1] == "'"):
        k -= 1
    return text[k:j].lower()


def refill_imc(text: str, seed: int, *, hosts: list[str], shortener: str | None = None,
               named_entity: str | None = None) -> str:
    """Replace IMC'25 Presidio tags in `text`. Unknown tags are left untouched."""
    if "<" not in text:
        return text
    rng = rng_for(seed, text)
    out: list[str] = []
    i = 0
    used_shortener = False
    while i < len(text):
        if text[i] == "<":
            j = text.find(">", i + 1)
            tag = text[i + 1:j] if j != -1 else ""
            if j != -1 and tag and all(c.isupper() or c == "_" for c in tag):
                value = _imc_value(tag, rng, text, i, hosts, shortener if not used_shortener else None,
                                   named_entity)
                if value is not None:
                    if tag == "URL" and shortener and not used_shortener:
                        used_shortener = True
                    out.append(value)
                    i = j + 1
                    continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _imc_value(tag: str, rng: random.Random, text: str, pos: int, hosts: list[str], shortener: str | None,
               named_entity: str | None) -> str | None:
    if tag == "URL":
        if shortener:
            return f"https://{shortener}/{_alnum(rng, rng.randint(5, 7), upper=False)}"
        host = rng.choice(hosts or SAFE_FALLBACK_HOSTS)
        path = rng.choice(["", "/" + _alnum(rng, rng.randint(4, 8), upper=False), "/verify", "/login",
                           "/track?id=" + _alnum(rng, 6)])
        scheme = rng.choice(["https://", "http://", ""])
        return scheme + host + path
    if tag == "DATE_TIME":
        return rng.choice(DATE_PHRASES)
    if tag == "PHONE_NUMBER":
        return fictional_phone(rng)
    if tag == "NAMED_ENTITY":
        if _preceding_word(text, pos) in PERSON_CUES:
            return rng.choice(FIRST_NAMES)
        return named_entity or rng.choice(BRANDS)
    if tag == "LOCATION":
        return rng.choice(CITIES)
    if tag == "NRP":
        return rng.choice(NATIONALITIES)
    if tag == "EMAIL_ADDRESS":
        return f"{rng.choice(FIRST_NAMES).lower()}{_digits(rng, rng.randint(2, 4))}@{rng.choice(FREE_MAIL)}"
    if tag == "IP_ADDRESS":
        return f"{rng.choice(['192.0.2', '198.51.100', '203.0.113'])}.{rng.randint(1, 254)}"
    if tag in ("US_DRIVER_LICENSE", "US_PASSPORT", "MEDICAL_LICENSE"):
        return rng.choice([_alnum(rng, rng.randint(8, 12)), _digits(rng, rng.randint(8, 10)),
                           _alnum(rng, 2) + _digits(rng, 7)])
    if tag == "US_BANK_NUMBER":
        return _digits(rng, rng.randint(8, 12))
    if tag in ("US_SSN", "US_ITIN"):
        return f"9{_digits(rng, 2)}-{_digits(rng, 2)}-{_digits(rng, 4)}"
    if tag == "UK_NHS":
        return f"{_digits(rng, 3)} {_digits(rng, 3)} {_digits(rng, 4)}"
    if tag == "CREDIT_CARD":
        return "4111 " + " ".join(_digits(rng, 4) for _ in range(3))
    if tag == "CRYPTO":
        return "bc1q" + _alnum(rng, 38, upper=False)
    return None


UCI_TAGS = ("<#>", "<DECIMAL>", "<TIME>", "<URL>", "<EMAIL>")


def refill_uci(text: str, seed: int) -> str:
    """Replace the UCI corpus's number/time/url/email placeholders (after HTML unescaping)."""
    if "<" not in text:
        return text
    rng = rng_for(seed, text)
    out: list[str] = []
    i = 0
    while i < len(text):
        for tag in UCI_TAGS:
            if text.startswith(tag, i):
                out.append(_uci_value(tag, rng))
                i += len(tag)
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _uci_value(tag: str, rng: random.Random) -> str:
    if tag == "<#>":
        return str(rng.choice([rng.randint(2, 12), rng.randint(10, 60), rng.randint(100, 900)]))
    if tag == "<DECIMAL>":
        return f"{rng.randint(0, 9)}.{rng.randint(1, 99)}"
    if tag == "<TIME>":
        return f"{rng.randint(1, 12)}:{rng.choice(['00', '15', '30', '45'])}"
    if tag == "<URL>":
        return rng.choice(BENIGN_HOSTS)
    return f"{rng.choice(FIRST_NAMES).lower()}{_digits(rng, 2)}@{rng.choice(FREE_MAIL)}"
