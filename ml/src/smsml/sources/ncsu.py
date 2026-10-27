"""NCSU SMS phishing data (Nahapetyan et al., "On SMS Phishing Tactics and Infrastructure", IEEE S&P 2024; MIT).

Files:
* `ncsu/phishing_messages.csv`: 68,029 messages scraped from public SMS gateways (May 2022 – Jun 2023). The header
  names six columns but every row has seven: messageID, objectID, destination number, message, **sender**,
  time (UNIX seconds), error in time (seconds).
* `ncsu/phishing_campaigns.csv`: 35,128 campaigns: campaignID, objectID, number of messages, start/stop
  timestamps, message template (URLs, IDs and codes replaced by `#URL`, `#ID`, `#OTP`, ...) and the campaign's URL
  list. Message counts sum to exactly 68,029, but messages carry no campaign id.

Campaign ids are reconstructed per message, in chronological order: candidates are the campaigns whose URL list
contains one of the message's URLs (or, without a URL hit, the campaigns sharing the most rare template words).
Among candidates within 0.15 word-set Jaccard of the best match, the winner prefers campaigns whose [start, stop]
window contains the message time, then campaigns that still have unassigned messages (their published counts),
then higher Jaccard. This reproduces the exact published message count for ~80% of campaigns. Unmatched messages
get no campaign id. Texts are kept raw (URLs intact, scraped spacing untouched).

The sender is kept as recorded (stripped), with the app's kind (schema.sender_type). Numbers the gateways masked
(`+1818355XXXX`, `***0192`) aren't a real sender, so only their kind is kept, reading the mask as digits.
"""

from __future__ import annotations

import ast
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .lang import is_english
from .schema import finalize, sender_type

MSG_PATH = Path("ncsu/phishing_messages.csv")
CAMP_PATH = Path("ncsu/phishing_campaigns.csv")
EDGE_CHARS = "([{<\"'.,;:!?)]}>"
MAX_WORD_DF = 400
MIN_JACCARD_URL = 0.3
MIN_JACCARD_WORDS = 0.6
NEAR_BEST_JACCARD = 0.15
MASK = "Xx*"  # what the gateways mask a number's digits with
NUMBER_CHARS = frozenset("0123456789+" + MASK)
UNMASK = str.maketrans(MASK, "0" * len(MASK))

csv.field_size_limit(1 << 30)


@dataclass(frozen=True)
class Campaign:
    id: int
    words: frozenset[str]
    start: float
    stop: float
    n_messages: int
    urls: tuple[str, ...]


def norm_url(u: str) -> str:
    low = u.strip().lower()
    for prefix in ("https://", "http://"):
        if low.startswith(prefix):
            low = low[len(prefix):]
            break
    return low.rstrip("/")


def url_host(u: str) -> str | None:
    host = norm_url(u)
    for sep in "/?#":
        host = host.split(sep, 1)[0]
    host = host.split("@")[-1].split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    if "." not in host or not host.isascii() or any(c.isspace() for c in host):
        return None
    labels = host.split(".")
    if any(not label for label in labels) or not all(c.isalnum() or c in "-." for c in host):
        return None
    return host


def words(text: str) -> frozenset[str]:
    """Lower-cased alphabetic words (≥ 2 letters) from tokens without digits."""
    out = set()
    for tok in text.split():
        if any(c.isdigit() for c in tok):
            continue
        for w in "".join(c.lower() if c.isalpha() else " " for c in tok).split():
            if len(w) >= 2:
                out.add(w)
    return frozenset(out)


def _strip_placeholders(template: str) -> str:
    out, i = [], 0
    while i < len(template):
        if template[i] == "#" and i + 1 < len(template) and template[i + 1].isupper():
            j = i + 1
            while j < len(template) and template[j].isupper():
                j += 1
            out.append(" ")
            i = j
        else:
            out.append(template[i])
            i += 1
    return "".join(out)


def load_campaigns(raw_dir: Path) -> dict[int, Campaign]:
    campaigns: dict[int, Campaign] = {}
    with (Path(raw_dir) / CAMP_PATH).open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for rec in reader:
            cid = int(rec[0])
            urls = tuple(ast.literal_eval(rec[6])) if rec[6].strip() else ()
            campaigns[cid] = Campaign(
                id=cid,
                words=words(_strip_placeholders(rec[5])),
                start=float(rec[3]),
                stop=float(rec[4]),
                n_messages=int(rec[2]),
                urls=urls,
            )
    return campaigns


def smishing_hosts(raw_dir: Path) -> list[str]:
    """Sorted unique hostnames from the NCSU campaign URL lists (used to refill IMC'25 `<URL>` tags)."""
    hosts = set()
    for c in load_campaigns(raw_dir).values():
        for u in c.urls:
            h = url_host(u)
            if h:
                hosts.add(h)
    return sorted(hosts)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


class CampaignMatcher:
    def __init__(self, campaigns: dict[int, Campaign]):
        self.campaigns = campaigns
        self.url_index: dict[str, list[int]] = defaultdict(list)
        for c in campaigns.values():
            for u in c.urls:
                self.url_index[norm_url(u)].append(c.id)
        df = Counter(w for c in campaigns.values() for w in c.words)
        self.word_index: dict[str, list[int]] = defaultdict(list)
        for c in campaigns.values():
            for w in c.words:
                if df[w] <= MAX_WORD_DF:
                    self.word_index[w].append(c.id)

    def match(self, text: str, ts: float | None, ts_err: float, remaining: dict[int, int]) -> int | None:
        """Best campaign for one message; decrements `remaining[cid]` for the winner."""
        url_cands: set[int] = set()
        kept_tokens = []
        for tok in text.split():
            t = tok.strip(EDGE_CHARS)
            hits = self.url_index.get(norm_url(t)) if t else None
            if hits:
                url_cands.update(hits)
            else:
                kept_tokens.append(tok)
        msg_words = words(" ".join(kept_tokens))
        if url_cands:
            cands, min_j = url_cands, (0.0 if len(url_cands) == 1 else MIN_JACCARD_URL)
        else:
            votes = Counter(cid for w in msg_words for cid in self.word_index.get(w, ()))
            cands, min_j = {cid for cid, _ in votes.most_common(30)}, MIN_JACCARD_WORDS
        scored = []
        for cid in cands:
            c = self.campaigns[cid]
            in_window = ts is not None and c.start - ts_err - 60 <= ts <= c.stop + ts_err + 60
            scored.append((cid, _jaccard(msg_words, c.words), in_window))
        if not scored:
            return None
        best_j = max(j for _, j, _ in scored)
        if best_j < min_j:
            return None
        near = [s for s in scored if s[1] >= best_j - NEAR_BEST_JACCARD]
        best = max(near, key=lambda s: (s[2], remaining.get(s[0], 0) > 0, s[1], -s[0]))[0]
        remaining[best] = remaining.get(best, 0) - 1
        return best


def sender_fields(raw: str) -> tuple[str | None, str | None]:
    """(sender, sender_type) for a recorded sender; a masked number keeps only its kind."""
    s = raw.strip()
    if set(s) <= NUMBER_CHARS and any(c in MASK for c in s):
        return None, sender_type(s.translate(UNMASK))
    kind = sender_type(s)
    return (s if kind else None), kind


def load(raw_dir: Path, *, min_chars: int = 8, min_ascii_ratio: float = 0.9, min_stopword_hits: int = 1,
         match_campaigns: bool = True) -> pd.DataFrame:
    with (Path(raw_dir) / MSG_PATH).open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        records = list(reader)
    for rec in records:
        if len(rec) != 7:
            raise ValueError(f"unexpected NCSU row width {len(rec)}: {rec[:2]}")

    campaign_of: dict[int, int | None] = {}
    if match_campaigns:
        campaigns = load_campaigns(raw_dir)
        matcher = CampaignMatcher(campaigns)
        remaining = {cid: c.n_messages for cid, c in campaigns.items()}
        order = sorted(range(len(records)), key=lambda i: (float(records[i][5] or 0), i))
        for i in order:
            rec = records[i]
            campaign_of[i] = matcher.match(rec[3].strip(), float(rec[5]) if rec[5] else None,
                                           float(rec[6] or 0), remaining)

    rows = []
    for i, (msg_id, _obj, _dest, message, raw_sender, time_s, _err) in enumerate(records):
        text = message.strip()
        if len(" ".join(text.split())) < min_chars:
            continue
        ts = float(time_s) if time_s else None
        cid = campaign_of.get(i)
        sender, kind = sender_fields(raw_sender)
        rows.append({
            "text": text,
            "label": 1,
            "source": "ncsu",
            "source_id": msg_id,
            "timestamp": pd.Timestamp(ts, unit="s", tz="UTC") if ts is not None else pd.NaT,
            "campaign_id": f"ncsu:{cid}" if cid is not None else None,
            "lang": "en" if is_english(text, min_ascii_ratio, min_stopword_hits) else "non-en",
            "sender": sender,
            "sender_type": kind,
        })
    return finalize(pd.DataFrame(rows))
