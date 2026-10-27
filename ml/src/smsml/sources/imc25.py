"""UCL/IMDEA smishing reports (Agarwal et al., "Fishing for Smishing", IMC 2025; CC BY 4.0).

File: `imc25/final_dataset_output.csv`, 33,869 messages OCR'd from screenshots that users posted publicly
(2017–2024), with columns time, sender_id (phone number / alphanumeric / email), telephone_number_type,
original_network_name, original_network_country, text, translation, language, url_shortener, named_entity
(impersonated brand), scam_type, lure_principles.

Only `language == "English"` rows are kept (22,078). PII was replaced by Presidio tags (`<URL>`, `<DATE_TIME>`,
`<NAMED_ENTITY>`, `<PHONE_NUMBER>`, ...); they are refilled deterministically (refill.py), using the row's recorded
URL shortener for its first `<URL>` and real NCSU smishing hosts otherwise. The `time` column is the time shown in
the screenshot ("Yesterday 22:50", "Fri, Apr 14 at 5:59 AM"); a timestamp is set only when it contains a full date.
A few rows hold several message bubbles as a Python list literal (`['msg 1', 'msg 2']`); those are split into one
row per message (`source_id` = `<row>.<k>`).
"""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

import pandas as pd

from .refill import refill_imc
from .schema import finalize

REL_PATH = Path("imc25/final_dataset_output.csv")
SENDER_TYPES = {"phone number": "PHONE", "alphanumeric": "ALPHANUMERIC", "email": "EMAIL"}
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
ISO_DATE = re.compile(r"\b(20[12]\d)-(\d{2})-(\d{2})\b")
YEAR = re.compile(r"\b(20[12]\d)\b")
MONTH_DAY = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b")
DAY_MONTH = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\b")
NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](20[12]\d)\b")

csv.field_size_limit(1 << 30)


def parse_imc_time(value: str) -> pd.Timestamp:
    """Best-effort date from a screenshot time string; NaT unless year, month and day are all present."""
    s = value.strip()
    if not s:
        return pd.NaT
    m = ISO_DATE.search(s)
    if m:
        return _ts(int(m[1]), int(m[2]), int(m[3]))
    m = NUMERIC_DATE.search(s)
    if m:
        a, b, year = int(m[1]), int(m[2]), int(m[3])
        if a > 12 >= b:
            return _ts(year, b, a)
        if b > 12 >= a:
            return _ts(year, a, b)
        return pd.NaT  # ambiguous MM/DD vs DD/MM
    y = YEAR.search(s)
    if not y:
        return pd.NaT
    for pattern, month_group, day_group in ((MONTH_DAY, 1, 2), (DAY_MONTH, 2, 1)):
        for m in pattern.finditer(s):
            month = MONTHS.get(m[month_group][:3].lower())
            if month:
                return _ts(int(y[1]), month, int(m[day_group]))
    return pd.NaT


def _ts(year: int, month: int, day: int) -> pd.Timestamp:
    try:
        return pd.Timestamp(year=year, month=month, day=day, tz="UTC")
    except ValueError:
        return pd.NaT


def load(raw_dir: Path, *, refill_seed: int, hosts: list[str]) -> pd.DataFrame:
    rows = []
    with (Path(raw_dir) / REL_PATH).open(encoding="utf-8", newline="") as f:
        for i, rec in enumerate(csv.DictReader(f), start=1):
            if rec["language"].strip() != "English":
                continue
            parts = split_bubbles(rec["text"])
            for k, original in enumerate(parts):
                text = refill_imc(
                    original, refill_seed, hosts=hosts,
                    shortener=rec["url_shortener"].strip() or None,
                    named_entity=rec["named_entity"].strip() or None,
                ).strip()
                if not text:
                    continue
                rows.append({
                    "text": text,
                    "label": 1,
                    "source": "imc25",
                    "source_id": str(i) if len(parts) == 1 else f"{i}.{k + 1}",
                    "timestamp": parse_imc_time(rec["time"]),
                    "category": rec["scam_type"].strip() or None,
                    "lang": "en",
                    "sender_type": SENDER_TYPES.get(rec["sender_id"].strip()),
                })
    return finalize(pd.DataFrame(rows))


def split_bubbles(raw: str) -> list[str]:
    """`['a', 'b']` (a list literal of several screenshot bubbles) → ['a', 'b']; anything else → [raw]."""
    s = raw.strip()
    if s[:2] in ("['", '["') and s.endswith("]"):
        try:
            value = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return [s]
        if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
            return [x.strip() for x in value if x.strip()]
    return [s] if s else []
