"""IIIT-Delhi SMS spam dataset (SMSAssassin; Yadav, Kumaraguru, Goyal, Gupta, Naik 2011), from a GitHub copy. No
license stated.

File: `india/iiitd_smsassassin/indian_spam.csv` with columns v1 (ham / spam) and v2 (text): 1,000 ham and 1,000 spam
messages crowdsourced from 43 Indian volunteers, much of the ham Hinglish chat, so no language filter applies. No
sender. Mojibake rows are dropped as in junioralive_spam_ham.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .junioralive_spam_ham import mojibake
from .lang import script_language
from .schema import finalize

REL_PATH = Path("india/iiitd_smsassassin/indian_spam.csv")
LABELS = {"spam": 1, "ham": 0}


def load(raw_dir: Path) -> pd.DataFrame:
    rows = []
    with (Path(raw_dir) / REL_PATH).open(encoding="utf-8", newline="") as f:
        for i, rec in enumerate(csv.DictReader(f), start=1):
            if rec["v1"] not in LABELS:
                raise ValueError(f"unexpected IIIT-D label {rec['v1']!r} at row {i}")
            text = rec["v2"].strip()
            if not text or mojibake(text):
                continue
            rows.append({
                "text": text,
                "label": LABELS[rec["v1"]],
                "source": "iiitd_smsassassin",
                "source_id": str(i),
                "lang": script_language(text),
            })
    return finalize(pd.DataFrame(rows))
