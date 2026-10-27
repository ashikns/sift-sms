"""Inbox dumps of 93 Indian delivery riders (Kaggle ramniwashkumar/sms-data, 2022; CC0), unlabeled.

File: `india/kaggle_rider_inbox/sms-data.zip`, read straight from the archive: `sms_data.csv` with columns
phoneNumber (the rider's own number; dropped), id (the rider), updateAt, senderAddress and text.

Not a corpus source: `pool()` gives one row per distinct (sender, text), with n (messages) and users (distinct
riders), for pseudo-labeling (scripts/s01b_rider_pool.py). Texts whose script was lost on export (most words are runs
of "?") are dropped. sender and sender_type follow the schema's rules.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pandas as pd

from .schema import sender_type

REL_PATH = Path("india/kaggle_rider_inbox/sms-data.zip")
MEMBER = "sms_data.csv"

csv.field_size_limit(1 << 30)


def garbled(text: str) -> bool:
    """Most words are runs of "?": a non-Latin script that the export couldn't encode."""
    words = [w for w in (w.strip(".,;:!-") for w in text.split()) if w]
    lost = sum(1 for w in words if len(w) >= 2 and set(w) == {"?"})
    return lost >= 2 and 2 * lost >= len(words)


def pool(raw_dir: Path) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(Path(raw_dir) / REL_PATH) as z, z.open(MEMBER) as raw:
        for rec in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline="")):
            text = rec["text"].strip()
            if not text or garbled(text):
                continue
            sender = rec["senderAddress"].strip()
            kind = sender_type(sender)
            rows.append((sender if kind else None, kind, text, rec["id"]))
    df = pd.DataFrame(rows, columns=["sender", "sender_type", "text", "user"])
    out = (df.groupby(["sender", "sender_type", "text"], dropna=False, sort=True)
           .agg(n=("user", "size"), users=("user", "nunique")).reset_index())
    for col in ("sender", "sender_type", "text"):
        out[col] = out[col].astype("string")
    return out
