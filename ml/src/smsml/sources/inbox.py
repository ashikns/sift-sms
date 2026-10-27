"""The user's own phone inbox, labeled by the user. Private: gitignored, never fetched by s00, never committed.

File: data.yaml `inbox.file`, one JSON object per line with id, date (epoch ms), sender (the address as received:
DLT headers such as `AX-HDFCBK-S`, phone numbers, short codes), text, label (0 ham, 1 spam), cluster (template
cluster: the messages of one template from one sender), half (`train` = the oldest 50%, `eval` = the newest 50%)
and marked (the user marked it in Sift; not used here).

- campaign_id = `inbox:<cluster>`, so s01 clusters a template's messages together and s02 splits them as one group.
- sender_type is the app's kind of the sender (schema.sender_type).
- Every language is kept, since Malayalam government alerts are real traffic too; `lang` comes from the script
  most of the letters are in (lang.script_language).
- The eval half is the `inbox_eval` holdout (schema `holdout`): kept apart from deduplication and clustering by s01
  and made its own split by s02, so it never trains, selects, calibrates or tunes anything. The train half is
  ordinary data in the grouped train/val/test split.

Rows come out in time order, so of a message's exact repeats s01 keeps the oldest. A missing file is an error, not a
skipped source: without it the model is neither trained nor evaluated on the user's own traffic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .lang import script_language
from .schema import finalize, sender_type

HOLDOUT = "inbox_eval"
HALVES = {"train": None, "eval": HOLDOUT}


def load(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: the labeled export of the phone inbox is private and local, so "
                                f"s00 doesn't fetch it (data.yaml inbox.file)")
    with path.open(encoding="utf-8") as f:
        records = [(lineno, json.loads(line)) for lineno, line in enumerate(f, start=1) if line.strip()]
    rows = []
    for lineno, rec in sorted(records, key=lambda r: (r[1]["date"], r[1]["id"])):
        if rec["label"] not in (0, 1) or rec["half"] not in HALVES:
            raise ValueError(f"unexpected inbox label/half {rec['label']!r}/{rec['half']!r} at line {lineno}")
        text = rec["text"].strip()
        if not text:
            continue
        sender = rec["sender"].strip()
        kind = sender_type(sender)
        rows.append({
            "text": text,
            "label": rec["label"],
            "source": "inbox",
            "source_id": str(rec["id"]),
            "timestamp": pd.Timestamp(rec["date"], unit="ms", tz="UTC"),
            "campaign_id": f"inbox:{rec['cluster']}",
            "lang": script_language(text),
            "sender": sender if kind else None,
            "sender_type": kind,
            "holdout": HALVES[rec["half"]],
        })
    return finalize(pd.DataFrame(rows))
