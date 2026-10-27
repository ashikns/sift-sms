"""Indian SMS corpus `cunique.csv` (~77k unique messages with sender and label; saranshbansal/spam-detection-analytics-
tool, listed as "Github1" in Salman et al. 2022, arXiv 2210.10451). No license stated.

File: `india/cube_sms77k/cunique.csv`, `;`-delimited and quoted, with columns id, Message, phone (the sender: DLT
headers such as `AD-HDFCBK`, numbers such as `919761943728`), batch, truth (spam / not-spam) and other classifiers'
guesses (cube, google*, ibm*; unused).

The text comes lowercased with every digit and punctuation mark already removed ("rs lac only for details misscall
on "). It's loaded as is, so its n-grams never hold digit runs or URL dots. About 10% is Hinglish chat, so no language
filter applies; `lang` comes from the script (lang.script_language).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .lang import script_language
from .schema import finalize, sender_type

REL_PATH = Path("india/cube_sms77k/cunique.csv")
LABELS = {"spam": 1, "not-spam": 0}

csv.field_size_limit(1 << 30)


def load(raw_dir: Path) -> pd.DataFrame:
    rows = []
    with (Path(raw_dir) / REL_PATH).open(encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f, delimiter=";"):
            if rec["truth"] not in LABELS:
                raise ValueError(f"unexpected cube_sms77k label {rec['truth']!r} at id {rec['id']}")
            text = rec["Message"].strip()
            if not text:
                continue
            sender = rec["phone"].strip()
            kind = sender_type(sender)
            rows.append({
                "text": text,
                "label": LABELS[rec["truth"]],
                "source": "cube_sms77k",
                "source_id": rec["id"],
                "lang": script_language(text),
                "sender": sender if kind else None,
                "sender_type": kind,
            })
    return finalize(pd.DataFrame(rows))
