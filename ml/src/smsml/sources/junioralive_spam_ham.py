"""India Spam SMS Classification (junioralive, 2024; MIT): 2,267 Indian messages (operator and bank offers,
government notices, chat).

File: `india/junioralive_spam_ham/spam_ham_india.csv` with columns Msg and Label (spam / ham); no sender. About 180
rows reached the CSV as mojibake (UTF-8 read as Latin-1/cp1252 upstream: "assistantÃ¢Â\\x80Â\\x99s", an emoji as
"Ã°Â\\x9f…"); they're dropped rather than guessed back.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .lang import script_language
from .schema import finalize

REL_PATH = Path("india/junioralive_spam_ham/spam_ham_india.csv")
LABELS = {"spam": 1, "ham": 0}
# How the lead bytes of multi-byte UTF-8 sequences look once read as Latin-1/cp1252.
MOJIBAKE_MARKS = ("Ã", "Â", "â€")


def mojibake(text: str) -> bool:
    """UTF-8 that was decoded as Latin-1/cp1252: C1 control characters, or the marks of misread UTF-8 lead bytes."""
    return any("\x80" <= c <= "\x9f" for c in text) or any(m in text for m in MOJIBAKE_MARKS)


def load(raw_dir: Path) -> pd.DataFrame:
    rows = []
    with (Path(raw_dir) / REL_PATH).open(encoding="utf-8", newline="") as f:
        for i, rec in enumerate(csv.DictReader(f), start=1):
            if rec["Label"] not in LABELS:
                raise ValueError(f"unexpected junioralive label {rec['Label']!r} at row {i}")
            text = rec["Msg"].strip()
            if not text or mojibake(text):
                continue
            rows.append({
                "text": text,
                "label": LABELS[rec["Label"]],
                "source": "junioralive_spam_ham",
                "source_id": str(i),
                "lang": script_language(text),
            })
    return finalize(pd.DataFrame(rows))
