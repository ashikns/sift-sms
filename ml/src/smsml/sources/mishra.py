"""SMS Phishing Dataset for ML and Pattern Recognition (Mishra & Soni, 2022; CC BY 4.0; Mendeley f45bkkt8pr).

File: `mishra/extracted/Dataset_5971/Dataset_5971.csv` with columns LABEL, TEXT, URL, EMAIL, PHONE.
LABEL is ham / spam / Smishing (inconsistent casing). Texts were OCR'd from screenshots; some contain U+FFFD
where a currency sign was lost. Much of the ham overlaps the UCI collection, so the same UCI unescape/refill is
applied to keep those duplicates byte-identical for deduplication.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .schema import finalize
from .uci import decode_uci_text

REL_PATH = Path("mishra/extracted/Dataset_5971/Dataset_5971.csv")


def load(raw_dir: Path, seed: int) -> pd.DataFrame:
    path = Path(raw_dir) / REL_PATH
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for i, rec in enumerate(csv.DictReader(f), start=1):
            label = rec["LABEL"].strip().lower()
            text = decode_uci_text(rec["TEXT"], seed)
            if not text:
                continue
            if label not in ("ham", "spam", "smishing"):
                raise ValueError(f"unexpected Mishra label {rec['LABEL']!r} at row {i}")
            rows.append({
                "text": text,
                "label": 0 if label == "ham" else 1,
                "source": "mishra",
                "source_id": str(i),
                "category": None if label == "ham" else label,
                "lang": "en",
            })
    return finalize(pd.DataFrame(rows))
