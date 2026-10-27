"""UCI SMS Spam Collection (Almeida & Gómez Hidalgo, 2011; CC BY 4.0): 5,574 English messages (UK/Singapore).

File: `uci/extracted/sms+spam+collection/SMSSpamCollection`, one `label<TAB>text` per line. Texts carry HTML
entities and ham-only placeholders (`&lt;#&gt;` etc.), which are unescaped and refilled (see refill.py).
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from .refill import refill_uci
from .schema import finalize

REL_PATH = Path("uci/extracted/sms+spam+collection/SMSSpamCollection")
LABELS = {"ham": 0, "spam": 1}


def decode_uci_text(raw: str, seed: int) -> str:
    """HTML-unescape and refill UCI-style placeholders (shared with the Mishra copy of these messages)."""
    return refill_uci(html.unescape(raw), seed).strip()


def load(raw_dir: Path, seed: int) -> pd.DataFrame:
    path = Path(raw_dir) / REL_PATH
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            label, _, text = line.partition("\t")
            text = decode_uci_text(text, seed)
            if not text:
                continue
            rows.append({
                "text": text,
                "label": LABELS[label.strip()],
                "source": "uci",
                "source_id": str(lineno),
                "lang": "en",
            })
    return finalize(pd.DataFrame(rows))
