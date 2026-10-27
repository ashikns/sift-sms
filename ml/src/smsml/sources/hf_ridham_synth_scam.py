"""Indian Scam SMS, synthetic and audited (Ridham115/indian-scam-sms-synthetic-audited on Hugging Face; CC BY 4.0).

File: `india/hf_ridham_synth_scam/synthetic_audited.csv` with columns row_id, text, label (scam / safe, after the
audit), original_label, scam_type, language, generator, topic, audit_action: 1,580 LLM-written scams and genuine
look-alikes (bank, courier, bill, job, KYC, loan, lottery, everyday chat) in Indian English, Hindi (Devanagari),
Hinglish and Roman-script Tamil, Telugu, Bengali and Marathi code-mixing. scam → 1, safe → 0; category = topic, lang =
the row's language. Its rubric counts a genuine promotion from a named business as safe, where Sift calls it spam,
so a safe row with promotional wording ([PROMO]) is labeled 1.

Synthetic, so it only ever trains (registry `train_only`): never val, test or future.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from .schema import finalize

REL_PATH = Path("india/hf_ridham_synth_scam/synthetic_audited.csv")
LABELS = {"scam": 1, "safe": 0}
PROMO = re.compile(r"(?i)pre-?approved|eligib|\boffer\b|apply now|discount|% ?off|limited (time|period)|\bsale\b")


def load(raw_dir: Path) -> pd.DataFrame:
    rows = []
    with (Path(raw_dir) / REL_PATH).open(encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            if rec["label"] not in LABELS:
                raise ValueError(f"unexpected synthetic-scam label {rec['label']!r} at row {rec['row_id']}")
            text = rec["text"].strip()
            if not text:
                continue
            rows.append({
                "text": text,
                "label": 1 if rec["label"] == "safe" and PROMO.search(text) else LABELS[rec["label"]],
                "source": "hf_ridham_synth_scam",
                "source_id": rec["row_id"],
                "category": rec["topic"],
                "lang": rec["language"],
            })
    return finalize(pd.DataFrame(rows))
