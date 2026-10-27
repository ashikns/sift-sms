"""Unified schema for every training/eval source.

Columns (in order):
  text         raw message text (decoded, outer whitespace stripped; no other normalization)
  label        0 = ham, 1 = spam (spam and smishing are both 1)
  source       source name: uci | mishra | ncsu | imc25 | templates
  source_id    id unique within the source
  timestamp    UTC send/report time when known, else NaT
  campaign_id  campaign/cluster id from the source ("ncsu:<id>"), else NA
  category     source-provided category (IMC scam type, template category, mishra spam/smishing), else NA
  template_id  generating template for synthetic ham, else NA
  lang         "en" when known/heuristically English, else another code or NA
  sender_type  PHONE | SHORT_CODE | ALPHANUMERIC | EMAIL when the source records it, else NA
"""

from __future__ import annotations

import pandas as pd

COLUMNS = [
    "text", "label", "source", "source_id", "timestamp", "campaign_id", "category", "template_id", "lang",
    "sender_type",
]
STRING_COLUMNS = ["text", "source", "source_id", "campaign_id", "category", "template_id", "lang", "sender_type"]
SENDER_TYPES = {"PHONE", "SHORT_CODE", "ALPHANUMERIC", "EMAIL"}


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing optional columns, coerce dtypes and column order, and validate."""
    df = df.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NaT if col == "timestamp" else pd.NA
    df = df[COLUMNS]
    for col in STRING_COLUMNS:
        df[col] = df[col].astype("string")
    df["label"] = df["label"].astype("int8")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    validate(df)
    return df.reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df["text"].isna().any() or (df["text"].str.len() == 0).any():
        raise ValueError("empty text")
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError(f"labels must be 0/1, got {sorted(df['label'].unique())}")
    dup = df.duplicated(["source", "source_id"])
    if dup.any():
        raise ValueError(f"duplicate (source, source_id): {df.loc[dup, ['source', 'source_id']].head().values}")
    bad_sender = set(df["sender_type"].dropna().unique()) - SENDER_TYPES
    if bad_sender:
        raise ValueError(f"unknown sender_type values: {bad_sender}")
