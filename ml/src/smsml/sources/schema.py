"""Unified schema for every training/eval source.

Columns (in order):
  text         raw message text (decoded, outer whitespace stripped; no other normalization)
  label        0 = ham, 1 = spam (spam and smishing are both 1)
  source       source name (sources.SOURCES): inbox | uci | mishra | ncsu | imc25 | templates | hf_craftogrammer_txn |
               cube_sms77k | junioralive_spam_ham | iiitd_smsassassin | hf_ridham_synth_scam
  source_id    id unique within the source
  timestamp    UTC send/report time when known, else NaT
  campaign_id  campaign/cluster id from the source ("ncsu:<id>", "inbox:<template cluster>"), else NA
  category     source-provided category (IMC scam type, template category, mishra spam/smishing, synthetic-scam
               topic), else NA
  template_id  generating template for synthetic ham, else NA
  lang         "en" when known/heuristically English, else another code or NA
  sender       the sender's address as received (a DLT header such as "AX-HDFCBK-S", a phone number, a short code,
               an email address) when the source records it, else NA. Recorded by inbox, ncsu, cube_sms77k and
               hf_craftogrammer_txn; imc25 records only the kind and the others none. Masked numbers
               ("+1818355XXXX") and addresses the app can't classify are NA too.
  sender_type  PHONE | SHORT_CODE | ALPHANUMERIC | EMAIL: sender_type(sender) when there is a sender, else the kind
               the source records (imc25, masked ncsu numbers), else NA
  holdout      the evaluation split a source reserves the row for ("inbox_eval": the inbox's newest half), else NA.
               s01 keeps these rows out of deduplication and clustering with the rest and s02 puts them in that
               split, so nothing trains, selects or calibrates on them.
"""

from __future__ import annotations

import pandas as pd

COLUMNS = [
    "text", "label", "source", "source_id", "timestamp", "campaign_id", "category", "template_id", "lang", "sender",
    "sender_type", "holdout",
]
STRING_COLUMNS = ["text", "source", "source_id", "campaign_id", "category", "template_id", "lang", "sender",
                  "sender_type", "holdout"]
SENDER_TYPES = {"PHONE", "SHORT_CODE", "ALPHANUMERIC", "EMAIL"}
# The names s02 gives rows itself ("" while unplaced); a holdout can't take one.
GROUPED_SPLITS = {"", "train", "val", "val_a", "val_b", "test", "future", "capped"}


def sender_type(sender: str) -> str | None:
    """The app's kind of a sender address (AddressNormalizer.classify); None where the app says UNKNOWN."""
    a = sender.strip()
    if not a:
        return None
    if "@" in a:
        return "EMAIL"
    if any(c.isalpha() for c in a):
        return "ALPHANUMERIC"
    digits = sum(1 for c in a if "0" <= c <= "9")
    if not digits:
        return None
    if not a.startswith("+") and 3 <= digits <= 6:
        return "SHORT_CODE"
    return "PHONE"


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
    has = df["sender"].notna()
    expected = df.loc[has, "sender"].map(sender_type).astype("string")
    wrong = (expected != df.loc[has, "sender_type"]).fillna(True)
    if wrong.any():
        raise ValueError(f"sender_type disagrees with sender: "
                         f"{df.loc[wrong[wrong].index, ['sender', 'sender_type']].head().values}")
    bad_holdout = set(df["holdout"].dropna().unique()) & GROUPED_SPLITS
    if bad_holdout:
        raise ValueError(f"holdout can't be a grouped split name: {bad_holdout}")
