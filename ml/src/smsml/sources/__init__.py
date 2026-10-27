"""Source loaders producing one unified DataFrame (see schema.py).

    from smsml.sources import load_all
    df = load_all()                        # every source; the public corpora's English rows only
    df = load_all(include_non_english=True)

SOURCES registers each corpus: its loader, whether a missing file is an error, whether its non-English rows are
dropped, and whether it may only train (synthetic data). The public corpora come from scripts/s00_fetch.py
(ml/data/raw/, the Indian ones under india/); one whose files are missing is skipped with a warning so the pipeline
still runs on whatever was fetched. The inbox is local and private, and required. A new source is a loader module
returning schema rows plus one entry here (and its s00 and data.yaml entries, if any). Order matters: of exact
duplicates, s01 keeps the first source's copy.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from . import (cube_sms77k, hf_craftogrammer_txn, hf_ridham_synth_scam, iiitd_smsassassin, imc25, inbox,
               junioralive_spam_ham, mishra, ncsu, templates, uci)
from .schema import COLUMNS, finalize

ML_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = ML_DIR / "data" / "raw"
DATA_CONFIG = ML_DIR / "config" / "data.yaml"

__all__ = ["COLUMNS", "SOURCES", "Source", "load_all", "load_config", "RAW_DIR", "ML_DIR"]


@dataclass(frozen=True)
class Source:
    name: str
    load: Callable[[Path, dict], pd.DataFrame]  # (raw dir, data.yaml) → rows in the unified schema
    required: bool = False  # a missing file fails instead of skipping the source
    english_only: bool = True  # non-English rows are dropped (unless include_non_english)
    train_only: bool = False  # synthetic: s02 never puts its rows in val, test or future


def _imc25(raw_dir: Path, cfg: dict) -> pd.DataFrame:
    try:
        hosts = ncsu.smishing_hosts(raw_dir)
    except FileNotFoundError:
        hosts = []
    return imc25.load(raw_dir, refill_seed=int(cfg["imc25"]["refill_seed"]), hosts=hosts)


SOURCES = (
    # The user's own traffic, in every language it arrives in. First, so a missing export fails before anything loads
    # (and an inbox message keeps its sender when a public corpus has the same text).
    Source("inbox", lambda raw, cfg: inbox.load(raw / cfg["inbox"]["file"]), required=True, english_only=False),
    Source("uci", lambda raw, cfg: uci.load(raw, int(cfg["seed"]))),
    Source("mishra", lambda raw, cfg: mishra.load(raw, int(cfg["seed"]))),
    Source("ncsu", lambda raw, cfg: ncsu.load(
        raw, min_chars=int(cfg["ncsu"]["min_chars"]), min_ascii_ratio=float(cfg["ncsu"]["min_ascii_letter_ratio"]),
        min_stopword_hits=int(cfg["ncsu"]["min_stopword_hits"]))),
    Source("imc25", _imc25),
    Source("templates", lambda raw, cfg: templates.load(
        ML_DIR / cfg["templates"]["file"], seed=int(cfg["seed"]), target_total=int(cfg["templates"]["target_total"]),
        max_per_template=int(cfg["templates"]["max_per_template"]))),
    # Indian corpora: the ones that record senders first, so their copy of a shared text wins.
    Source("hf_craftogrammer_txn", lambda raw, cfg: hf_craftogrammer_txn.load(raw)),
    Source("cube_sms77k", lambda raw, cfg: cube_sms77k.load(raw), english_only=False),
    Source("junioralive_spam_ham", lambda raw, cfg: junioralive_spam_ham.load(raw)),
    Source("iiitd_smsassassin", lambda raw, cfg: iiitd_smsassassin.load(raw), english_only=False),
    Source("hf_ridham_synth_scam", lambda raw, cfg: hf_ridham_synth_scam.load(raw), english_only=False,
           train_only=True),
)


def load_config(path: Path = DATA_CONFIG) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all(raw_dir: Path = RAW_DIR, config: dict | None = None, *, include_non_english: bool = False,
             sources: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Rows of every registered source, or only of `sources`."""
    cfg = config or load_config()
    frames: list[pd.DataFrame] = []
    for src in SOURCES:
        if sources is not None and src.name not in sources:
            continue
        try:
            df = src.load(Path(raw_dir), cfg)
        except FileNotFoundError as e:
            if src.required:
                raise
            warnings.warn(f"source {src.name!r} skipped: {e}")
            continue
        if src.english_only and not include_non_english:
            df = df[df["lang"] == "en"]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no sources found under {raw_dir}; run scripts/s00_fetch.py first")
    return finalize(pd.concat(frames, ignore_index=True))
