"""Source loaders producing one unified DataFrame (see schema.py).

    from smsml.sources import load_all
    df = load_all()                        # English rows from every available source
    df = load_all(include_non_english=True)

Raw files come from scripts/s00_fetch.py (ml/data/raw/). A source whose files are missing is skipped with a
warning so the pipeline still runs on whatever was fetched.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import yaml

from . import imc25, mishra, ncsu, templates, uci
from .schema import COLUMNS, finalize

ML_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = ML_DIR / "data" / "raw"
DATA_CONFIG = ML_DIR / "config" / "data.yaml"

__all__ = ["COLUMNS", "load_all", "load_config", "RAW_DIR", "ML_DIR"]


def load_config(path: Path = DATA_CONFIG) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all(raw_dir: Path = RAW_DIR, config: dict | None = None, *, include_non_english: bool = False,
             sources: tuple[str, ...] = ("uci", "mishra", "ncsu", "imc25", "templates")) -> pd.DataFrame:
    cfg = config or load_config()
    seed = int(cfg["seed"])
    raw_dir = Path(raw_dir)
    frames: list[pd.DataFrame] = []

    def attempt(name: str, fn):
        if name not in sources:
            return
        try:
            frames.append(fn())
        except FileNotFoundError as e:
            warnings.warn(f"source {name!r} skipped: {e}")

    attempt("uci", lambda: uci.load(raw_dir, seed))
    attempt("mishra", lambda: mishra.load(raw_dir, seed))
    attempt("ncsu", lambda: ncsu.load(
        raw_dir, min_chars=int(cfg["ncsu"]["min_chars"]),
        min_ascii_ratio=float(cfg["ncsu"]["min_ascii_letter_ratio"]),
        min_stopword_hits=int(cfg["ncsu"]["min_stopword_hits"])))
    if "imc25" in sources:
        try:
            hosts = ncsu.smishing_hosts(raw_dir)
        except FileNotFoundError:
            hosts = []
        attempt("imc25", lambda: imc25.load(raw_dir, refill_seed=int(cfg["imc25"]["refill_seed"]), hosts=hosts))
    tcfg = cfg["templates"]
    attempt("templates", lambda: templates.load(
        ML_DIR / tcfg["file"], seed=seed, target_total=int(tcfg["target_total"]),
        max_per_template=int(tcfg["max_per_template"])))

    if not frames:
        raise FileNotFoundError(f"no sources found under {raw_dir}; run scripts/s00_fetch.py first")
    df = pd.concat(frames, ignore_index=True)
    if not include_non_english:
        df = df[df["lang"] == "en"]
    return finalize(df)
