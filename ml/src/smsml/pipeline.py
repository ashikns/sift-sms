"""Shared helpers for the training pipeline scripts (paths, config, parallel featurization)."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from .features import N_META, bucket_set, metadata, sparse_features
from .hashing import BUCKETS
from .normalize import normalize
from .resources import default_resources

ML_DIR = Path(__file__).resolve().parents[2]
REPO = ML_DIR.parent
WORK = ML_DIR / "data" / "work"
BUNDLE_WORK = WORK / "bundle"
PIPELINE_CONFIG = ML_DIR / "config" / "pipeline.yaml"
MODERN_EVAL = ML_DIR / "eval" / "modern_v1.jsonl"
APP_BUNDLE = REPO / "app" / "src" / "main" / "assets" / "spam_model"
HF_SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
WORKERS = max(1, (os.cpu_count() or 2) - 2)


def config() -> dict:
    with PIPELINE_CONFIG.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def snapshot_dir(cfg: dict) -> Path:
    return HF_SNAPSHOT / cfg["model"]["revision"]


def work(*parts: str) -> Path:
    p = WORK.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_one(text: str) -> tuple[str, str, str, int]:
    n = normalize(text, default_resources())
    return n.clean, n.enc, n.ng, len(n.urls)


def normalize_frame(df: pd.DataFrame, chunksize: int = 512) -> pd.DataFrame:
    """Adds clean / enc / ng / n_urls columns (norm_v1), computed in parallel."""
    with ProcessPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(_norm_one, df["text"].tolist(), chunksize=chunksize))
    out = df.copy()
    out["clean"] = [r[0] for r in rows]
    out["enc"] = [r[1] for r in rows]
    out["ng"] = [r[2] for r in rows]
    out["n_urls"] = np.array([r[3] for r in rows], dtype=np.int16)
    return out


def _buckets_one(text: str) -> list[int]:
    return sorted(bucket_set(normalize(text, default_resources())))


def document_frequencies(texts: list[str]) -> np.ndarray:
    df = np.zeros(BUCKETS, dtype=np.int64)
    with ProcessPoolExecutor(WORKERS) as ex:
        for buckets in ex.map(_buckets_one, texts, chunksize=512):
            df[buckets] += 1
    return df


_IDF: np.ndarray | None = None


def _init_idf(idf: np.ndarray) -> None:
    global _IDF
    _IDF = idf


def _feat_one(text: str):
    n = normalize(text, default_resources())
    idx, val = sparse_features(n, _IDF)
    return idx, val, metadata(n)


def featurize(texts: list[str], idf: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    """Hashed sparse features (CSR, n × 2^18) and metadata (n × 26) for raw texts, offline sender facts."""
    with ProcessPoolExecutor(WORKERS, initializer=_init_idf, initargs=(idf,)) as ex:
        rows = list(ex.map(_feat_one, texts, chunksize=256))
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(r[0]) for r in rows])
    indices = np.concatenate([r[0] for r in rows]) if rows else np.zeros(0, dtype=np.int32)
    data = np.concatenate([r[1] for r in rows]) if rows else np.zeros(0, dtype=np.float32)
    X = sp.csr_matrix((data, indices, indptr), shape=(len(rows), BUCKETS), dtype=np.float32)
    meta = np.stack([r[2] for r in rows]) if rows else np.zeros((0, N_META), dtype=np.float32)
    return X, meta


def load_modern_eval() -> pd.DataFrame:
    rows = [json.loads(line) for line in MODERN_EVAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)
