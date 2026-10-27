"""s05: embed every message with the int8 ORT encoder, batch size 1 and no padding (as on the device).

    uv run python -I scripts/s05_embed.py

Dynamic quantization picks activation ranges per run, so batching or padding would change a message's
embedding. Writes data/work/emb_corpus.npy (+ emb_corpus_rows.npy: row positions in corpus_split.parquet, every
row except `capped`) and data/work/emb_modern.npy (rows of eval/modern_v1.jsonl).
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import onnxruntime as ort
import pandas as pd

from smsml.normalize import normalize
from smsml.pipeline import BUNDLE_WORK, config, load_modern_eval, snapshot_dir, work
from smsml.tokenizer import WordPiece

_SESS = None
_WP = None
THREADS = 2


def _init(model_path: str, vocab_path: str) -> None:
    global _SESS, _WP
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    _SESS = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    _WP = WordPiece.from_file(vocab_path)


def _embed(text: str) -> np.ndarray:
    ids = np.asarray([_WP.encode(normalize(text).enc)], dtype=np.int64)
    return _SESS.run(["embedding"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0].astype(np.float32)


def embed_all(texts: list[str], model: str, vocab: str) -> np.ndarray:
    procs = max(1, (os.cpu_count() or 4) // THREADS - 1)
    with ProcessPoolExecutor(procs, initializer=_init, initargs=(model, vocab)) as ex:
        out = list(ex.map(_embed, texts, chunksize=64))
    return np.stack(out) if out else np.zeros((0, 384), dtype=np.float32)


def main() -> int:
    cfg = config()
    model = str(BUNDLE_WORK / "encoder.int8.ort")
    vocab = str(snapshot_dir(cfg) / "vocab.txt")
    df = pd.read_parquet(work("corpus_split.parquet"))
    rows = np.flatnonzero((df["split"] != "capped").to_numpy())
    emb = embed_all(df["text"].iloc[rows].tolist(), model, vocab)
    np.save(work("emb_corpus.npy"), emb)
    np.save(work("emb_corpus_rows.npy"), rows)
    print(f"corpus: {emb.shape}")
    modern = load_modern_eval()
    em = embed_all(modern["text"].tolist(), model, vocab)
    np.save(work("emb_modern.npy"), em)
    print(f"modern: {em.shape}")
    norms = np.linalg.norm(emb, axis=1)
    print(f"norm range [{norms.min():.4f}, {norms.max():.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
