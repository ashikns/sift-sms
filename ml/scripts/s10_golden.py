"""Writes parity fixtures for spamcore's JVM tests (SPEC.md).

    uv run python -I scripts/s10_golden.py --prelim          # normalization/tokens/features only (idf = 1)
    uv run python -I scripts/s10_golden.py --bundle <dir>    # + real idf, embeddings and scores from a model bundle

Outputs go to spamcore/src/test/resources/golden/: cases jsonl, the resource files and vocab used to make them.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from smsml.features import metadata, sparse_features
from smsml.golden import EDGE_CASES, random_strings
from smsml.normalize import normalize
from smsml.resources import DEFAULT_DIR, FILES, Resources
from smsml.tokenizer import WordPiece

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "spamcore/src/test/resources/golden"
VOCAB = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/vocab.txt"


def corpus(extra: list[str], n_random: int) -> list[str]:
    return EDGE_CASES + random_strings(n_random) + extra


def case(raw: str, res: Resources, wp: WordPiece, idf: np.ndarray) -> dict:
    n = normalize(raw, res)
    idx, val = sparse_features(n, idf)
    return {
        "raw": raw,
        "clean": n.clean,
        "enc": n.enc,
        "ng": n.ng,
        "urls": [asdict(u) for u in n.urls],
        "url_tokens": sorted(n.url_tokens),
        "counts": [n.cp_count, n.emoji_count, n.invisible_count, n.lookalike_count, n.dropped_count, int(n.non_gsm7)],
        "ids": wp.encode(n.enc),
        "idx": idx.tolist(),
        "val": [float(v) for v in val],
        "meta": [float(v) for v in metadata(n)],
    }


def trainer_fixture(seed: int = 99) -> dict:
    """A compact random problem + the exact MAP optimum, for spamcore's trainer parity test."""
    from smsml.features import N_META
    from smsml.head import EMBED_DIM, Example, Priors, Weighting, empty_head, example_weights, personalize

    rng = np.random.default_rng(seed)
    pool = rng.choice(1 << 18, size=600, replace=False)
    head = empty_head()
    w_embed = rng.normal(0, 0.5, EMBED_DIM).astype(np.float32)
    mu_embed = rng.normal(0, 0.05, EMBED_DIM).astype(np.float32)
    w_meta = rng.normal(0, 0.5, N_META).astype(np.float32)
    mu_meta = rng.uniform(0, 0.2, N_META).astype(np.float32)
    hash_idx = np.sort(rng.choice(pool, size=300, replace=False)).astype(np.int32)
    hash_val = rng.normal(0, 0.8, hash_idx.size).astype(np.float32)
    w_hash = head.w_hash.copy()
    w_hash[hash_idx] = hash_val
    from dataclasses import replace
    head = replace(head, bias=-1.25, w_embed=w_embed, mu_embed=mu_embed, w_meta=w_meta, mu_meta=mu_meta, w_hash=w_hash)

    def rand_sparse():
        idx = np.sort(rng.choice(pool, size=int(rng.integers(5, 40)), replace=False)).astype(np.int32)
        val = rng.normal(0, 1, idx.size)
        return idx, (val / np.linalg.norm(val)).astype(np.float32)

    examples = []
    for i in range(60):
        if i % 7 == 3 and examples:  # near-duplicate of an earlier example, same label
            src = examples[int(rng.integers(len(examples)))]
            val = src.val.astype(np.float64) + rng.normal(0, 0.05, src.val.size)
            idx, val = src.idx, (val / np.linalg.norm(val)).astype(np.float32)
            label = src.label
        else:
            idx, val = rand_sparse()
            label = int(rng.integers(2))
        emb = None if i % 11 == 5 else rng.normal(0, 1, EMBED_DIM)
        if emb is not None:
            emb = (emb / np.linalg.norm(emb)).astype(np.float32)
        meta = rng.uniform(0, 1, N_META).astype(np.float32)
        examples.append(Example(emb, meta, idx, val, label, 0.0, key=1000 + i))
    weighting = Weighting(cluster_cos=0.8)
    weights = example_weights(examples, weighting)
    priors = Priors()
    out = personalize(head, examples, weights, priors)
    active = np.unique(np.concatenate([e.idx for e in examples]))
    return {
        "base": {
            "bias": head.bias, "w_embed": w_embed.tolist(), "mu_embed": mu_embed.tolist(), "w_meta": w_meta.tolist(),
            "mu_meta": mu_meta.tolist(), "hash_idx": hash_idx.tolist(), "hash_val": hash_val.tolist(),
        },
        "weighting": weighting.__dict__,
        "priors": priors.__dict__,
        "examples": [
            {"key": e.key, "label": e.label, "created_at": i,
             "emb": None if e.emb is None else e.emb.tolist(), "meta": e.meta.tolist(),
             "idx": e.idx.tolist(), "val": e.val.tolist()}
            for i, e in enumerate(examples)
        ],
        "expected": {
            "weights": weights.tolist(),
            "w_embed": out.w_embed.tolist(), "w_meta": out.w_meta.tolist(),
            "active_idx": active.tolist(), "active_val": out.w_hash[active].tolist(),
        },
    }


def bundle_mode(args) -> int:
    """Cases with the packaged bundle: real idf, int8 embeddings (batch 1), logits and probabilities."""
    import onnxruntime as ort
    import pandas as pd

    from smsml.head import Head
    from smsml.pipeline import APP_BUNDLE, load_modern_eval, work

    bundle = args.bundle or APP_BUNDLE
    res = Resources.load(bundle / "resources")
    wp = WordPiece.from_file(bundle / "vocab.txt")
    head = Head.load(bundle / "head_base.bin")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(bundle / "encoder.int8.ort"), so, providers=["CPUExecutionProvider"])
    df = pd.read_parquet(work("corpus_split.parquet"))
    corpus = df[df["split"] == "test"].sample(n=300, random_state=5)["text"].tolist()
    modern = load_modern_eval().sample(n=200, random_state=5)["text"].tolist()
    texts = EDGE_CASES + random_strings(200, seed=77) + corpus + modern
    device = ROOT / "app/src/androidTest/assets/golden"
    device.mkdir(parents=True, exist_ok=True)
    out = device / "bundle_v1.jsonl.gz"  # read by spamcore's GoldenBundleTest and the app's BundleParityTest
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for raw in texts:
            c = case(raw, res, wp, head.idf)
            n = normalize(raw, res)
            ids = np.asarray([c["ids"]], dtype=np.int64)
            e = sess.run(["embedding"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]
            # Stored with 7 significant digits; the logit below uses the stored values so both sides see the same input.
            e = np.asarray([float(f"{x:.7g}") for x in e], dtype=np.float32)
            idx = np.asarray(c["idx"], dtype=np.int32)
            val = np.asarray(c["val"], dtype=np.float32)
            meta = metadata(n)
            z = head.z(e, meta, idx, val)
            c.update({
                "emb": [float(x) for x in e],
                "z": z,
                "z_fb": head.z_fallback(meta, idx, val),
                "p": float(1 / (1 + np.exp(-z))),
            })
            f.write(json.dumps(c, ensure_ascii=True) + "\n")
    print(f"wrote {len(texts)} bundle cases to {out} (bundle {bundle})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prelim", action="store_true")
    ap.add_argument("--bundle", type=Path, help="bundle directory (default: the app's assets/spam_model)")
    ap.add_argument("--resources", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--extra", type=Path, help="jsonl with a 'text' field to add to the corpus")
    ap.add_argument("--n-random", type=int, default=1500)
    args = ap.parse_args()
    if not args.prelim:
        return bundle_mode(args)

    res = Resources.load(args.resources)
    wp = WordPiece.from_file(VOCAB)
    idf = np.ones(1 << 18, dtype=np.float32)
    extra = []
    if args.extra:
        extra = [json.loads(line)["text"] for line in args.extra.read_text(encoding="utf-8").splitlines() if line.strip()]

    OUT.mkdir(parents=True, exist_ok=True)
    res_out = OUT / "resources"
    res_out.mkdir(exist_ok=True)
    for name in FILES:
        src = args.resources / name
        dst = res_out / name
        if src.exists():
            shutil.copyfile(src, dst)
        elif dst.exists():
            dst.unlink()
    shutil.copyfile(VOCAB, OUT / "vocab.txt")
    cases = corpus(extra, args.n_random)
    with gzip.open(OUT / "norm_v1_prelim.jsonl.gz", "wt", encoding="utf-8") as f:
        for raw in cases:
            f.write(json.dumps(case(raw, res, wp, idf), ensure_ascii=True) + "\n")
    print(f"wrote {len(cases)} cases to {OUT / 'norm_v1_prelim.jsonl.gz'}")
    (OUT / "trainer_v1.json").write_text(json.dumps(trainer_fixture()), encoding="utf-8")
    print(f"wrote {OUT / 'trainer_v1.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
