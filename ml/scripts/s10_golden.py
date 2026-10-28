"""Writes parity fixtures for spamcore's JVM tests (SPEC.md).

    uv run python -I scripts/s10_golden.py --prelim          # normalization/tokens/features only (idf = 1)
    uv run python -I scripts/s10_golden.py --bundle <dir>    # + real idf, embeddings and scores from a model bundle

--prelim writes to spamcore/src/test/resources/golden/: cases jsonl, the trainer fixture, and the resource files and
vocab used to make them. Bundle mode writes app/src/androidTest/assets/golden/bundle_v1.jsonl.gz (read by spamcore's
GoldenBundleTest and the app's BundleParityTest). Every case carries a sender key (or none) and checks its sid, DLT
category and SND vector; its metadata has the DLT category set (the other sender features stay 0).

The fixtures are committed, so they only hold synthetic and hand-written cases and public-source messages: rows of a
private source (the user's own inbox) never go in, and both modes refuse to write a fixture with such a text.
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

from smsml.features import DLT_CATEGORIES, DLT_START, dlt_category, metadata, sender_id, snd_features, sparse_features
from smsml.golden import DLT_CASES, EDGE_CASES, edge_senders, random_senders, random_strings
from smsml.normalize import normalize
from smsml.resources import DEFAULT_DIR, FILES, Resources
from smsml.tokenizer import WordPiece

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "spamcore/src/test/resources/golden"
VOCAB = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/vocab.txt"
PRIVATE_SOURCES = {"inbox", "cube_sms77k", "iiitd_smsassassin", "hf_craftogrammer_txn"}  # the user's own messages, and sources with real people's numbers or no license: never in a committed fixture


def private_texts() -> set[str]:
    """Texts of private-source rows in the corpus (empty when the corpus hasn't been built)."""
    import pandas as pd

    from smsml.pipeline import WORK

    path = WORK / "corpus_split.parquet"
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["text", "source"])
    return set(df.loc[df["source"].isin(PRIVATE_SOURCES), "text"].tolist())


def check_public(cases: list[tuple[str, str | None]]) -> None:
    private = private_texts()
    leaked = sum(1 for raw, _ in cases if raw in private)
    if leaked:
        raise SystemExit(f"refusing to write a fixture: {leaked} cases are messages from {sorted(PRIVATE_SOURCES)}")


def corpus(extra: list[tuple[str, str | None]], n_random: int) -> list[tuple[str, str | None]]:
    """(text, sender key) cases."""
    return (list(zip(EDGE_CASES, edge_senders())) + [(t, k) for k, t in DLT_CASES]
            + list(zip(random_strings(n_random), random_senders(n_random))) + extra)


def case(raw: str, sender: str | None, res: Resources, wp: WordPiece, idf: np.ndarray) -> dict:
    n = normalize(raw, res)
    idx, val = sparse_features(n, idf)
    sid = sender_id(sender)
    snd_idx, snd_val = snd_features(sid, n)
    dlt = dlt_category(sender)
    meta = metadata(n)
    if dlt is not None:
        meta[DLT_START + DLT_CATEGORIES.index(dlt)] = 1.0
    return {
        "raw": raw,
        "sender": sender,
        "sid": sid,
        "dlt": dlt,
        "clean": n.clean,
        "enc": n.enc,
        "ng": n.ng,
        "urls": [asdict(u) for u in n.urls],
        "url_tokens": sorted(n.url_tokens),
        "counts": [n.cp_count, n.emoji_count, n.invisible_count, n.lookalike_count, n.dropped_count, int(n.non_gsm7)],
        "ids": wp.encode(n.enc),
        "idx": idx.tolist(),
        "val": [float(v) for v in val],
        "snd_idx": snd_idx.tolist(),
        "snd_val": [float(v) for v in snd_val],
        "meta": [float(v) for v in meta],
    }


def trainer_fixture(seed: int = 99) -> dict:
    """A compact random problem + the exact MAP optimum, for spamcore's trainer parity test. SND buckets partly share
    indices with the n-gram ones (the two blocks must stay separate), some examples have no sender, and each block
    has its own prior."""
    from smsml.features import N_META
    from smsml.head import EMBED_DIM, Example, Priors, Weighting, empty_head, example_weights, personalize

    rng = np.random.default_rng(seed)
    pool = rng.choice(1 << 18, size=600, replace=False)
    snd_pool = np.unique(np.concatenate([pool[:60], rng.choice(1 << 18, size=240, replace=False)]))
    head = empty_head()
    w_embed = rng.normal(0, 0.5, EMBED_DIM).astype(np.float32)
    mu_embed = rng.normal(0, 0.05, EMBED_DIM).astype(np.float32)
    w_meta = rng.normal(0, 0.5, N_META).astype(np.float32)
    mu_meta = rng.uniform(0, 0.2, N_META).astype(np.float32)
    hash_idx = np.sort(rng.choice(pool, size=300, replace=False)).astype(np.int32)
    hash_val = rng.normal(0, 0.8, hash_idx.size).astype(np.float32)
    w_hash = head.w_hash.copy()
    w_hash[hash_idx] = hash_val
    snd_base_idx = np.sort(rng.choice(snd_pool, size=150, replace=False)).astype(np.int32)
    snd_base_val = rng.normal(0, 0.8, snd_base_idx.size).astype(np.float32)
    w_snd = head.w_snd.copy()
    w_snd[snd_base_idx] = snd_base_val
    from dataclasses import replace
    head = replace(head, bias=-1.25, w_embed=w_embed, mu_embed=mu_embed, w_meta=w_meta, mu_meta=mu_meta, w_hash=w_hash,
                   w_snd=w_snd)

    def rand_sparse():
        idx = np.sort(rng.choice(pool, size=int(rng.integers(5, 40)), replace=False)).astype(np.int32)
        val = rng.normal(0, 1, idx.size)
        return idx, (val / np.linalg.norm(val)).astype(np.float32)

    def rand_snd():
        idx = np.sort(rng.choice(snd_pool, size=int(rng.integers(4, 30)), replace=False)).astype(np.int32)
        return idx, (rng.choice([-1.0, 1.0], idx.size) / np.sqrt(idx.size)).astype(np.float32)

    examples = []
    for i in range(60):
        if i % 7 == 3 and examples:  # near-duplicate of an earlier example from the same sender, same label
            src = examples[int(rng.integers(len(examples)))]
            val = src.val.astype(np.float64) + rng.normal(0, 0.05, src.val.size)
            idx, val = src.idx, (val / np.linalg.norm(val)).astype(np.float32)
            snd_idx, snd_val = src.snd_idx, src.snd_val
            label = src.label
        else:
            idx, val = rand_sparse()
            snd_idx, snd_val = (np.zeros(0, np.int32), np.zeros(0, np.float32)) if i % 5 == 2 else rand_snd()
            label = int(rng.integers(2))
        emb = None if i % 11 == 5 else rng.normal(0, 1, EMBED_DIM)
        if emb is not None:
            emb = (emb / np.linalg.norm(emb)).astype(np.float32)
        meta = rng.uniform(0, 1, N_META).astype(np.float32)
        examples.append(Example(emb, meta, idx, val, snd_idx, snd_val, label, key=1000 + i))
    weighting = Weighting(cluster_cos=0.8)
    weights = example_weights(examples, weighting)
    priors = Priors(embed=1.0, meta=4.0, hash=0.5, snd=2.0)
    out = personalize(head, examples, weights, priors)
    active = np.unique(np.concatenate([e.idx for e in examples]))
    snd_active = np.unique(np.concatenate([e.snd_idx for e in examples]))
    return {
        "base": {
            "bias": head.bias, "w_embed": w_embed.tolist(), "mu_embed": mu_embed.tolist(), "w_meta": w_meta.tolist(),
            "mu_meta": mu_meta.tolist(), "hash_idx": hash_idx.tolist(), "hash_val": hash_val.tolist(),
            "snd_idx": snd_base_idx.tolist(), "snd_val": snd_base_val.tolist(),
        },
        "weighting": weighting.__dict__,
        "priors": priors.__dict__,
        "examples": [
            {"key": e.key, "label": e.label, "created_at": i,
             "emb": None if e.emb is None else e.emb.tolist(), "meta": e.meta.tolist(),
             "idx": e.idx.tolist(), "val": e.val.tolist(), "snd_idx": e.snd_idx.tolist(), "snd_val": e.snd_val.tolist()}
            for i, e in enumerate(examples)
        ],
        "expected": {
            "weights": weights.tolist(),
            "z": [head.z(e.emb, e.meta, e.idx, e.val, e.snd_idx, e.snd_val) for e in examples],
            "w_embed": out.w_embed.tolist(), "w_meta": out.w_meta.tolist(),
            "active_idx": active.tolist(), "active_val": out.w_hash[active].tolist(),
            "snd_active_idx": snd_active.tolist(), "snd_active_val": out.w_snd[snd_active].tolist(),
        },
    }


def bundle_mode(args) -> int:
    """Cases with the packaged bundle: real idf, int8 embeddings (batch 1), logits and probabilities."""
    import onnxruntime as ort
    import pandas as pd

    from smsml.head import Head
    from smsml.pipeline import APP_BUNDLE, load_modern_eval, sender_key, senders_of, work

    bundle = args.bundle or APP_BUNDLE
    res = Resources.load(bundle / "resources")
    wp = WordPiece.from_file(bundle / "vocab.txt")
    head = Head.load(bundle / "head_base.bin")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(bundle / "encoder.int8.ort"), so, providers=["CPUExecutionProvider"])
    df = pd.read_parquet(work("corpus_split.parquet"))
    public = df[(df["split"] == "test") & ~df["source"].isin(PRIVATE_SOURCES)].sample(n=300, random_state=5)
    modern = load_modern_eval().sample(n=200, random_state=5)

    def keyed(frame) -> list[tuple[str, str | None]]:
        return [(t, None if s is None else sender_key(s)) for t, s in zip(frame["text"].tolist(), senders_of(frame))]

    cases = (list(zip(EDGE_CASES, edge_senders())) + [(t, k) for k, t in DLT_CASES]
             + list(zip(random_strings(200, seed=77), random_senders(200, seed=78))) + keyed(public) + keyed(modern))
    check_public(cases)
    device = ROOT / "app/src/androidTest/assets/golden"
    device.mkdir(parents=True, exist_ok=True)
    out = device / "bundle_v1.jsonl.gz"  # read by spamcore's GoldenBundleTest and the app's BundleParityTest
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for raw, sender in cases:
            c = case(raw, sender, res, wp, head.idf)
            ids = np.asarray([c["ids"]], dtype=np.int64)
            e = sess.run(["embedding"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]
            # Stored with 7 significant digits; the logit below uses the stored values so both sides see the same input.
            e = np.asarray([float(f"{x:.7g}") for x in e], dtype=np.float32)
            idx = np.asarray(c["idx"], dtype=np.int32)
            val = np.asarray(c["val"], dtype=np.float32)
            snd_idx = np.asarray(c["snd_idx"], dtype=np.int32)
            snd_val = np.asarray(c["snd_val"], dtype=np.float32)
            meta = np.asarray(c["meta"], dtype=np.float32)
            z = head.z(e, meta, idx, val, snd_idx, snd_val)
            c.update({
                "emb": [float(x) for x in e],
                "z": z,
                "z_fb": head.z_fallback(meta, idx, val, snd_idx, snd_val),
                "p": float(1 / (1 + np.exp(-z))),
            })
            f.write(json.dumps(c, ensure_ascii=True) + "\n")
    print(f"wrote {len(cases)} bundle cases to {out} (bundle {bundle})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prelim", action="store_true")
    ap.add_argument("--bundle", type=Path, help="bundle directory (default: the app's assets/spam_model)")
    ap.add_argument("--resources", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--extra", type=Path, help="jsonl with a 'text' and optional 'sender' (key) field to add to the corpus")
    ap.add_argument("--n-random", type=int, default=1500)
    args = ap.parse_args()
    if not args.prelim:
        return bundle_mode(args)

    res = Resources.load(args.resources)
    wp = WordPiece.from_file(VOCAB)
    idf = np.ones(1 << 18, dtype=np.float32)
    extra = []
    if args.extra:
        rows = [json.loads(line) for line in args.extra.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(r.get("source") in PRIVATE_SOURCES for r in rows):
            raise SystemExit(f"refusing --extra rows from {sorted(PRIVATE_SOURCES)}")
        extra = [(r["text"], r.get("sender")) for r in rows]

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
    check_public(extra)  # the hand-written and random cases aren't anyone's messages (a lone "lol" is also in a chat set)
    with gzip.open(OUT / "norm_v1_prelim.jsonl.gz", "wt", encoding="utf-8") as f:
        for raw, sender in cases:
            f.write(json.dumps(case(raw, sender, res, wp, idf), ensure_ascii=True) + "\n")
    print(f"wrote {len(cases)} cases to {OUT / 'norm_v1_prelim.jsonl.gz'}")
    (OUT / "trainer_v1.json").write_text(json.dumps(trainer_fixture()), encoding="utf-8")
    print(f"wrote {OUT / 'trainer_v1.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
