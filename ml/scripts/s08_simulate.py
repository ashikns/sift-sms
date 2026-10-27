"""s08: simulate users giving feedback, to tune the on-device personalization defaults.

    uv run python -I scripts/s08_simulate.py [--quick]

Streams (validation clusters + modern dev messages, ~25% spam, campaigns arriving in bursts, ham from a set of
repeat senders) are replayed against simulated users. A message whose source records its sender keeps that sender
(its sid) and its sender × content (SND) vector; the others get a simulated sender for the memory's same-sender rule
and no SND vector, so for them the snd prior is a no-op.
  diligent   marks every missed spam; finds a false positive in the spam folder with probability 0.3
  spam_only  only ever marks missed spam (the drift scenario)
  noisy      like diligent, but 5% of marks are wrong
Each message is decided like the engine does, from its own content (near-duplicate memory → model → OTP
guardrail). The user's marks are the only training signal: each one updates the memory immediately and the head is
retrained (MAP). The probe set measures drift (legit messages a retrain newly flags).
Cost = 5·FP + 1·FN. Picks the grid point with the lowest diligent-user cost whose spam_only probe-FPR increase
stays ≤ 0.3 points, and writes data/work/personalization.json.
"""

from __future__ import annotations

import os

# One BLAS thread per process: the pool runs one simulation per core, and forked workers inherit the parent's BLAS
# setup, so this must happen before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from smsml.decide import SPAM, Thresholds, decide
from smsml.features import META_NAMES, sender_id
from smsml.head import Example, Head, Priors, Weighting, example_weights, personalize
from smsml.pipeline import (BUNDLE_WORK, WORKERS, config, featurize, load_modern_eval, sender_key, senders_of, work,
                            write_json)

FROZEN = 1e6  # a prior this strong keeps a block at its base weights
PERSONAL = {"personal", "first_contact"}
OTP = META_NAMES.index("otp_pattern")
URLS = META_NAMES.index("url_count")
KIND = {"SHORT_CODE": 21, "ALPHANUMERIC": 23, "EMAIL": 24}


@dataclass
class Msg:
    y: int
    emb: np.ndarray
    meta: np.ndarray
    idx: np.ndarray
    val: np.ndarray
    snd_idx: np.ndarray
    snd_val: np.ndarray
    sid: str | None  # the message's own sender, when its source records one
    sender: str  # who sends it in the stream: the sid, else a simulated sender
    cluster: str
    text: str = ""


def build_stream(seed: int, pool: list[Msg], n: int = 2400, spam_frac: float = 0.25) -> list[Msg]:
    rng = np.random.default_rng(seed)
    spam = [m for m in pool if m.y == 1]
    ham = [m for m in pool if m.y == 0]
    n_spam = min(len(spam), int(n * spam_frac))
    n_ham = min(len(ham), n - n_spam)
    chosen = [spam[i] for i in rng.choice(len(spam), n_spam, replace=False)] + \
             [ham[i] for i in rng.choice(len(ham), n_ham, replace=False)]
    # Campaign bursts: each cluster starts at a random time; its messages follow closely.
    # Sorted: set iteration order of strings changes between processes (hash randomization).
    start = {c: rng.uniform(0, 1000) for c in sorted({m.cluster for m in chosen})}
    times = [start[m.cluster] + rng.exponential(2.0) for m in chosen]
    order = np.argsort(times)
    friends = [f"friend{i}" for i in range(60)]
    zipf = 1 / np.arange(1, 61)
    zipf /= zipf.sum()
    out = []
    for i in order:
        m = chosen[i]
        if m.y == 0:
            sender = friends[rng.choice(60, p=zipf)]
        else:
            sender = f"camp-{m.cluster}" if rng.random() < 0.3 else f"spam-{rng.integers(1 << 40)}"
        out.append(Msg(m.y, m.emb, m.meta, m.idx, m.val, m.snd_idx, m.snd_val, m.sid, m.sid or sender, m.cluster,
                       m.text))
    return out


class Memory:
    """Vectorized near-duplicate memory with the same rules as spamcore's LabeledMemory (a sender is a sid, or a
    simulated one)."""

    def __init__(self, tau_spam: float, tau_ham: float):
        self.tau_spam, self.tau_ham = tau_spam, tau_ham
        self.rows, self.labels, self.senders = [], [], []
        self.M = None

    def add(self, idx, val, label, sender):
        v = val.astype(np.float64)
        n = np.linalg.norm(v)
        self.rows.append(sp.csr_matrix((v / (n or 1.0), idx, [0, len(idx)]), shape=(1, 1 << 18)))
        self.labels.append(label)
        self.senders.append(sender)
        self.M = sp.vstack(self.rows, format="csr")

    def match(self, idx, val, sender):
        if self.M is None:
            return None
        v = val.astype(np.float64)
        n = np.linalg.norm(v)
        if n == 0:
            return None
        q = sp.csr_matrix((v / n, idx, [0, len(idx)]), shape=(1, 1 << 18))
        cos = (self.M @ q.T).toarray().ravel()
        lab = np.asarray(self.labels)
        best = None
        spam = np.where((lab == 1) & (cos >= self.tau_spam))[0]
        if len(spam):
            j = spam[np.argmax(cos[spam])]
            best = (1, j, cos[j])
        # Domain bookkeeping is skipped here: ham matches require the same sender.
        ham = [j for j in np.where((lab == 0) & (cos >= self.tau_ham))[0] if self.senders[j] == sender]
        if ham:
            j = max(ham, key=lambda k: cos[k])
            if best is None or cos[j] >= best[2]:
                best = (0, j, cos[j])
        return best


def z_of(head: Head, m: Msg) -> float:
    return head.z(m.emb, m.meta, m.idx, m.val, m.snd_idx, m.snd_val)


def priors_of(params: dict) -> Priors:
    return Priors(params["embed"], params["meta"], params["hash"], params["snd"])


def example_of(m: Msg, label: int, key: int) -> Example:
    return Example(m.emb, m.meta, m.idx, m.val, m.snd_idx, m.snd_val, label, key=key)


def probe_fpr(head: Head, probes: list[Msg], t: float) -> float:
    return float(np.mean([1 / (1 + np.exp(-z_of(head, p))) >= t for p in probes]))


def simulate(args) -> dict:
    (cfg_key, params, user, seed, stream, probes, base, thr) = args
    rng = np.random.default_rng(seed + 7)
    priors = priors_of(params)
    weighting = Weighting(ham=params["ham"])
    mem = Memory(params["tau_spam"], 0.90)
    head, examples = base, []
    t_bal = thr.balanced
    probe_base = [1 / (1 + np.exp(-z_of(base, p))) >= t_bal for p in probes]
    cost, errs, marks = 0, [], 0
    repeat_errs, seen = [], set()
    learn = user != "none"
    for k, m in enumerate(stream):
        is_repeat = m.cluster in seen
        seen.add(m.cluster)
        match = mem.match(m.idx, m.val, m.sender)
        p = 1 / (1 + np.exp(-z_of(head, m)))
        otp_no_url = m.meta[OTP] == 1.0 and m.meta[URLS] == 0.0
        label, _ = decide(memory_match=match, p=p, otp_no_url=otp_no_url, thresholds=thr)
        filtered = label == SPAM
        fp, fn = filtered and m.y == 0, (not filtered) and m.y == 1
        cost += 5 * fp + fn
        errs.append(fp or fn)
        if is_repeat:
            repeat_errs.append(fp or fn)
        if not learn:
            continue
        mark = None
        if fn:
            mark = 1
        elif fp and user in ("diligent", "noisy") and rng.random() < 0.3:
            mark = 0
        if mark is not None and user == "noisy" and rng.random() < 0.05:
            mark = 1 - mark
        if mark is None:
            continue
        examples.append(example_of(m, mark, k))
        mem.add(m.idx, m.val, mark, m.sender)
        marks += 1
        head = personalize(base, examples, example_weights(examples, weighting), priors, tol=1e-6, max_iter=200)
    n = len(errs)
    thirds = [float(np.mean(errs[i * n // 3:(i + 1) * n // 3])) for i in range(3)]
    return {"cfg": cfg_key, "user": user, "seed": seed, "cost": int(cost), "marks": marks, "thirds": thirds,
            "repeat_error": float(np.mean(repeat_errs)) if repeat_errs else 0.0, "n_repeat": len(repeat_errs),
            "probe_fpr_delta": probe_fpr(head, probes, t_bal) - float(np.mean(probe_base))}


def missed_campaign(args) -> tuple[int, int, int, int]:
    """One campaign the base model misses, replayed alone: the user marks every miss. Returns (later messages,
    caught with learning, caught by the base model, marks)."""
    msgs, params, base, thr = args
    priors = priors_of(params)
    weighting = Weighting(ham=params["ham"])
    mem = Memory(params["tau_spam"], 0.90)
    head, examples, marks, later, caught, caught_base = base, [], 0, 0, 0, 0
    for k, m in enumerate(msgs):
        otp_no_url = m.meta[OTP] == 1.0 and m.meta[URLS] == 0.0
        label, _ = decide(memory_match=mem.match(m.idx, m.val, m.sender), p=1 / (1 + np.exp(-z_of(head, m))),
                          otp_no_url=otp_no_url, thresholds=thr)
        base_label, _ = decide(memory_match=None, p=1 / (1 + np.exp(-z_of(base, m))), otp_no_url=otp_no_url, thresholds=thr)
        if marks > 0:
            later += 1
            caught += label == SPAM
            caught_base += base_label == SPAM
        if label != SPAM:
            examples.append(example_of(m, 1, k))
            mem.add(m.idx, m.val, 1, m.sender)
            marks += 1
            head = personalize(base, examples, example_weights(examples, weighting), priors, tol=1e-6, max_iter=200)
    return later, caught, caught_base, marks


def variant_campaign(args) -> tuple[int, int, int]:
    """A message the base model gets wrong plus its variants, as one sender campaign. The user corrects every
    mistake. Returns (variants, right with learning, right with the base model)."""
    msgs, params, base, thr = args
    priors = priors_of(params)
    weighting = Weighting(ham=params["ham"])
    mem = Memory(params["tau_spam"], 0.90)
    head, examples, n, right, right_base = base, [], 0, 0, 0
    for k, m in enumerate(msgs):
        otp_no_url = m.meta[OTP] == 1.0 and m.meta[URLS] == 0.0
        label, _ = decide(memory_match=mem.match(m.idx, m.val, m.sender), p=1 / (1 + np.exp(-z_of(head, m))),
                          otp_no_url=otp_no_url, thresholds=thr)
        base_label, _ = decide(memory_match=None, p=1 / (1 + np.exp(-z_of(base, m))), otp_no_url=otp_no_url, thresholds=thr)
        if k > 0:
            n += 1
            right += (label == SPAM) == (m.y == 1)
            right_base += (base_label == SPAM) == (m.y == 1)
        if (label == SPAM) != (m.y == 1):
            examples.append(example_of(m, m.y, k))
            mem.add(m.idx, m.val, m.y, m.sender)
            head = personalize(base, examples, example_weights(examples, weighting), priors, tol=1e-6, max_iter=200)
    return n, right, right_base


def variant_scenarios(pool, params, base, thr, n_variants: int = 4, cap: int = 300) -> dict:
    """Repeats the user would notice: scams resent with new names/amounts/links from new numbers, and a legit
    sender's recurring notices. Variants come from smsml.augment and are embedded with the int8 encoder. A recurring
    notice keeps the seed's sender (and so its SND vector, when the seed has one); a new number has no learned sender
    weights, so a resent scam's variants have no SND vector."""
    import random

    import onnxruntime as ort

    from smsml.augment import augment
    from smsml.normalize import normalize
    from smsml.pipeline import config, snapshot_dir
    from smsml.tokenizer import WordPiece

    wp = WordPiece.from_file(snapshot_dir(config()) / "vocab.txt")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(str(BUNDLE_WORK / "encoder.int8.ort"), so, providers=["CPUExecutionProvider"])
    r = random.Random(7)
    p_base = lambda m: 1 / (1 + np.exp(-z_of(base, m)))  # noqa: E731
    missed = [m for m in pool if m.y == 1 and p_base(m) < thr.balanced][:cap]
    flagged = [m for m in pool if m.y == 0 and p_base(m) >= thr.balanced][:cap]
    out = {}
    for name, seeds, same_sender in (("missed_spam", missed, False), ("flagged_ham", flagged, True)):
        texts = [[augment(m.text, r) for _ in range(n_variants)] for m in seeds]
        flat = [t for ts in texts for t in ts]
        if not flat:
            out[name] = {"seeds": 0}
            continue
        # A sid is its own sender key, so featurize maps it back to the same sid.
        senders = [m.sid if same_sender else None for m in seeds for _ in range(n_variants)]
        X, S, meta = featurize(flat, base.idf, senders)
        embs = []
        for t in flat:
            ids = np.asarray([wp.encode(normalize(t).enc)], dtype=np.int64)
            embs.append(sess.run(["embedding"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0])
        jobs, j = [], 0
        for c, m in enumerate(seeds):
            own = m.sid or f"s{c}"
            camp = [Msg(m.y, m.emb, m.meta, m.idx, m.val, m.snd_idx, m.snd_val, m.sid, own, f"v{c}", m.text)]
            for _ in range(n_variants):
                sid, sender = (m.sid, own) if same_sender else (None, f"n{c}-{j}")
                camp.append(Msg(m.y, embs[j], meta[j], X[j].indices, X[j].data, S[j].indices, S[j].data, sid, sender,
                                f"v{c}", flat[j]))
                j += 1
            jobs.append((camp, params, base, thr))
        with ProcessPoolExecutor(WORKERS) as ex:
            res = list(ex.map(variant_campaign, jobs))
        n = sum(x[0] for x in res)
        out[name] = {"seeds": len(seeds), "variants": n, "right_with_learning": sum(x[1] for x in res) / n,
                     "right_with_base_model": sum(x[2] for x in res) / n}
    return out


def confirm(pool, probes, base, thr, override: dict | None = None, write: bool = True) -> int:
    pers = json.loads(work("personalization.json").read_text())
    params = {**pers["best"]["params"], **(override or {})}
    seeds = [11, 12, 13, 14, 15, 16, 17, 18]
    streams = {s: build_stream(s, pool) for s in seeds}
    jobs = [("best", params, user, s, streams[s], probes, base, thr)
            for user in ("diligent", "spam_only", "noisy", "none") for s in seeds]
    with ProcessPoolExecutor(WORKERS) as ex:
        results = list(ex.map(simulate, jobs))

    def mean(user, field):
        return float(np.mean([r[field] for r in results if r["user"] == user]))

    def thirds(user):
        return np.mean([r["thirds"] for r in results if r["user"] == user], axis=0).tolist()

    out = {"params": params, "seeds": seeds}
    for user in ("diligent", "spam_only", "noisy", "none"):
        out[user] = {"cost": mean(user, "cost"), "thirds": thirds(user), "repeat_error": mean(user, "repeat_error"),
                     "probe_fpr_delta": mean(user, "probe_fpr_delta"), "marks": mean(user, "marks"),
                     "n_repeat": mean(user, "n_repeat")}
    b = out["none"]
    for user in ("diligent", "spam_only", "noisy"):
        u = out[user]
        u["cost_ratio"] = u["cost"] / b["cost"]
        u["last_third_ratio"] = u["thirds"][2] / max(1e-9, b["thirds"][2])
        u["repeat_error_ratio"] = u["repeat_error"] / max(1e-9, b["repeat_error"])
    # New campaigns the base model misses: how many of their later messages are caught once the user marks misses.
    by_cluster: dict[str, list[Msg]] = {}
    for m in pool:
        if m.y == 1:
            by_cluster.setdefault(m.cluster, []).append(m)
    hard = []
    for msgs in by_cluster.values():
        if len(msgs) < 3:
            continue
        missed = [1 / (1 + np.exp(-z_of(base, m))) < thr.balanced for m in msgs]
        if any(missed):
            hard.append(msgs[:12])
    with ProcessPoolExecutor(WORKERS) as ex:
        camp = list(ex.map(missed_campaign, [(msgs, params, base, thr) for msgs in hard]))
    later = sum(c[0] for c in camp)
    out["missed_campaigns"] = {
        "campaigns": len(hard), "later_messages": later,
        "caught_with_learning": sum(c[1] for c in camp) / max(1, later),
        "caught_by_base_model": sum(c[2] for c in camp) / max(1, later),
        "marks_per_campaign": float(np.mean([c[3] for c in camp])) if camp else 0.0,
    }
    mc = out["missed_campaigns"]
    print(f"missed campaigns: {mc['campaigns']} campaigns, {later} later messages: caught {mc['caught_with_learning']:.0%} "
          f"with learning vs {mc['caught_by_base_model']:.0%} without; {mc['marks_per_campaign']:.1f} marks per campaign")
    out["variants"] = variant_scenarios(pool, params, base, thr)
    for name, v in out["variants"].items():
        if v.get("seeds"):
            print(f"{name}: {v['seeds']} messages x 4 variants: variants handled right {v['right_with_learning']:.0%} "
                  f"with learning vs {v['right_with_base_model']:.0%} with the base model")
    if write:
        if override:
            pers["best"]["params"] = params
        pers["confirm"] = out
        write_json(work("personalization.json"), pers)
    for user in ("diligent", "spam_only", "noisy"):
        u = out[user]
        print(f"{user:9s} cost {u['cost']:.1f} ({u['cost_ratio']:.0%} of baseline {b['cost']:.1f}); "
              f"repeat-campaign error {u['repeat_error']:.4f} vs {b['repeat_error']:.4f} ({u['repeat_error_ratio']:.0%}); "
              f"last third {u['last_third_ratio']:.0%}; probe FPR {u['probe_fpr_delta']:+.4f}; marks {u['marks']:.0f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--confirm", action="store_true", help="re-run the chosen settings on 8 fresh streams")
    ap.add_argument("--override", type=json.loads, default=None, help="with --confirm: JSON params to try instead")
    ap.add_argument("--no-write", action="store_true", help="with --confirm: print only")
    ap.add_argument("--traffic", choices=["mixed", "transactional"], default="transactional",
                    help="legit messages in the streams: transactional only (no personal chat or first contacts), or all")
    args = ap.parse_args()
    cfg = config()
    base = Head.load(BUNDLE_WORK / "head_base.bin")
    cal = json.loads(work("calibration.json").read_text())
    thr = Thresholds(**cal["thresholds"]["full"])

    df = pd.read_parquet(work("corpus_split.parquet"))
    rows = np.load(work("emb_corpus_rows.npy"))
    emb = np.load(work("emb_corpus.npy"))
    X = sp.load_npz(work("features_X.npz"))
    S = sp.load_npz(work("features_S.npz"))
    meta = np.load(work("features_meta.npy"))
    sub = df.iloc[rows].reset_index(drop=True)
    sids = [sender_id(sender_key(s)) for s in senders_of(sub)]
    # Noisy labels would make even a diligent simulated user mark wrongly.
    pool_mask = (sub["split"].isin(["val_a", "val_b"]) & ~sub["source"].isin(cfg["noisy_sources"])).to_numpy()
    if args.traffic == "transactional":
        chat = sub["source"].isin(["uci", "mishra", "iiitd_smsassassin"]) | ((sub["source"] == "cube_sms77k") & (sub["sender_type"] == "PHONE"))
        personal = (sub["label"] == 0) & (chat | sub["category"].isin(PERSONAL))
        pool_mask = pool_mask & ~personal.to_numpy()
    pool = [Msg(int(sub["label"][i]), emb[i], meta[i], X[i].indices, X[i].data, S[i].indices, S[i].data, sids[i], "",
                str(sub["cluster_id"][i]), str(sub["text"][i])) for i in np.flatnonzero(pool_mask)]
    modern = load_modern_eval()
    dev = modern["split"] == "dev"
    em = np.load(work("emb_modern.npy"))
    modern_senders = senders_of(modern[dev])
    Xm, Sm, mm = featurize(modern.loc[dev, "text"].tolist(), base.idf, modern_senders)
    for j, (i, row) in enumerate(modern[dev].iterrows()):
        if args.traffic == "transactional" and row["label"] == 0 and row["category"] in PERSONAL:
            continue
        mj = mm[j].copy()
        if row["sender_type"] in KIND:
            mj[KIND[row["sender_type"]]] = 1.0
        pool.append(Msg(int(row["label"]), em[i], mj, Xm[j].indices, Xm[j].data, Sm[j].indices, Sm[j].data,
                        sender_id(sender_key(modern_senders[j])), "", f"modern-{i}", row["text"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    hams = [m for m in pool if m.y == 0]
    probe_idx = rng.choice(len(hams), size=min(260, len(hams) // 4), replace=False)  # drift measurement only
    probes = [hams[i] for i in probe_idx]
    probe_ids = {id(p) for p in probes}
    pool = [m for m in pool if id(m) not in probe_ids]

    pc = cfg["personalization"]
    if args.confirm:
        return confirm(pool, probes, base, thr, override=args.override, write=not args.no_write)
    # FROZEN as a prior strength pins a block at its base weights. Learning the dense blocks (embedding, message
    # features) from a few mostly-spam marks shifts every message's score; the sparse n-gram weights only touch
    # messages that share the marked messages' n-grams, and the SND weights only those that also share their sender.
    if args.quick:
        grid = {"embed": [FROZEN], "meta": [FROZEN], "hash": [pc["priors"]["hash"]], "snd": [pc["priors"]["snd"]],
                "ham": [1.0], "tau_spam": [0.85]}
        seeds = [1]
    else:
        grid = {"embed": [1.0, FROZEN], "meta": [4.0, FROZEN], "hash": [0.15, 0.25, 0.5, 1.0],
                "snd": [0.1, 0.5, FROZEN], "ham": [1.0], "tau_spam": [0.80, 0.90]}
        seeds = [1, 2, 3]
    streams = {s: build_stream(s, pool) for s in seeds}
    jobs = []
    configs = []
    for emb_p, meta_p, h, snd, ham, tau in itertools.product(*grid.values()):
        params = {"embed": emb_p, "meta": meta_p, "hash": h, "snd": snd, "ham": ham, "tau_spam": tau}
        key = json.dumps(params, sort_keys=True)
        configs.append((key, params))
        for user in ("diligent", "spam_only", "noisy"):
            for s in seeds:
                jobs.append((key, params, user, s, streams[s], probes, base, thr))
    for s in seeds:  # no-learning baseline
        jobs.append(("baseline", configs[0][1], "none", s, streams[s], probes, base, thr))
    with ProcessPoolExecutor(WORKERS) as ex:
        results = list(ex.map(simulate, jobs))

    base_cost = np.mean([r["cost"] for r in results if r["cfg"] == "baseline"])
    base_thirds = np.mean([r["thirds"] for r in results if r["cfg"] == "baseline"], axis=0).tolist()
    table = []
    for key, params in configs:
        rs = [r for r in results if r["cfg"] == key]

        def agg(user, field):
            return float(np.mean([r[field] for r in rs if r["user"] == user]))

        dil_thirds = np.mean([r["thirds"] for r in rs if r["user"] == "diligent"], axis=0).tolist()
        table.append({
            "params": params,
            "cost_diligent": agg("diligent", "cost"), "cost_spam_only": agg("spam_only", "cost"),
            "cost_noisy": agg("noisy", "cost"), "marks_diligent": agg("diligent", "marks"),
            "probe_fpr_delta_spam_only": agg("spam_only", "probe_fpr_delta"),
            "diligent_thirds": dil_thirds,
        })
    ok = [t for t in table if t["probe_fpr_delta_spam_only"] <= 0.003]
    best = min(ok or table, key=lambda t: t["cost_diligent"])
    out = {"baseline_cost": float(base_cost), "baseline_thirds": base_thirds, "best": best,
           "table": sorted(table, key=lambda t: t["cost_diligent"])}
    write_json(work("personalization.json"), out)
    print(f"baseline thirds {base_thirds}")
    print(f"baseline cost {base_cost:.0f}; best diligent cost {best['cost_diligent']:.0f} "
          f"({best['cost_diligent'] / base_cost:.2%} of baseline); thirds {best['diligent_thirds']}; "
          f"spam-only probe FPR delta {best['probe_fpr_delta_spam_only']:+.4f}")
    print("best params:", best["params"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
