"""Tests for the corpus build (s01) and the split rules (s02): a source's holdout never reaches anything else."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from smsml.pipeline import config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import s01_corpus  # noqa: E402
import s02_split  # noqa: E402

# s01 hashes in a fork-based process pool; Python 3.12 warns when it forks while the pool's own thread runs.
pytestmark = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")


def normalized(rows: list[dict]) -> pd.DataFrame:
    """Rows as s01 sees them after normalization (only the columns it reads)."""
    df = pd.DataFrame(rows)
    df["clean"] = df["ng"]
    return df


POOL = [
    {"source_id": "p1", "ng": "rs 000 spent on card x0000 at shop not you call 0000", "label": 0, "campaign_id": "inbox:1"},
    {"source_id": "p2", "ng": "your otp for login is 000000 do not share it", "label": 0, "campaign_id": "inbox:2"},
    {"source_id": "p3", "ng": "get 00% off on every plan today only", "label": 1, "campaign_id": "inbox:3"},
    {"source_id": "p4", "ng": "your otp for login is 000000 do not share it", "label": 0, "campaign_id": None},
]
HELD = [  # placed first, so processed together with the pool they'd take its rows' places
    {"source_id": "h1", "ng": POOL[1]["ng"], "label": 0, "campaign_id": "inbox:2"},  # an exact duplicate of p2
    {"source_id": "h2", "ng": POOL[1]["ng"], "label": 0, "campaign_id": "inbox:2"},  # ...repeated
    {"source_id": "h3", "ng": POOL[2]["ng"], "label": 0, "campaign_id": "inbox:3"},  # p3's text, the other label
    {"source_id": "h4", "ng": "your card x0000 statement is ready", "label": 0, "campaign_id": "inbox:1"},  # p1's template
]


def test_holdout_rows_never_touch_the_rest():
    alone = s01_corpus.build(normalized([{**r, "holdout": None} for r in POOL]), 0.7)
    rows = [{**r, "holdout": "inbox_eval"} for r in HELD] + [{**r, "holdout": None} for r in POOL]
    both = s01_corpus.build(normalized(rows), 0.7)
    rest, held = both[both["holdout"].isna()].reset_index(drop=True), both[both["holdout"].notna()]
    pd.testing.assert_frame_equal(rest.drop(columns="holdout"), alone.drop(columns="holdout"))
    assert rest["source_id"].tolist() == ["p1", "p2", "p3"]  # p4 duplicates p2
    assert held["source_id"].tolist() == ["h1", "h2", "h3", "h4"]  # as received, repeats and conflicts included
    assert held["cluster_id"].str.startswith("h").all() and not set(held["cluster_id"]) & set(rest["cluster_id"])
    # Processed as ordinary rows, they would have dropped two of the pool's rows.
    ordinary = s01_corpus.build(normalized([{**r, "holdout": None} for r in HELD + POOL]), 0.7)
    assert {"p2", "p3"}.isdisjoint(ordinary["source_id"])


def split_frame() -> pd.DataFrame:
    """A corpus as s01 leaves it: public rows, the inbox's oldest half, and its newest half held out."""
    rows = []
    t0 = pd.Timestamp("2022-05-01", tz="UTC")
    for k in range(60):  # timestamped smishing campaigns, two messages each
        rows += [{"source": "ncsu", "label": 1, "cluster_id": f"c{k}-1", "timestamp": t0 + pd.Timedelta(days=5 * k + j)}
                 for j in range(2)]
    rows += [{"source": "uci", "label": k % 2, "cluster_id": f"c{100 + k}-{k % 2}"} for k in range(80)]
    rows += [{"source": "templates", "label": 0, "cluster_id": f"c{200 + k}-0", "template_id": f"t{k}"}
             for k in range(8) for _ in range(3)]
    t1 = pd.Timestamp("2025-06-01", tz="UTC")  # the inbox is newer than every public message
    for k in range(80):
        label = int(k % 4 == 0)
        rows.append({"source": "inbox", "label": label, "cluster_id": f"c{300 + k // 2}-{label}",
                     "timestamp": t1 + pd.Timedelta(days=k)})
        rows.append({"source": "inbox", "label": label, "cluster_id": f"h{k // 2}-{label}",
                     "timestamp": t1 + pd.Timedelta(days=200 + k), "holdout": "inbox_eval"})
    # One big template cluster of the inbox's (the per-cluster cap would keep 6 of these) ...
    rows += [{"source": "inbox", "label": 0, "cluster_id": "c400-0", "timestamp": t1 + pd.Timedelta(hours=k)}
             for k in range(30)]
    # ... and an inbox message s01 clustered with the newest NCSU campaign: that cluster trains, so it isn't future.
    rows.append({"source": "inbox", "label": 1, "cluster_id": "c59-1", "timestamp": t1})
    return frame(rows)


def frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["source_id"] = [str(i) for i in range(len(df))]
    for col in ("holdout", "template_id"):
        if col not in df:
            df[col] = None
    return df


def split(df: pd.DataFrame, train_only: frozenset[str] = frozenset()) -> pd.DataFrame:
    return s02_split.assign_splits(df, config()["split"], 1234, train_only)


def test_inbox_split_rule():
    sc = config()["split"]
    out = split(split_frame())
    inbox, held = out["source"] == "inbox", out["holdout"].notna()
    assert ((out["split"] == "inbox_eval") == held).all()
    # The oldest half trains by time: its newest `time_val` validate, both halves, cluster by cluster; none is capped.
    own = out[inbox & ~held].sort_values(["timestamp", "source_id"])
    n_val = round(sc["time_val"] * len(own))
    assert (own["split"].iloc[:-n_val] == "train").all()
    late = own.iloc[-n_val:]
    assert set(late["split"]) == {"val_a", "val_b"}
    assert not set(late.loc[late["split"] == "val_a", "cluster_id"]) & set(late.loc[late["split"] == "val_b", "cluster_id"])
    assert (out.loc[out["cluster_id"] == "c400-0", "split"] == "train").all()
    # The future holdout is the public campaigns' own: the inbox neither joins it nor moves its cutoff, and the NCSU
    # campaign clustered with an inbox message trains with it.
    future = out["split"] == "future"
    assert not (future & inbox).any() and set(out.loc[out["cluster_id"] == "c59-1", "split"]) == {"train"}
    public = out[(out["source"] == "ncsu") & (out["cluster_id"] != "c59-1")]
    start = public.groupby("cluster_id")["timestamp"].min()
    expected = set(start[start >= public["timestamp"].quantile(1 - sc["future_frac"])].index)
    assert set(out.loc[future, "cluster_id"]) == expected and expected
    assert s02_split.leaking_clusters(out) == []


def test_template_near_duplicates_follow_the_template():
    df = split_frame()
    df.loc[df["source"] == "uci", "cluster_id"] = "c200-0"  # every uci message is a near-duplicate of template t0
    df.loc[df["source"] == "uci", "label"] = 0
    out = split(df)
    assert out.loc[out["cluster_id"] == "c200-0", "split"].nunique() == 1
    assert s02_split.leaking_clusters(out) == []


def test_training_rows_take_their_near_duplicates_along():
    df = split_frame()
    synthetic = [{"source": "synth", "label": k % 2, "cluster_id": f"c{100 + k}-{k % 2}"} for k in range(10)]
    inbox_copy = {"source": "inbox", "label": 0, "cluster_id": "c201-0", "timestamp": pd.Timestamp("2025-06-02", tz="UTC")}
    out = split(frame(df.drop(columns="source_id").to_dict("records") + synthetic + [inbox_copy]), frozenset({"synth"}))
    assert set(out.loc[out["source"] == "synth", "split"]) == {"train"}
    # The uci messages clustered with synthetic ones train, and template t1 (clustered with an inbox one) trains whole.
    assert set(out.loc[out["cluster_id"].isin({r["cluster_id"] for r in synthetic}), "split"]) <= {"train", "capped"}
    assert set(out.loc[out["template_id"] == "t1", "split"]) == {"train"}
    assert s02_split.leaking_clusters(out) == []


def test_inbox_rows_count_toward_the_spam_ratio_but_are_never_capped():
    t1 = pd.Timestamp("2025-06-01", tz="UTC")
    rows = [{"source": "inbox", "label": 0, "cluster_id": f"c{k}-0", "timestamp": t1 + pd.Timedelta(days=k)}
            for k in range(10)]
    rows += [{"source": "uci", "label": 1, "cluster_id": f"c{100 + k}-1"} for k in range(40)]
    rows += [{"source": "synth", "label": 1, "cluster_id": f"c{200 + k}-1"} for k in range(60)]
    rows.append({"source": "inbox", "label": 0, "cluster_id": "h0-0", "timestamp": t1, "holdout": "inbox_eval"})
    out = split(frame(rows), frozenset({"synth"}))
    train = out["split"] == "train"
    inbox_ham = int((train & (out["source"] == "inbox")).sum())
    assert inbox_ham == 8 and "capped" not in set(out.loc[out["source"] == "inbox", "split"])
    assert int((train & (out["label"] == 1)).sum()) == config()["split"]["max_spam_ratio"] * inbox_ham


def test_leaking_clusters():
    rows = [
        ("inbox", None, "train", "c1"), ("inbox", None, "val_a", "c1"),  # the inbox's templates recur: fine
        ("inbox", None, "train", "c2"), ("cube_sms77k", None, "test", "c2"),  # a training row's near-copy in test
        ("uci", None, "train", "c3"), ("uci", None, "val_a", "c3"),
        ("uci", None, "val_a", "c4"), ("uci", None, "val_b", "c4"),
        ("uci", None, "train", "c5"), ("uci", None, "capped", "c5"),
        ("inbox", "inbox_eval", "inbox_eval", "h0"),
    ]
    df = pd.DataFrame(rows, columns=["source", "holdout", "split", "cluster_id"])
    assert s02_split.leaking_clusters(df) == ["c2", "c3"]
