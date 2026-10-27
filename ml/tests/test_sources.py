"""Tests for the data-side loaders, templates, refill, and the frozen modern eval set."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from smsml.sources import RAW_DIR, imc25, lang, mishra, ncsu, refill, schema, templates, uci

ML = Path(__file__).resolve().parents[1]
TEMPLATES = ML / "templates" / "ham_v1.yaml"
EVAL = ML / "eval" / "modern_v1.jsonl"
RES = ML / "resources"


def expand(seed=20261027, total=3400, cap=40):
    return templates.load(TEMPLATES, seed=seed, target_total=total, max_per_template=cap)


# ── schema ──────────────────────────────────────────────────────────────────────────────────────────

def test_finalize_orders_columns_and_types():
    df = schema.finalize(pd.DataFrame([{"text": "hi", "label": 0, "source": "x", "source_id": "1"}]))
    assert list(df.columns) == schema.COLUMNS
    assert str(df["label"].dtype) == "int8"
    assert str(df["timestamp"].dtype).startswith("datetime64")
    assert pd.isna(df.loc[0, "campaign_id"])


@pytest.mark.parametrize("row", [
    {"text": "", "label": 0, "source": "x", "source_id": "1"},
    {"text": "a", "label": 2, "source": "x", "source_id": "1"},
    {"text": "a", "label": 0, "source": "x", "source_id": "1", "sender_type": "FAX"},
])
def test_finalize_rejects_bad_rows(row):
    with pytest.raises(ValueError):
        schema.finalize(pd.DataFrame([row]))


def test_finalize_rejects_duplicate_ids():
    rows = [{"text": "a", "label": 0, "source": "x", "source_id": "1"}] * 2
    with pytest.raises(ValueError):
        schema.finalize(pd.DataFrame(rows))


# ── templates ───────────────────────────────────────────────────────────────────────────────────────

def test_template_expansion_is_deterministic_and_ham():
    a, b = expand(), expand()
    pd.testing.assert_frame_equal(a, b)
    assert (a["label"] == 0).all() and (a["source"] == "templates").all()
    assert a["template_id"].notna().all() and a["category"].notna().all()
    assert 2500 <= len(a) <= 3500
    assert not a["text"].str.contains("{", regex=False).any()
    assert not a["text"].str.contains("}", regex=False).any()


def test_every_template_expands_and_ids_unique():
    doc = templates.load_templates(TEMPLATES)
    df = expand()
    assert set(df["template_id"]) == {t["id"] for t in doc["templates"]}
    assert len(doc["templates"]) >= 150


def test_seed_changes_output_but_one_template_is_isolated():
    a, b = expand(seed=1), expand(seed=2)
    assert set(a["text"]) != set(b["text"])
    # Per-template RNGs: dropping other templates doesn't change a template's expansions.
    doc = templates.load_templates(TEMPLATES)
    only = dict(doc, templates=[t for t in doc["templates"] if t["id"] == "otp_google"])
    solo = templates.expand_templates(only, seed=1, target_total=3400, max_per_template=40)
    full = a[a["template_id"] == "otp_google"]["text"].tolist()
    # Alone, the template gets a larger share, so its expansions extend the full run's list as a prefix.
    assert len(solo) >= len(full) and solo["text"].tolist()[: len(full)] == full


def test_bound_variables_repeat_the_same_value():
    df = expand()
    for text in df[df["template_id"] == "otp_google_webotp"]["text"]:
        code = text.split("is ")[1].split()[0]
        assert text.endswith(f"#{code}")


def test_expander_grammar():
    import random
    e = templates.Expander({"x": ["A", "B"]}, None, random.Random(0))
    out = e.expand("{x} {d4} {n:3-3} {p|q} {{x}|z} {d6@k}/{d6@k}")
    parts = out.split()
    assert parts[0] in "AB" and len(parts[1]) == 4 and parts[1].isdigit() and parts[2] == "3"
    assert parts[3] in ("p", "q") and parts[4] in ("A", "B", "z")
    left, right = parts[5].split("/")
    assert left == right and len(left) == 6
    with pytest.raises(KeyError):
        e.expand("{nope}")
    with pytest.raises(ValueError):
        e.expand("{x")


# ── refill ──────────────────────────────────────────────────────────────────────────────────────────

def test_imc_refill_deterministic_and_content_seeded():
    text = "Dear <NAMED_ENTITY>, your <NAMED_ENTITY> account is locked until <DATE_TIME>. Visit <URL> or call <PHONE_NUMBER>"
    a = refill.refill_imc(text, 7, hosts=["evil.example"], named_entity="PayPal")
    assert a == refill.refill_imc(text, 7, hosts=["evil.example"], named_entity="PayPal")
    assert "<" not in a
    assert "PayPal account" in a  # brand context gets the row's impersonated brand
    assert a.split(",")[0].split()[-1] in refill.FIRST_NAMES  # "Dear <NAMED_ENTITY>" gets a person name
    assert "555-01" in a or "55501" in a
    assert refill.refill_imc(text, 8, hosts=["evil.example"], named_entity="PayPal") != a


def test_imc_refill_uses_shortener_once_and_keeps_unknown_tags():
    out = refill.refill_imc("<ADV> go <URL> then <URL>", 1, hosts=["h.example"], shortener="bit.ly")
    assert out.startswith("<ADV> go https://bit.ly/")
    assert out.count("bit.ly") == 1 and "h.example" in out


def test_uci_decode_removes_placeholders():
    raw = "I'll be there in  &lt;#&gt;  mins at &lt;TIME&gt; &amp; bring &lt;DECIMAL&gt; kg"
    out = uci.decode_uci_text(raw, 3)
    assert "&" not in out.replace(" & ", "") and "<" not in out
    assert out == uci.decode_uci_text(raw, 3)


# ── per-source helpers ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("Your package is waiting, click the link to confirm your address", True),
    ("New COIN~Login || www.usdtjerc.com || Username:Rom88 P***word:R66888 Balance:3,862,587.65", True),
    ("Ηello аll, guуsǃ I knоw, my mesѕаge maу bе toо sрeсifіc", True),
    ("Su paquete no pudo ser entregado, actualice su direccion aqui", False),
    ("Uw pakket kon niet worden bezorgd, klik hier", False),
    ("[BN] 地 被限制使用, 立即更新: binenae.com", False),
])
def test_is_english(text, expected):
    assert lang.is_english(text) is expected


@pytest.mark.parametrize("sender,kind", [
    ("+16289992825", "PHONE"), ("+1818355XXXX", "PHONE"), ("72975", "SHORT_CODE"), ("Google", "ALPHANUMERIC"),
    ("a@b.com", "EMAIL"), ("\n", None), ("", None),
])
def test_ncsu_sender_type(sender, kind):
    assert ncsu.sender_type(sender) == kind


def test_campaign_matcher_prefers_window_and_capacity():
    c = {
        1: ncsu.Campaign(1, ncsu.words("pay your toll now at"), 0, 100, 1, ("toll.example/a",)),
        2: ncsu.Campaign(2, ncsu.words("pay your toll now at"), 200, 300, 2, ("toll.example/a",)),
    }
    m = ncsu.CampaignMatcher(c)
    remaining = {1: 1, 2: 2}
    assert m.match("Pay your toll now at https://toll.example/a", 250, 0, remaining) == 2
    assert m.match("Pay your toll now at toll.example/a", 50, 0, remaining) == 1
    assert m.match("Totally unrelated words here", 50, 0, remaining) is None


@pytest.mark.parametrize("value,expected", [
    ("2023-05-19", "2023-05-19"), ("Monday, Aug 16, 2021 • 5:47 PM", "2021-08-16"), ("19/05/2023", "2023-05-19"),
    ("05/19/2023", "2023-05-19"), ("04/05/2023", None), ("Yesterday 22:50", None), ("Fri, Apr 14 at 5:59 AM", None),
])
def test_parse_imc_time(value, expected):
    ts = imc25.parse_imc_time(value)
    assert (pd.isna(ts) if expected is None else ts.strftime("%Y-%m-%d") == expected)


def test_split_bubbles():
    assert imc25.split_bubbles("['a b', 'c']") == ["a b", "c"]
    assert imc25.split_bubbles("[ I.R.S ] refund") == ["[ I.R.S ] refund"]
    assert imc25.split_bubbles("['broken") == ["['broken"]


# ── loaders on the real raw files (skipped until s00_fetch has run) ────────────────────────────────

needs_raw = pytest.mark.skipif(not (RAW_DIR / "MANIFEST.json").exists(), reason="run scripts/s00_fetch.py first")


@needs_raw
def test_uci_loader():
    df = uci.load(RAW_DIR, 20261027)
    assert len(df) == 5574 and Counter(df["label"]) == {0: 4827, 1: 747}
    assert not df["text"].str.contains("<#>", regex=False).any()


@needs_raw
def test_mishra_loader():
    df = mishra.load(RAW_DIR, 20261027)
    assert len(df) == 5971 and Counter(df["label"]) == {0: 4844, 1: 1127}
    assert set(df["category"].dropna()) == {"spam", "smishing"}


# ── frozen modern eval set ──────────────────────────────────────────────────────────────────────────

def test_modern_eval_file():
    rows = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 400
    assert {tuple(sorted(r)) for r in rows} == {("category", "id", "label", "sender", "sender_type", "split", "text")}
    assert len({r["id"] for r in rows}) == 400 and len({r["text"] for r in rows}) == 400
    assert Counter((r["label"], r["split"]) for r in rows) == {(1, "dev"): 100, (1, "test"): 100,
                                                                (0, "dev"): 100, (0, "test"): 100}
    per_group = Counter((r["label"], r["category"], r["split"]) for r in rows)
    for label, cat in {(r["label"], r["category"]) for r in rows}:
        assert abs(per_group[(label, cat, "dev")] - per_group[(label, cat, "test")]) <= 1
    for r in rows:
        assert r["sender_type"] in schema.SENDER_TYPES
        if r["sender_type"] == "PHONE":
            # fictional: +1 555-01xx, other countries' 555-01xx tails, or the UK drama range 07700 900xxx
            assert r["sender"].startswith("+") and ("55501" in r["sender"] or r["sender"].startswith("+44770090"))
        if r["sender_type"] == "SHORT_CODE":
            assert r["sender"].isdigit() and 5 <= len(r["sender"]) <= 6
        if r["sender_type"] == "EMAIL":
            assert "@" in r["sender"]


# ── resource lists ──────────────────────────────────────────────────────────────────────────────────

def _entries(name):
    lines = (RES / name).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


@pytest.mark.parametrize("name", ["shorteners_v1.txt", "risky_tlds_v1.txt"])
def test_lists_are_lowercase_unique(name):
    items = _entries(name)
    assert items and len(items) == len(set(items))
    assert all(i == i.lower() and " " not in i for i in items)


def test_risky_tlds_exclude_mainstream():
    assert not {"com", "net", "org", "gov", "edu", "us", "io", "co", "app", "me"} & set(_entries("risky_tlds_v1.txt"))


def test_brands_table():
    rows = [line.split("\t") for line in _entries("brands_v1.tsv")]
    assert all(len(r) in (2, 3) for r in rows)
    assert all(r[2] in ("sub", "word") for r in rows if len(r) == 3)  # optional match mode (SPEC brand_lookalike)
    keys = [r[0] for r in rows]
    assert len(keys) == len(set(keys)) and all(k == k.lower() for k in keys)
    for r in rows:
        assert all("." in d and d == d.lower() for d in r[1].split(","))
