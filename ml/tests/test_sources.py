"""Tests for the data-side loaders, templates, refill, and the frozen modern eval set."""

from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from smsml.sources import (RAW_DIR, SOURCES, cube_sms77k, hf_craftogrammer_txn, hf_ridham_synth_scam,
                           iiitd_smsassassin, imc25, inbox, junioralive_spam_ham, kaggle_rider_inbox, lang, load_all,
                           load_config, mishra, ncsu, refill, schema, templates, uci)

ML = Path(__file__).resolve().parents[1]
TEMPLATES = ML / "templates" / "ham_v1.yaml"
EVAL = ML / "eval" / "modern_v1.jsonl"
RES = ML / "resources"
INBOX_FILE = RAW_DIR / load_config()["inbox"]["file"]


def expand(seed=20261027, total=3400, cap=40):
    return templates.load(TEMPLATES, seed=seed, target_total=total, max_per_template=cap)


# ── schema ──────────────────────────────────────────────────────────────────────────────────────────

def test_finalize_orders_columns_and_types():
    df = schema.finalize(pd.DataFrame([{"text": "hi", "label": 0, "source": "x", "source_id": "1"}]))
    assert list(df.columns) == schema.COLUMNS
    assert str(df["label"].dtype) == "int8"
    assert str(df["timestamp"].dtype).startswith("datetime64")
    assert pd.isna(df.loc[0, "campaign_id"]) and pd.isna(df.loc[0, "sender"]) and pd.isna(df.loc[0, "holdout"])


def test_finalize_keeps_sender_and_holdout():
    df = schema.finalize(pd.DataFrame([
        {"text": "a", "label": 0, "source": "x", "source_id": "1", "sender": "AX-MYBANK-S",
         "sender_type": "ALPHANUMERIC", "holdout": "x_eval"},
        {"text": "b", "label": 1, "source": "x", "source_id": "2", "sender_type": "PHONE"},  # the kind alone (imc25)
    ]))
    assert df.loc[0, "sender"] == "AX-MYBANK-S" and df.loc[0, "holdout"] == "x_eval"
    assert pd.isna(df.loc[1, "sender"]) and df.loc[1, "sender_type"] == "PHONE" and pd.isna(df.loc[1, "holdout"])


@pytest.mark.parametrize("row", [
    {"text": "", "label": 0, "source": "x", "source_id": "1"},
    {"text": "a", "label": 2, "source": "x", "source_id": "1"},
    {"text": "a", "label": 0, "source": "x", "source_id": "1", "sender_type": "FAX"},
    {"text": "a", "label": 0, "source": "x", "source_id": "1", "sender": "AX-MYBANK-S", "sender_type": "PHONE"},
    {"text": "a", "label": 0, "source": "x", "source_id": "1", "sender": "AX-MYBANK-S"},
    {"text": "a", "label": 0, "source": "x", "source_id": "1", "sender": "#"},  # the app can't classify it
    {"text": "a", "label": 0, "source": "x", "source_id": "1", "holdout": "test"},
    {"text": "a", "label": 0, "source": "x", "source_id": "1", "holdout": ""},
])
def test_finalize_rejects_bad_rows(row):
    with pytest.raises(ValueError):
        schema.finalize(pd.DataFrame([row]))


def test_finalize_rejects_duplicate_ids():
    rows = [{"text": "a", "label": 0, "source": "x", "source_id": "1"}] * 2
    with pytest.raises(ValueError):
        schema.finalize(pd.DataFrame(rows))


# ── templates ───────────────────────────────────────────────────────────────────────────────────────

def test_template_expansion_is_deterministic_and_promos_are_spam():
    a, b = expand(), expand()
    pd.testing.assert_frame_equal(a, b)
    assert (a["label"] == (a["category"] == "promo").astype("int8")).all() and (a["source"] == "templates").all()
    assert (a["category"] == "promo").any()
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
    ("AX-MYBANK-S", "ALPHANUMERIC"), ("CP-123456-P", "ALPHANUMERIC"), ("  Amazon  Alerts ", "ALPHANUMERIC"),
    ("饭团外卖", "ALPHANUMERIC"), ("Bob@Example.COM", "EMAIL"), ("72975", "SHORT_CODE"), ("123", "SHORT_CODE"),
    ("12345678", "PHONE"), ("12", "PHONE"), ("+12345", "PHONE"), ("+447700900123", "PHONE"),
    ("(650) 555-0001", "PHONE"), ("", None), (" \n", None), ("+", None), ("#", None),
])
def test_sender_type_mirrors_the_app(sender, kind):
    assert schema.sender_type(sender) == kind


def test_modern_eval_sender_types_follow_the_app():
    rows = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines()]
    assert all(schema.sender_type(r["sender"]) == r["sender_type"] for r in rows)


@pytest.mark.parametrize("raw,expected", [
    ("+16289992825", ("+16289992825", "PHONE")), (" 72975 ", ("72975", "SHORT_CODE")),
    ("Google", ("Google", "ALPHANUMERIC")), ("Fedex", ("Fedex", "ALPHANUMERIC")), ("a@b.com", ("a@b.com", "EMAIL")),
    # Masked by the gateway: not a real sender, but the kind of the number it was.
    ("+1818355XXXX", (None, "PHONE")), ("***0192", (None, "PHONE")), ("8759X", (None, "SHORT_CODE")),
    ("\n", (None, None)), ("#", (None, None)),
])
def test_ncsu_sender_fields(raw, expected):
    assert ncsu.sender_fields(raw) == expected


@pytest.mark.parametrize("text,expected", [
    ("Rs.500 spent on your card at SHOP", "en"),
    ("കനത്ത മഴയ്ക്ക് സാധ്യത. KSDMA", "ml"),
    ("भारी बारिश की चेतावनी", "hi"),
    ("கனமழை எச்சரிக்கை", "ta"),
    ("Your OTP is 4821. मत बताइए", "en"),
    ("482913", "en"),
    ("你好世界", None),
])
def test_script_language(text, expected):
    assert lang.script_language(text) == expected


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


# ── the inbox loader and the source registry ────────────────────────────────────────────────────────

INBOX = [  # made up, in the export's format
    {"id": 11, "date": 1748090774738, "sender": "AX-MYBANK-S", "label": 0, "cluster": 3, "half": "train",
     "text": " Rs.500.00 spent on card x1234 at SHOP on 24-05-25. Not you? Call 18000000000 ", "marked": False},
    {"id": 12, "date": 1748090800000, "sender": "12345678", "label": 0, "cluster": 4, "half": "train",
     "text": "482913 is your OTP for login. Do not share it.", "marked": True},
    {"id": 13, "date": 1771000000000, "sender": "56161", "label": 1, "cluster": 5, "half": "eval",
     "text": "Get 50% off on every plan today only!", "marked": False},
    {"id": 14, "date": 1771000500000, "sender": "AL-WEATHR-G", "label": 0, "cluster": 6, "half": "eval",
     "text": "കനത്ത മഴയ്ക്ക് സാധ്യത. ജാഗ്രത പാലിക്കുക", "marked": False},
    {"id": 15, "date": 1771000600000, "sender": "+447700900123", "label": 0, "cluster": 6, "half": "eval",
     "text": "Hi", "marked": False},
]


def write_inbox(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def test_inbox_loader(tmp_path):
    df = inbox.load(write_inbox(tmp_path / "inbox.jsonl", INBOX))
    assert list(df.columns) == schema.COLUMNS
    assert (df["source"] == "inbox").all() and df["source_id"].tolist() == ["11", "12", "13", "14", "15"]
    assert df["label"].tolist() == [0, 0, 1, 0, 0]
    assert df.loc[0, "text"] == INBOX[0]["text"].strip()
    assert df.loc[0, "timestamp"] == pd.Timestamp("2025-05-24 12:46:14.738", tz="UTC")
    assert df["campaign_id"].tolist() == ["inbox:3", "inbox:4", "inbox:5", "inbox:6", "inbox:6"]
    assert df["sender"].tolist() == [r["sender"] for r in INBOX]
    assert df["sender_type"].tolist() == ["ALPHANUMERIC", "PHONE", "SHORT_CODE", "ALPHANUMERIC", "PHONE"]
    assert df["lang"].tolist() == ["en", "en", "en", "ml", "en"]
    assert df["holdout"].isna().tolist() == [True, True, False, False, False]
    assert set(df["holdout"].dropna()) == {inbox.HOLDOUT} == {"inbox_eval"}


def test_inbox_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="s00 doesn't fetch it"):
        inbox.load(tmp_path / "inbox.jsonl")


@pytest.mark.parametrize("field,value", [("half", "test"), ("label", 2)])
def test_inbox_rejects_unknown_values(tmp_path, field, value):
    with pytest.raises(ValueError):
        inbox.load(write_inbox(tmp_path / "inbox.jsonl", [{**INBOX[0], field: value}]))


def test_source_registry():
    by_name = {s.name: s for s in SOURCES}
    assert len(by_name) == len(SOURCES) and SOURCES[0].name == "inbox"
    assert by_name["inbox"].required and not by_name["inbox"].english_only
    assert by_name["ncsu"].english_only and not by_name["ncsu"].required
    assert {s.name for s in SOURCES if s.train_only} == {"hf_ridham_synth_scam"}
    assert not any(by_name[n].english_only for n in ("cube_sms77k", "iiitd_smsassassin", "hf_ridham_synth_scam"))


def test_load_all_keeps_every_inbox_language_and_requires_the_inbox(tmp_path):
    cfg = {**load_config(), "inbox": {"file": "inbox/inbox.jsonl"}}
    with pytest.raises(FileNotFoundError):
        load_all(tmp_path, cfg, sources=("inbox",))
    write_inbox(tmp_path / "inbox" / "inbox.jsonl", INBOX)
    with pytest.warns(UserWarning, match="'uci' skipped"):
        df = load_all(tmp_path, cfg, sources=("uci", "inbox"))
    assert set(df["source"]) == {"inbox"} and len(df) == len(INBOX)
    assert "ml" in set(df["lang"])  # a Malayalam alert survives the English-only filter of the public corpora


# ── the Indian corpora and the rider pool (made-up rows in each file's format) ─────────────────────

def write_csv(path: Path, header: list[str], rows: list[list[str]], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=delimiter, quoting=csv.QUOTE_ALL)
        w.writerow(header)
        w.writerows(rows)


@pytest.mark.parametrize("text,expected", [
    ("Is the assistantÃ¢Â\x80Â\x99s sharing tonight", True), ("Wishing everyone wellÃ°Â\x9fÂ\x99Â\x8f", True),
    ("hasrat tumharÃ¬", True), ("Pay ₹500 by 30-08-2024 , Rs. 20 off", False), ("I knw möntu frwded me", False),
])
def test_mojibake(text, expected):
    assert junioralive_spam_ham.mojibake(text) is expected


def test_cube_loader(tmp_path):
    guesses = ["spam", "spam", "0.9", "0.1", "spam", "0.9", "0.1"]
    write_csv(tmp_path / cube_sms77k.REL_PATH, [
        "id", "Message", "phone", "batch", "truth", "cube", "google", "google_spam", "google_not_spam", "ibm",
        "ibm_spam", "ibm_not_spam"], [
        ["1", "flat  off on all fashion shop now ", "AD-SHOPKR", "5", "spam", *guesses],
        ["2", "bhai kal milte hain", "919000000000", "6", "not-spam", *guesses],
        ["3", "your parcel is out for delivery", "56767", "7", "not-spam", *guesses],
    ], delimiter=";")
    df = cube_sms77k.load(tmp_path)
    assert df["label"].tolist() == [1, 0, 0] and df["source_id"].tolist() == ["1", "2", "3"]
    assert df.loc[0, "text"] == "flat  off on all fashion shop now"  # as given: lowercase, no digits or punctuation
    assert df["sender"].tolist() == ["AD-SHOPKR", "919000000000", "56767"]
    assert df["sender_type"].tolist() == ["ALPHANUMERIC", "PHONE", "SHORT_CODE"]
    write_csv(tmp_path / cube_sms77k.REL_PATH, ["id", "Message", "phone", "truth"], [["1", "hi", "x", "maybe"]],
              delimiter=";")
    with pytest.raises(ValueError):
        cube_sms77k.load(tmp_path)


def test_junioralive_and_iiitd_loaders_drop_mojibake(tmp_path):
    write_csv(tmp_path / junioralive_spam_ham.REL_PATH, ["Msg", "Label"], [
        ["CONGRATULATIONS! FREE 2GB data is yours! Claim on the app", "spam"],
        ["Is the assistantÃ¢Â\x80Â\x99s sharing tonight", "ham"],
        ["See you at 5", "ham"],
    ])
    df = junioralive_spam_ham.load(tmp_path)
    assert df["label"].tolist() == [1, 0] and df["source_id"].tolist() == ["1", "3"] and df["sender"].isna().all()
    write_csv(tmp_path / iiitd_smsassassin.REL_PATH, ["v1", "v2"], [
        ["ham", "kal milte hain yaar :-P"], ["spam", "Win a free trip! Call now"], ["ham", "hasrat tumharÃ¬"],
    ])
    df = iiitd_smsassassin.load(tmp_path)
    assert df["label"].tolist() == [0, 1] and df["text"].tolist()[0] == "kal milte hain yaar :-P"


def test_synthetic_scam_loader(tmp_path):
    header = ["row_id", "text", "label", "original_label", "scam_type", "language", "generator", "topic",
              "audit_action"]
    write_csv(tmp_path / hf_ridham_synth_scam.REL_PATH, header, [
        ["a1", "Aapka KYC expire ho gaya, link pe click karein", "scam", "scam", "fake_kyc", "hinglish", "g",
         "fake_kyc", "keep"],
        ["b2", "आपका OTP 1234 है। किसी को न बताएं", "safe", "safe", "none", "hi", "g", "fake_bank_otp", "unreviewed"],
    ])
    df = hf_ridham_synth_scam.load(tmp_path)
    assert df["label"].tolist() == [1, 0] and df["source_id"].tolist() == ["a1", "b2"]
    assert df["category"].tolist() == ["fake_kyc", "fake_bank_otp"] and df["lang"].tolist() == ["hinglish", "hi"]


def conversation(sender: str, text: str, **result) -> dict:
    submit = {"skip": False, "amount_cents": None, "direction": None, "bank_name": None, "account_mask": None,
              "merchant": None, "category": None, **result}
    return {"tools": [], "messages": [
        {"role": "system", "content": "You are an SMS transaction extraction agent."},
        {"role": "user", "content": f"SMS sender: {sender}\nSMS message: {text}"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {
            "name": "classify_intent", "arguments": {"sms_text": text, "sender": sender}}}]},
        {"role": "tool", "name": "classify_intent", "content": "{}"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {
            "name": "submit_extraction", "arguments": submit}}]},
    ]}


def test_transaction_loader_keeps_completed_transactions_once(tmp_path):
    spent = conversation("JX-MYBANK-S", "Rs.500.00 spent on card x1234 at SHOP", direction="out")
    rows = {
        "sft_tool_train.jsonl": [spent, conversation("AD-MYBANK-S", "Rs.900 credited to a/c x1234", direction="in"),
                                 conversation("AD-MYBANK-S", "482913 is your OTP", skip=True),
                                 conversation("AD-MYBANK-S", "Txn of Rs.20 failed", direction=None)],
        "sft_tool_eval.jsonl": [spent],
    }
    for name, convs in rows.items():
        path = tmp_path / hf_craftogrammer_txn.DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(c) + "\n" for c in convs), encoding="utf-8")
    df = hf_craftogrammer_txn.load(tmp_path)
    assert df["source_id"].tolist() == ["train:1", "train:2"] and (df["label"] == 0).all()
    assert df["sender"].tolist() == ["JX-MYBANK-S", "AD-MYBANK-S"] and (df["sender_type"] == "ALPHANUMERIC").all()
    assert df["text"].tolist() == ["Rs.500.00 spent on card x1234 at SHOP", "Rs.900 credited to a/c x1234"]


@pytest.mark.parametrize("text,expected", [
    ("????? ???????? ?????? ?? ???????? ???", True), ("Flipkart ?? ???? ???? ??? ??? ???????", True),
    ("De ae???", False), ("Kyu ?", False), ("Tamne Password Aavi Gayo Che Sir Kotak No ???", False),
])
def test_garbled(text, expected):
    assert kaggle_rider_inbox.garbled(text) is expected


def test_rider_pool(tmp_path):
    path = tmp_path / kaggle_rider_inbox.REL_PATH
    path.parent.mkdir(parents=True)
    when = "Sat, 7 May 2022 19:48:00 UTC"
    lines = [["phoneNumber", "id", "updateAt", "senderAddress", "text"],
             ["9000000001", "u1", when, "TX-SHOPZY", "Your order is out for delivery"],
             ["9000000001", "u1", when, "TX-SHOPZY", "Your order is out for delivery"],
             ["9000000002", "u2", when, "TX-SHOPZY", "Your order is out for delivery "],
             ["9000000002", "u2", when, "57575", "????? ???? ????"],
             ["9000000003", "u3", when, "VM-MYBANK\r", "482913 is your OTP"],
             ["9000000003", "u3", when, "AD-MYBANK", "  "]]
    with zipfile.ZipFile(path, "w") as z, z.open(kaggle_rider_inbox.MEMBER, "w") as raw:
        raw.write("".join(",".join(f'"{v}"' for v in line) + "\n" for line in lines).encode("utf-8"))
    df = kaggle_rider_inbox.pool(tmp_path)
    assert list(df.columns) == ["sender", "sender_type", "text", "n", "users"]
    assert df.to_dict("records") == [
        {"sender": "TX-SHOPZY", "sender_type": "ALPHANUMERIC", "text": "Your order is out for delivery", "n": 3,
         "users": 2},
        {"sender": "VM-MYBANK", "sender_type": "ALPHANUMERIC", "text": "482913 is your OTP", "n": 1, "users": 1},
    ]


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


needs_india = pytest.mark.skipif(not (RAW_DIR / "india" / "MANIFEST.json").exists(),
                                 reason="run scripts/s00_fetch.py for the Indian corpora first")


@needs_india
@pytest.mark.parametrize("module,n,labels", [
    (cube_sms77k, 77038, {0: 45978, 1: 31060}), (junioralive_spam_ham, 2091, {0: 1350, 1: 741}),
    (iiitd_smsassassin, 1998, {0: 998, 1: 1000}), (hf_ridham_synth_scam, 1580, {0: 725, 1: 855}),
    (hf_craftogrammer_txn, 694, {0: 694}),
])
def test_indian_loaders(module, n, labels):
    df = module.load(RAW_DIR)
    assert len(df) == n and Counter(df["label"]) == labels


@needs_india
def test_rider_pool_file():
    df = kaggle_rider_inbox.pool(RAW_DIR)
    assert len(df) == 62871 and df["n"].sum() == 100230 and df["users"].max() <= 93
    assert not df.duplicated(["sender", "text"]).any()


@pytest.mark.skipif(not INBOX_FILE.exists(), reason="the private inbox export isn't on this machine")
def test_inbox_export():
    df = inbox.load(INBOX_FILE)
    held = df["holdout"].notna()
    assert 0 < held.sum() < len(df)
    assert df.loc[held, "timestamp"].min() >= df.loc[~held, "timestamp"].max()  # the held-out half is the newest
    assert (df["lang"] == "ml").any()  # the Malayalam alerts are kept


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
