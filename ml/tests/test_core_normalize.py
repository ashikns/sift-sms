import dataclasses

from hypothesis import given, settings
from hypothesis import strategies as st

from smsml.normalize import CURRENCY, normalize
from smsml.resources import default_resources


def res_with(**kw):
    return dataclasses.replace(default_resources(), **kw)


RES = res_with(
    shorteners=frozenset({"bit.ly", "tinyurl.com"}),
    risky_tlds=frozenset({"top", "xyz"}),
    brands=(("usps", frozenset({"usps.com"}), False), ("paypal", frozenset({"paypal.com"}), True),
            ("ups", frozenset({"ups.com"}), False), ("chase", frozenset({"chase.com"}), False)),
)


def test_plain_text_passes_through():
    n = normalize("Hello, World!  See you at 7?", RES)
    assert n.clean == "Hello, World! See you at 7?"
    assert n.ng == "hello, world! see you at 0?"
    assert n.urls == ()
    assert (n.invisible_count, n.lookalike_count, n.dropped_count, n.emoji_count) == (0, 0, 0, 0)
    assert not n.non_gsm7


def test_invisible_characters_are_removed_and_counted():
    n = normalize("fr​ee gi­ft", RES)
    assert n.clean == "free gift"
    assert n.invisible_count == 2


def test_emoji_sequences_do_not_count_as_invisible():
    n = normalize("\U0001F468‍\U0001F469‍\U0001F467 hi ❤️", RES)
    assert n.clean == "hi"
    assert n.invisible_count == 0
    assert n.emoji_count == 4
    assert n.non_gsm7


def test_homoglyphs_fullwidth_and_math_letters_fold_to_ascii():
    assert normalize("Аpple ID", RES).clean == "Apple ID"  # Cyrillic A
    assert normalize("Аpple ID", RES).lookalike_count == 1
    assert normalize("ＦＲＥＥ", RES).clean == "FREE"  # fullwidth
    assert normalize("\U0001D405\U0001D411\U0001D404\U0001D404", RES).clean == "FREE"  # math bold
    assert normalize("ΙRS", RES).clean == "IRS"  # Greek capital iota


def test_accents_are_stripped_without_counting_as_lookalikes():
    n = normalize("café naıve", RES)
    assert n.clean == "cafe naive"
    assert n.lookalike_count == 1  # dotless i is a confusable, the precomposed e-acute is not


def test_punctuation_map_and_dropped_characters():
    n = normalize("It’s “fine” — 你好", RES)
    assert n.clean == "It's \"fine\" -"
    assert n.dropped_count == 2


def test_currency_symbols_survive():
    n = normalize("Win £500 or €20 or ₹100", RES)
    assert n.clean == "Win £500 or €20 or ₹100"


def test_schemed_url_is_scanned_and_replaced_by_host():
    n = normalize("Pay now: (https://EZPass-Toll.top/pay?id=1).", RES)
    assert len(n.urls) == 1
    u = n.urls[0]
    assert (u.host, u.tld, u.domain, u.risky_tld) == ("ezpass-toll.top", "top", "ezpass-toll.top", True)
    assert n.enc == "Pay now: (ezpass-toll.top)."


def test_userinfo_trick_sets_at_sign_and_uses_real_host():
    u = normalize("https://www.paypal.com@evil.top/login", RES).urls[0]
    assert u.host == "evil.top" and u.at_sign and not u.brand_lookalike


def test_bare_domains_and_stoplisted_word_tlds():
    assert normalize("visit usps-redeliver.info now", RES).urls[0].brand_lookalike
    assert normalize("See you soon.Love you", RES).urls == ()
    assert normalize("track at usps.love/abc", RES).urls[0].host == "usps.love"
    assert normalize("go to pay-now.live", RES).urls[0].host == "pay-now.live"
    u = normalize("bit.ly/3xYz", RES).urls[0]
    assert u.shortener and u.domain == "bit.ly"


def test_short_brand_keywords_only_match_whole_host_words():
    assert not normalize("https://startups.example.com", RES).urls[0].brand_lookalike
    assert normalize("https://ups-parcel.example.com", RES).urls[0].brand_lookalike
    assert not normalize("https://purchase.example.com", RES).urls[0].brand_lookalike  # word-mode keyword
    assert normalize("https://chase-alert.example.com", RES).urls[0].brand_lookalike
    assert normalize("https://my-paypal-login.top", RES).urls[0].brand_lookalike  # substring-mode keyword


def test_email_is_not_a_url_and_ip_hosts_are():
    assert normalize("mail bob@gmail.com", RES).urls == ()
    u = normalize("http://192.168.1.10/x", RES).urls[0]
    assert u.ip_host and u.tld == "" and u.domain == "192.168.1.10"


def test_psl_lite_registrable_domain():
    assert normalize("https://evil.github.io/x", RES).urls[0].domain == "evil.github.io"
    assert normalize("https://shop.example.co.uk", RES).urls[0].domain == "example.co.uk"


def test_surrogates_and_truncation():
    n = normalize("a\ud800b", RES)
    assert n.clean == "a b" and n.dropped_count == 0  # U+FFFD is decoding damage, not counted
    assert normalize("x" * 5000, RES).cp_count == 2000


ALLOWED = {chr(c) for c in range(0x20, 0x7F)} | {chr(c) for c in CURRENCY}


@settings(max_examples=400, deadline=None)
@given(st.text(max_size=300))
def test_properties(raw):
    n = normalize(raw, RES)
    assert set(n.clean) <= ALLOWED
    assert "  " not in n.clean and n.clean == n.clean.strip()
    assert normalize(n.clean, RES).clean == n.clean
    assert len(n.ng) == len(n.enc)
