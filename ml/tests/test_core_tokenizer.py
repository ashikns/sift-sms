from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from smsml.normalize import normalize
from smsml.tokenizer import WordPiece

MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason="all-MiniLM-L6-v2 snapshot not downloaded")


@pytest.fixture(scope="module")
def tokenizers():
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    ours = WordPiece.from_file(MODEL_DIR / "vocab.txt")
    return hf, ours


SAMPLES = [
    "Hello, World! See you at 7?",
    "URGENT: Your E-ZPass account has an unpaid toll of $4.15. Pay now at ezpass-toll.top to avoid a $50 fee.",
    "Your verification code is 482913. Don't share it with anyone.",
    "WINNER!! You've won a £1000 prize. Txt CLAIM to 81010 (18+ only) T&Cs apply",
    "lol ok see u l8r :) ^_^ ~~~ `backticks` {braces} [brackets] |pipes| \\slash",
    "supercalifragilisticexpialidocious" * 4,
    "₹500 €20 ¥300 ¢ £",
    "x" * 101,
]


def hf_ids(hf, text):
    return hf(text, truncation=True, max_length=128)["input_ids"]


@pytest.mark.parametrize("raw", SAMPLES)
def test_matches_hf_on_samples(tokenizers, raw):
    hf, ours = tokenizers
    enc = normalize(raw).enc
    assert ours.encode(enc) == hf_ids(hf, enc)


def test_truncation_matches_hf(tokenizers):
    hf, ours = tokenizers
    enc = normalize(" ".join(f"word{i}" for i in range(400))).enc
    ids = ours.encode(enc)
    assert len(ids) == 128 and ids == hf_ids(hf, enc)


@settings(max_examples=300, deadline=None)
@given(st.text(alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E), max_size=200))
def test_matches_hf_on_random_ascii(tokenizers, raw):
    hf, ours = tokenizers
    enc = normalize(raw).enc
    assert ours.encode(enc) == hf_ids(hf, enc)
