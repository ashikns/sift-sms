"""tok_v1 — BERT-uncased WordPiece over norm_v1's `enc` (SPEC.md). Mirrored by spamcore's WordPieceTokenizer."""

from __future__ import annotations

from pathlib import Path

from .normalize import ascii_lower

CLS, SEP, UNK = 101, 102, 100
MAX_PIECES = 126  # 128 including [CLS] and [SEP]
MAX_WORD_CPS = 100


def is_ascii_punct(cp: int) -> bool:
    return 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126


def load_vocab(path: Path | str) -> dict[str, int]:
    """Line index = id. Split on LF only (and drop a trailing CR), matching the Kotlin loader."""
    text = Path(path).read_bytes().decode("utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    vocab: dict[str, int] = {}
    for i, line in enumerate(lines):
        token = line[:-1] if line.endswith("\r") else line
        vocab.setdefault(token, i)
    return vocab


class WordPiece:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    @staticmethod
    def from_file(path: Path | str) -> "WordPiece":
        return WordPiece(load_vocab(path))

    @staticmethod
    def words(enc: str) -> list[str]:
        out: list[str] = []
        for w in ascii_lower(enc).split(" "):
            if not w:
                continue
            cur: list[str] = []
            for c in w:
                if is_ascii_punct(ord(c)):
                    if cur:
                        out.append("".join(cur))
                        cur = []
                    out.append(c)
                else:
                    cur.append(c)
            if cur:
                out.append("".join(cur))
        return out

    def wordpiece(self, word: str) -> list[int]:
        if len(word) > MAX_WORD_CPS:
            return [UNK]
        out: list[int] = []
        start = 0
        while start < len(word):
            end = len(word)
            found = None
            while start < end:
                sub = word[start:end] if start == 0 else "##" + word[start:end]
                tid = self.vocab.get(sub)
                if tid is not None:
                    found = tid
                    break
                end -= 1
            if found is None:
                return [UNK]
            out.append(found)
            start = end
        return out

    def encode(self, enc: str) -> list[int]:
        pieces: list[int] = []
        for word in self.words(enc):
            pieces.extend(self.wordpiece(word))
            if len(pieces) >= MAX_PIECES:
                break
        return [CLS] + pieces[:MAX_PIECES] + [SEP]
