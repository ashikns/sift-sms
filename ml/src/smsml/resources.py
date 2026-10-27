"""Loads the resource lists that norm_v1 / feat_v2 depend on (SPEC.md). Kotlin parses the same files."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "resources"

FILES = (
    "gsm7_v1.txt",
    "confusables_v1.tsv",
    "tlds_v1.txt",
    "tld_stoplist_v1.txt",
    "psl_lite_v1.txt",
    "shorteners_v1.txt",
    "risky_tlds_v1.txt",
    "brands_v1.tsv",
)


def read_lines(path: Path) -> list[str]:
    """Non-empty lines with `#` comments and surrounding whitespace removed."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


@dataclass(frozen=True)
class Resources:
    gsm7: frozenset[int]
    confusables: dict[int, str]
    tlds: frozenset[str]
    tld_stoplist: frozenset[str]
    psl: frozenset[str]
    shorteners: frozenset[str]
    risky_tlds: frozenset[str]
    brands: tuple[tuple[str, frozenset[str], bool], ...]  # (keyword, official domains, substring match)

    @property
    def official_domains(self) -> frozenset[str]:
        return frozenset(d for _, officials, _ in self.brands for d in officials)

    @staticmethod
    def load(directory: Path = DEFAULT_DIR) -> "Resources":
        def lines(name: str) -> list[str]:
            path = directory / name
            return read_lines(path) if path.exists() else []

        confusables = {}
        for line in lines("confusables_v1.tsv"):
            src, dst = line.split("\t")
            confusables[int(src, 16)] = dst.strip()
        brands = []
        for line in lines("brands_v1.tsv"):
            cols = line.split("\t")
            keyword, officials = cols[0].strip(), cols[1]
            mode = cols[2].strip() if len(cols) > 2 else ("sub" if len(keyword) >= 5 else "word")
            if mode not in ("sub", "word"):
                raise ValueError(f"bad brand match mode {mode!r} for {keyword!r}")
            brands.append((keyword, frozenset(d.strip() for d in officials.split(",") if d.strip()), mode == "sub"))
        return Resources(
            gsm7=frozenset(int(x, 16) for x in lines("gsm7_v1.txt")),
            confusables=confusables,
            tlds=frozenset(lines("tlds_v1.txt")),
            tld_stoplist=frozenset(lines("tld_stoplist_v1.txt")),
            psl=frozenset(lines("psl_lite_v1.txt")),
            shorteners=frozenset(lines("shorteners_v1.txt")),
            risky_tlds=frozenset(lines("risky_tlds_v1.txt")),
            brands=tuple(brands),
        )


@lru_cache(maxsize=None)
def default_resources() -> Resources:
    return Resources.load(DEFAULT_DIR)
