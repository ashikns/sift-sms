"""Near-duplicate memory and the decision order (mirrors spamcore's LabeledMemory and SpamDecider)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import sparse_cosine

HAM, SUSPICIOUS, SPAM = "HAM", "SUSPICIOUS", "SPAM"
SENSITIVITIES = ("conservative", "balanced", "aggressive")


@dataclass(frozen=True)
class Thresholds:
    conservative: float
    balanced: float
    aggressive: float
    warn: float

    def spam_at(self, sensitivity: str) -> float:
        return getattr(self, sensitivity)

    def as_dict(self) -> dict:
        return {"conservative": self.conservative, "balanced": self.balanced,
                "aggressive": self.aggressive, "warn": self.warn}


@dataclass(frozen=True)
class MemoryEntry:
    key: int
    label: int
    idx: np.ndarray
    val: np.ndarray
    sender: str
    domains: frozenset[str]


@dataclass
class LabeledMemory:
    tau_spam: float = 0.80
    tau_ham: float = 0.90
    entries: dict[int, MemoryEntry] = field(default_factory=dict)

    def add(self, e: MemoryEntry) -> None:
        self.entries[e.key] = e

    def remove(self, key: int) -> None:
        self.entries.pop(key, None)

    def match(self, idx: np.ndarray, val: np.ndarray, sender: str, domains: frozenset[str]):
        """Returns (label, key, cosine) or None. Ham matches need the same sender or no new link domains."""
        best_spam = best_ham = None
        for e in self.entries.values():
            cos = sparse_cosine(idx, val, e.idx, e.val)
            if e.label == 1:
                if cos >= self.tau_spam and (best_spam is None or cos > best_spam[2]):
                    best_spam = (1, e.key, cos)
            elif cos >= self.tau_ham and (sender == e.sender or domains <= e.domains):
                if best_ham is None or cos > best_ham[2]:
                    best_ham = (0, e.key, cos)
        if best_ham is not None and (best_spam is None or best_ham[2] >= best_spam[2]):
            return best_ham
        return best_spam


def model_label(p: float, t: Thresholds, sensitivity: str) -> str:
    if p >= t.spam_at(sensitivity):
        return SPAM
    if p >= t.warn:
        return SUSPICIOUS
    return HAM


def decide(*, memory_match, p: float | None, otp_no_url: bool, thresholds: Thresholds,
           sensitivity: str = "balanced") -> tuple[str, str]:
    """(label, reason) for one message from its own content: near-duplicate memory, then model + OTP guardrail.

    There are no sender-level rules: one sender can send both a promo and a one-time passcode.
    """
    if memory_match is not None:
        return (SPAM, "SIMILAR_TO_SPAM") if memory_match[0] == 1 else (HAM, "SIMILAR_TO_HAM")
    if p is None or not np.isfinite(p):
        return HAM, "UNAVAILABLE"
    label = model_label(p, thresholds, sensitivity)
    if label == SPAM and otp_no_url:
        return SUSPICIOUS, "OTP_GUARDRAIL"
    return label, "MODEL"
