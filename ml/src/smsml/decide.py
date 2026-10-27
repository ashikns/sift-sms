"""The decision order for one message (mirrors spamcore's SpamDecider)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HAM, SUSPICIOUS, SPAM = "HAM", "SUSPICIOUS", "SPAM"


@dataclass(frozen=True)
class Thresholds:
    conservative: float
    balanced: float
    aggressive: float
    warn: float

    def spam_at(self, sensitivity: str) -> float:
        return getattr(self, sensitivity)


def model_label(p: float, t: Thresholds, sensitivity: str) -> str:
    if p >= t.spam_at(sensitivity):
        return SPAM
    if p >= t.warn:
        return SUSPICIOUS
    return HAM


def decide(*, memory_match, p: float | None, otp_no_url: bool, thresholds: Thresholds,
           sensitivity: str = "balanced") -> tuple[str, str]:
    """(label, reason) for one message from its own content: near-duplicate memory, then model + OTP guardrail."""
    if memory_match is not None:
        return (SPAM, "SIMILAR_TO_SPAM") if memory_match[0] == 1 else (HAM, "SIMILAR_TO_HAM")
    if p is None or not np.isfinite(p):
        return HAM, "UNAVAILABLE"
    label = model_label(p, thresholds, sensitivity)
    if label == SPAM and otp_no_url:
        return SUSPICIOUS, "OTP_GUARDRAIL"
    return label, "MODEL"
