package dev.ashik.sift.spamcore.combine

import dev.ashik.sift.spamcore.DecisionPolicy
import dev.ashik.sift.spamcore.Reason
import dev.ashik.sift.spamcore.Sensitivity
import dev.ashik.sift.spamcore.SpamLabel
import dev.ashik.sift.spamcore.memory.LabeledMemory

data class Thresholds(val conservative: Double, val balanced: Double, val aggressive: Double, val warn: Double) {
    fun spamAt(s: Sensitivity) = when (s) {
        Sensitivity.CONSERVATIVE -> conservative
        Sensitivity.BALANCED -> balanced
        Sensitivity.AGGRESSIVE -> aggressive
    }
}

/**
 * The decision order for one message (mirrors ml/src/smsml/decide.py decide): a near-duplicate of a message the
 * user labeled, else the model. There are no sender-level rules.
 */
object SpamDecider {
    fun memoryRule(match: LabeledMemory.Match?): Pair<SpamLabel, Reason>? = when (match?.label) {
        1 -> SpamLabel.SPAM to Reason.SIMILAR_TO_SPAM
        0 -> SpamLabel.HAM to Reason.SIMILAR_TO_HAM
        else -> null
    }

    /** Model verdict with the OTP guardrail; a non-finite probability fails open. */
    fun modelRule(p: Double, thresholds: Thresholds, policy: DecisionPolicy, fallback: Boolean, otpNoUrl: Boolean): Pair<SpamLabel, Reason> {
        if (!p.isFinite()) return SpamLabel.HAM to Reason.UNAVAILABLE
        val reason = if (fallback) Reason.MODEL_FALLBACK else Reason.MODEL
        val label = when {
            p >= thresholds.spamAt(policy.sensitivity) -> SpamLabel.SPAM
            p >= thresholds.warn -> SpamLabel.SUSPICIOUS
            else -> SpamLabel.HAM
        }
        if (label == SpamLabel.SPAM && otpNoUrl) return SpamLabel.SUSPICIOUS to Reason.OTP_GUARDRAIL
        return label to reason
    }
}
