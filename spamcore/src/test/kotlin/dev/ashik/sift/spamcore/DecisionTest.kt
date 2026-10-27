package dev.ashik.sift.spamcore

import dev.ashik.sift.spamcore.combine.SpamDecider
import dev.ashik.sift.spamcore.combine.Thresholds
import dev.ashik.sift.spamcore.features.Features
import dev.ashik.sift.spamcore.features.SparseVector
import dev.ashik.sift.spamcore.memory.LabeledMemory
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Test

class DecisionTest {
    private val t = Thresholds(conservative = 0.95, balanced = 0.85, aggressive = 0.6, warn = 0.4)
    private val policy = DecisionPolicy()

    @Test
    fun `model thresholds sensitivity guardrail and fail open`() {
        assertEquals(SpamLabel.SPAM to Reason.MODEL, SpamDecider.modelRule(0.9, t, policy, fallback = false, otpNoUrl = false))
        assertEquals(SpamLabel.SUSPICIOUS to Reason.MODEL, SpamDecider.modelRule(0.7, t, policy, false, false))
        assertEquals(SpamLabel.SPAM to Reason.MODEL_FALLBACK, SpamDecider.modelRule(0.7, t, DecisionPolicy(Sensitivity.AGGRESSIVE), true, false))
        assertEquals(SpamLabel.SUSPICIOUS to Reason.MODEL, SpamDecider.modelRule(0.9, t, DecisionPolicy(Sensitivity.CONSERVATIVE), false, false))
        assertEquals(SpamLabel.HAM to Reason.MODEL, SpamDecider.modelRule(0.1, t, policy, false, false))
        assertEquals(SpamLabel.SUSPICIOUS to Reason.OTP_GUARDRAIL, SpamDecider.modelRule(0.99, t, policy, false, otpNoUrl = true))
        assertEquals(SpamLabel.HAM to Reason.UNAVAILABLE, SpamDecider.modelRule(Double.NaN, t, policy, false, false))
    }

    private fun vec(vararg pairs: Pair<Int, Float>) =
        SparseVector(pairs.map { it.first }.toIntArray(), pairs.map { it.second }.toFloatArray())

    @Test
    fun `memory matches spam broadly and ham only for the same sender or known domains`() {
        val spam = vec(1 to 1f, 2 to 1f, 3 to 1f)
        val ham = vec(10 to 1f, 11 to 1f, 12 to 1f)
        val mem = LabeledMemory(tauSpam = 0.8, tauHam = 0.9)
            .with(LabeledMemory.Entry(1, 1, spam, "a", setOf("evil.top")))
            .with(LabeledMemory.Entry(2, 0, ham, "bank", setOf("bank.com")))
        assertEquals(1L, mem.match(vec(1 to 1f, 2 to 1f, 3 to 0.9f), "zzz", emptySet())?.key)
        assertNull(mem.match(vec(1 to 1f, 7 to 1f), "zzz", emptySet()))
        assertEquals(2L, mem.match(ham, "bank", setOf("phish.top"))?.key)
        assertEquals(2L, mem.match(ham, "other", setOf("bank.com"))?.key)
        assertNull(mem.match(ham, "other", setOf("phish.top")))
        assertNull(mem.match(ham, null, setOf("phish.top")))
        assertEquals(0, mem.without(listOf(1L, 2L)).size)
    }

    @Test
    fun `every route of a DLT sender is the same sender for ham matches`() {
        val ham = vec(10 to 1f, 11 to 1f, 12 to 1f)
        val mem = LabeledMemory(tauSpam = 0.8, tauHam = 0.9)
            .with(LabeledMemory.Entry(2, 0, ham, Features.senderId("AX-ICICIT-S"), setOf("icici.com")))
        assertEquals(2L, mem.match(ham, Features.senderId("JD-ICICIT-T"), setOf("icici-rewards.top"))?.key)
        assertEquals(2L, mem.match(ham, Features.senderId("ICICIT"), setOf("icici-rewards.top"))?.key)
        assertNull(mem.match(ham, Features.senderId("VM-HDFCBK-S"), setOf("icici-rewards.top")))
        // Without a sender identity there is no same sender; known domains still match.
        val unknown = LabeledMemory(tauSpam = 0.8, tauHam = 0.9).with(LabeledMemory.Entry(3, 0, ham, Features.senderId("unknown"), emptySet()))
        assertNull(unknown.match(ham, Features.senderId(""), setOf("phish.top")))
        assertEquals(3L, unknown.match(ham, null, emptySet())?.key)
    }
}
