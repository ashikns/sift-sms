package dev.ashik.sift

import dev.ashik.sift.data.db.ClassState
import dev.ashik.sift.data.db.ClassificationEntity
import dev.ashik.sift.data.db.Folder
import dev.ashik.sift.data.db.LabelCode
import dev.ashik.sift.data.db.Source
import dev.ashik.sift.spam.ClassificationCoordinator
import dev.ashik.sift.spam.CueCodec
import dev.ashik.sift.spam.EmbeddingCodec
import dev.ashik.sift.spam.FolderMoves
import dev.ashik.sift.spam.FolderPolicy
import dev.ashik.sift.spam.LatencyStats
import dev.ashik.sift.spam.RescoreService
import dev.ashik.sift.spamcore.Cue
import dev.ashik.sift.spamcore.Reason
import dev.ashik.sift.spamcore.SpamLabel
import dev.ashik.sift.spamcore.UserLabel
import dev.ashik.sift.spamcore.Verdict
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class FolderAndRescoreTest {
    @Test
    fun `user label beats the engine and unclassified messages count as not spam`() {
        assertEquals(LabelCode.HAM, FolderPolicy.effectiveLabel(UserLabel.HAM.ordinal, LabelCode.SPAM))
        assertEquals(LabelCode.SPAM, FolderPolicy.effectiveLabel(UserLabel.SPAM.ordinal, LabelCode.HAM))
        assertEquals(LabelCode.SUSPICIOUS, FolderPolicy.effectiveLabel(null, LabelCode.SUSPICIOUS))
        assertEquals(LabelCode.HAM, FolderPolicy.effectiveLabel(null, null))
    }

    @Test
    fun `rescan summary counts messages that changed folders`() {
        val before = mapOf(1L to Folder.INBOX, 2L to Folder.SPAM, 3L to Folder.INBOX, 4L to Folder.SPAM)
        val after = mapOf(1L to Folder.SPAM, 2L to Folder.INBOX, 3L to Folder.INBOX, 5L to Folder.SPAM)
        assertEquals(FolderMoves(toSpam = 1, toInbox = 1), FolderMoves.between(before, after))
        assertEquals(FolderMoves(0, 0), FolderMoves.between(before, before))
    }

    private fun v(label: SpamLabel, reason: Reason, p: Float = Float.NaN) = Verdict(label, p, reason)

    @Test
    fun `user-driven verdicts move messages in both directions`() {
        assertEquals(LabelCode.SPAM, RescoreService.decide(LabelCode.HAM, 0.1f, v(SpamLabel.SPAM, Reason.SIMILAR_TO_SPAM)))
        assertEquals(LabelCode.SPAM, RescoreService.decide(LabelCode.HAM, null, v(SpamLabel.SPAM, Reason.SIMILAR_TO_SPAM)))
        assertEquals(LabelCode.HAM, RescoreService.decide(LabelCode.SPAM, 0.9f, v(SpamLabel.HAM, Reason.SIMILAR_TO_HAM)))
        assertEquals(LabelCode.HAM, RescoreService.decide(LabelCode.SPAM, 0.9f, v(SpamLabel.HAM, Reason.SIMILAR_TO_HAM)))
    }

    @Test
    fun `model-only flips never move an inbox message to spam`() {
        assertEquals(LabelCode.SUSPICIOUS, RescoreService.decide(LabelCode.HAM, 0.2f, v(SpamLabel.SPAM, Reason.MODEL, 0.95f)))
        assertEquals(LabelCode.SUSPICIOUS, RescoreService.decide(LabelCode.SUSPICIOUS, 0.6f, v(SpamLabel.SPAM, Reason.MODEL, 0.95f)))
        assertEquals(LabelCode.HAM, RescoreService.decide(LabelCode.SUSPICIOUS, 0.6f, v(SpamLabel.HAM, Reason.MODEL, 0.1f)))
    }

    @Test
    fun `spam is rescued by the model only past the hysteresis margin`() {
        assertEquals(LabelCode.SPAM, RescoreService.decide(LabelCode.SPAM, 0.82f, v(SpamLabel.SUSPICIOUS, Reason.MODEL, 0.79f)))
        assertEquals(LabelCode.SUSPICIOUS, RescoreService.decide(LabelCode.SPAM, 0.82f, v(SpamLabel.SUSPICIOUS, Reason.MODEL, 0.70f)))
        assertEquals(LabelCode.HAM, RescoreService.decide(LabelCode.SPAM, null, v(SpamLabel.HAM, Reason.MODEL, 0.1f)))
        assertEquals(LabelCode.SPAM, RescoreService.decide(LabelCode.SPAM, 0.9f, v(SpamLabel.HAM, Reason.UNAVAILABLE)))
    }

    @Test
    fun `spam from a near-duplicate match is withdrawn once the match is gone`() {
        val model = v(SpamLabel.HAM, Reason.MODEL, 0.02f)
        assertEquals(LabelCode.SPAM, RescoreService.decide(LabelCode.SPAM, 0.02f, model))
        assertEquals(LabelCode.HAM, RescoreService.decide(LabelCode.SPAM, 0.02f, model, Reason.SIMILAR_TO_SPAM.name))
        assertEquals(LabelCode.SUSPICIOUS, RescoreService.decide(LabelCode.SPAM, 0.6f, v(SpamLabel.SUSPICIOUS, Reason.MODEL, 0.6f), Reason.SIMILAR_TO_SPAM.name))
        assertEquals(LabelCode.SPAM, RescoreService.decide(LabelCode.SPAM, 0.02f, v(SpamLabel.SPAM, Reason.SIMILAR_TO_SPAM), Reason.SIMILAR_TO_SPAM.name))
    }

    private fun c(label: Int, source: String, score: Float? = null) =
        ClassificationEntity(1, ClassState.DONE, modelScore = score, effectiveLabel = label, effectiveSource = source)

    @Test
    fun `only a verdict that changes an earlier label counts as a change`() {
        val ham = c(LabelCode.HAM, Source.ENGINE, 0.1f)
        assertFalse(ClassificationCoordinator.changesLabel(null, ham), "a first verdict")
        assertFalse(ClassificationCoordinator.changesLabel(ClassificationEntity(1), c(LabelCode.SPAM, Source.ENGINE)), "a new message's pending row")
        assertTrue(ClassificationCoordinator.changesLabel(ham, c(LabelCode.SUSPICIOUS, Source.ENGINE, 0.7f)))
        assertTrue(ClassificationCoordinator.changesLabel(c(LabelCode.SPAM, Source.ENGINE), ham))
        assertFalse(ClassificationCoordinator.changesLabel(ham, c(LabelCode.HAM, Source.ENGINE, 0.3f)), "same label, new score")
        // A deleted mark keeps its label until the model's own verdict replaces it.
        val unmarked = c(LabelCode.SPAM, Source.USER).copy(state = ClassState.PENDING)
        assertTrue(ClassificationCoordinator.changesLabel(unmarked, ham))
        assertFalse(ClassificationCoordinator.changesLabel(unmarked, c(LabelCode.SPAM, Source.ENGINE)))
        assertTrue(ClassificationCoordinator.changesLabel(c(LabelCode.HAM, Source.FAIL_OPEN), c(LabelCode.SPAM, Source.ENGINE)))
    }
}

class CodecTest {
    @Test
    fun `embeddings round-trip as little-endian float32`() {
        val v = floatArrayOf(0f, -1.5f, 3.25f, Float.MIN_VALUE, 1e-7f)
        val bytes = EmbeddingCodec.encode(v)
        assertEquals(20, bytes.size)
        assertEquals(0xC0.toByte(), bytes[6]) // -1.5f = 0xBFC00000, little-endian
        assertEquals(0xBF.toByte(), bytes[7])
        assertArrayEquals(v, EmbeddingCodec.decode(bytes))
    }

    @Test
    fun `cues round-trip and bad json decodes to nothing`() {
        val cues = listOf(Cue("Shortened link", 1.25f), Cue("Unknown sender", 0.5f))
        assertEquals(cues, CueCodec.decode(CueCodec.encode(cues)))
        assertEquals(null, CueCodec.encode(emptyList()))
        assertTrue(CueCodec.decode("{not json").isEmpty())
        assertTrue(CueCodec.decode(null).isEmpty())
    }

    @Test
    fun `latency percentiles`() {
        val stats = LatencyStats(capacity = 100)
        assertEquals(null, stats.summary())
        (1..100L).forEach(stats::record)
        val (p50, p95, n) = stats.summary()!!
        assertEquals(50L, p50)
        assertEquals(95L, p95)
        assertEquals(100, n)
        stats.record(1000)
        assertEquals(100, stats.summary()!!.third)
    }
}
