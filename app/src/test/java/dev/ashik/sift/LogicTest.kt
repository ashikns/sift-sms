package dev.ashik.sift

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.LinkAnnotation
import dev.ashik.sift.data.db.Folder
import dev.ashik.sift.data.db.LabelCode
import dev.ashik.sift.data.db.MsgStatus
import dev.ashik.sift.outgoing.PartAggregator
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
import dev.ashik.sift.ui.components.findLinks
import dev.ashik.sift.ui.components.messageText
import dev.ashik.sift.ui.newconversation.looksLikeNumber
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class PartAggregatorTest {
    @Test
    fun `all parts must report before a message is final`() {
        assertEquals(PartAggregator.Outcome.PENDING, PartAggregator.outcome(3, 0b011, 0))
        assertEquals(PartAggregator.Outcome.SENT, PartAggregator.outcome(3, 0b111, 0))
        assertEquals(PartAggregator.Outcome.FAILED, PartAggregator.outcome(3, 0b101, 0b010))
        assertEquals(PartAggregator.Outcome.PENDING, PartAggregator.outcome(3, 0b001, 0b010))
        assertEquals(PartAggregator.Outcome.SENT, PartAggregator.outcome(1, 1, 0))
        assertEquals(-1L, PartAggregator.allBits(64))
    }

    @Test
    fun `tp status maps to provider delivery status`() {
        assertEquals(MsgStatus.COMPLETE, PartAggregator.deliveryStatus(0x00))
        assertEquals(MsgStatus.PENDING, PartAggregator.deliveryStatus(0x21))
        assertEquals(MsgStatus.FAILED, PartAggregator.deliveryStatus(0x45))
    }
}

class FolderAndRescoreTest {
    @Test
    fun `user label beats the engine and unclassified messages count as not spam`() {
        assertEquals(LabelCode.HAM, FolderPolicy.effectiveLabel(UserLabel.HAM.ordinal, LabelCode.SPAM))
        assertEquals(LabelCode.SPAM, FolderPolicy.effectiveLabel(UserLabel.SPAM.ordinal, LabelCode.HAM))
        assertEquals(LabelCode.SUSPICIOUS, FolderPolicy.effectiveLabel(null, LabelCode.SUSPICIOUS))
        assertEquals(LabelCode.HAM, FolderPolicy.effectiveLabel(null, null))
        assertEquals(1, FolderPolicy.folderOf(LabelCode.SPAM))
        assertEquals(0, FolderPolicy.folderOf(LabelCode.SUSPICIOUS))
        assertEquals(0, FolderPolicy.folderOf(null))
    }

    @Test
    fun `rescan summary counts conversations that changed folders`() {
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
}

class UiLogicTest {
    @Test
    fun `links are found with surrounding punctuation trimmed`() {
        val text = "Track it (https://ups.com/t/1Z9), or visit www.usps.com. Call me at 5pm."
        val links = findLinks(text)
        assertEquals(listOf("https://ups.com/t/1Z9", "https://www.usps.com"), links.map { it.second })
        assertEquals("https://ups.com/t/1Z9", text.substring(links[0].first))
    }

    @Test
    fun `bare common domains are links but sentence joins are not`() {
        assertEquals(listOf("https://bit.ly/3xYz"), findLinks("Claim now bit.ly/3xYz").map { it.second })
        assertTrue(findLinks("See you soon.Love you").isEmpty())
        assertTrue(findLinks("Version 1.2.3 is out").isEmpty())
    }

    @Test
    fun `suspected spam gets no clickable links`() {
        val body = "Your toll is unpaid: https://ezpass-tolls.top/pay"
        val on = messageText(body, linksEnabled = true, linkColor = Color.Blue)
        val off = messageText(body, linksEnabled = false, linkColor = Color.Blue)
        assertEquals(1, on.getLinkAnnotations(0, body.length).size)
        assertTrue(on.getLinkAnnotations(0, body.length).single().item is LinkAnnotation.Url)
        assertTrue(off.getLinkAnnotations(0, body.length).isEmpty())
        assertEquals(body, off.text)
    }

    @Test
    fun `recipient field recognizes numbers`() {
        assertTrue(looksLikeNumber("(650) 555-0001"))
        assertTrue(looksLikeNumber("+44 7700 900123"))
        assertFalse(looksLikeNumber("Alice"))
        assertFalse(looksLikeNumber("12"))
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
