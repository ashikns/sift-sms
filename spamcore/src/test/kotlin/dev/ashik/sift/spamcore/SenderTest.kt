package dev.ashik.sift.spamcore

import dev.ashik.sift.spamcore.features.Features
import dev.ashik.sift.spamcore.features.SenderFeatures
import dev.ashik.sift.spamcore.features.SparseVector
import dev.ashik.sift.spamcore.head.LinearHead
import dev.ashik.sift.spamcore.normalize.Resources
import dev.ashik.sift.spamcore.normalize.TextNormalizer
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.sqrt

/**
 * The sender identity (sid), the DLT category, the sender × content (SND) vector and the head's SND term (ml/SPEC.md
 * feat_v3, score_v2).
 */
class SenderTest {
    private val normalizer = TextNormalizer(Resources.load { javaClass.classLoader.getResourceAsStream("golden/resources/$it") })
    private val alert = normalizer.normalize("INR 2,500.00 spent using ICICI Bank Card XX1234 at AMAZON. Not you? Call 18002662. icici.com/x")

    @Test
    fun `sid drops a DLT header's route and category and keeps any other key`() {
        for (key in listOf("AX-ICICIT-S", "JD-ICICIT-T", "VM-ICICIT", "ICICIT")) assertEquals("ICICIT", Features.senderId(key), key)
        assertEquals("ICI", Features.senderId("AX-ICI-S"))
        assertEquals("ICICIBNK", Features.senderId("AX-ICICIBNK-S"))
        assertEquals("123456", Features.senderId("AX-123456-S"))
        for (key in listOf(
            "+915550100123", "57575", "jane.doe@example.com", "AMAZON ALERTS", "AX-ICICIBANK-S", "AX-IC-S", "A1-ICICIT-S",
            "AXX-ICICIT-S", "AX-ICICIT-SS", "AX-ICICIT-1", "AX-ICICIT-", "AX--S", "ax-icicit-s", "AX-ICICIT-S-T", "AX-ICİCIT-S",
        )) {
            assertEquals(key, Features.senderId(key), key)
        }
        assertNull(Features.senderId("unknown"))
        assertNull(Features.senderId(""))
        assertNull(Features.senderId(null))
    }

    @Test
    fun `the DLT category is a header's P, S, T or G suffix`() {
        val cases = mapOf("AX-ICICIT-S" to 'S', "JD-ICICIT-T" to 'T', "CP-BLNKIT-P" to 'P', "VK-NDMAEW-G" to 'G', "AX-123456-S" to 'S')
        for ((key, cat) in cases) assertEquals(cat, Features.dltCategory(key), key)
        for (key in listOf(
            "VM-HDFCBK", "AX-ICICIT-X", "ICICIT", "AX-ICICIBANK-S", "AX-IC-S", "A1-ICICIT-S", "AX-ICICIT-SS", "AX-ICICIT-1",
            "AX-ICICIT-", "ax-icicit-s", "AX-ICICIT-S-T", "AX-ICICIT-S ", "+915550100123", "57575", "unknown", "", null,
        )) {
            assertNull(Features.dltCategory(key), key)
        }
    }

    @Test
    fun `the DLT category sets one of metadata 27-30`() {
        val names = Features.META_NAMES
        for ((cat, name) in listOf('P' to "dlt_promotional", 'S' to "dlt_service", 'T' to "dlt_transactional", 'G' to "dlt_government")) {
            val m = Features.metadata(alert, SenderFeatures(dlt = cat))
            assertEquals(listOf(name), (27 until 31).filter { m[it] == 1f }.map { names[it] }, "$cat")
        }
        assertTrue((27 until 31).all { Features.metadata(alert)[it] == 0f })
        assertEquals(30, Features.SENDER_RANGE.last)
    }

    @Test
    fun `the SND family pairs the sid with the word and link-domain strings`() {
        val strings = Features.sndStrings("ICICIT", alert)
        val expected = (Features.wordStrings(alert) + Features.urlStrings(alert).filter { it.startsWith("u:") }).map { "s:ICICIT|$it" }
        assertEquals(expected, strings)
        assertTrue("s:ICICIT|b:spent using" in strings && "s:ICICIT|u:icici.com" in strings)
        assertTrue(strings.none { it.startsWith("s:ICICIT|c:") || it.startsWith("s:ICICIT|h:") || it.startsWith("s:ICICIT|t:") })
        assertTrue(Features.sndStrings(null, alert).isEmpty())
    }

    @Test
    fun `the SND vector is binary and L2-normalized and the same for every route`() {
        val v = Features.snd(Features.senderId("AX-ICICIT-S"), alert)
        assertTrue(v.size > 20)
        assertTrue((1 until v.size).all { v.indices[it] > v.indices[it - 1] })
        assertEquals(1.0, v.norm, 1e-6)
        val sums = HashMap<Int, Int>()
        for (f in Features.sndStrings("ICICIT", alert)) Features.hashFeature(f).let { (k, sign) -> sums[k] = (sums[k] ?: 0) + sign }
        val nonzero = sums.filterValues { it != 0 }.toSortedMap()
        val norm = sqrt(nonzero.values.sumOf { it.toDouble() * it })
        assertArrayEquals(nonzero.keys.toIntArray(), v.indices)
        assertArrayEquals(nonzero.values.map { (it / norm).toFloat() }.toFloatArray(), v.values)
        val other = Features.snd(Features.senderId("JD-ICICIT-T"), alert)
        assertArrayEquals(v.indices, other.indices)
        assertArrayEquals(v.values, other.values)
        assertTrue(SparseVector.cosine(v, Features.snd(Features.senderId("VM-HDFCBK-S"), alert)) < 0.2)
        assertEquals(0, Features.snd(null, alert).size)
    }

    private fun head(wSnd: FloatArray, wHash: FloatArray = FloatArray(Features.BUCKETS)) = LinearHead(
        bias = -1.0, wEmbed = FloatArray(LinearHead.EMBED_DIM) { 0.01f * (it % 7) }, muEmbed = FloatArray(LinearHead.EMBED_DIM),
        wMeta = FloatArray(Features.N_META) { 0.1f }, muMeta = FloatArray(Features.N_META) { 0.05f }, wHash = wHash, wSnd = wSnd,
        idf = FloatArray(Features.BUCKETS) { 1f }, fbBias = -2.0, fbScale = 3.0, wMetaFb = FloatArray(Features.N_META) { 0.2f },
    )

    @Test
    fun `the head adds the SND term to both logits, unscaled in the fallback`() {
        val snd = Features.snd("ICICIT", alert)
        val wSnd = FloatArray(Features.BUCKETS).also { w -> snd.indices.forEachIndexed { i, k -> w[k] = 0.5f + i % 3 } }
        val h = head(wSnd)
        var term = 0.0
        for (k in 0 until snd.size) term += wSnd[snd.indices[k]].toDouble() * snd.values[k]
        val emb = FloatArray(LinearHead.EMBED_DIM) { 0.05f }
        val meta = Features.metadata(alert)
        val sparse = Features.sparse(alert, h.idf)
        assertEquals(h.z(emb, meta, sparse, SparseVector.EMPTY) + term, h.z(emb, meta, sparse, snd), 1e-12)
        assertEquals(h.zFallback(meta, sparse, SparseVector.EMPTY) + term, h.zFallback(meta, sparse, snd), 1e-12)
        // The SND weights are a separate block: the same buckets in the n-gram vector don't read them.
        assertEquals(h.z(emb, meta, SparseVector.EMPTY, SparseVector.EMPTY), h.z(emb, meta, snd, SparseVector.EMPTY), 1e-12)
    }

    private fun stringsByIndex(sid: String?): Map<Int, List<String>> {
        val out = HashMap<Int, MutableList<String>>()
        (Features.featureStrings(alert).flatten() + Features.sndStrings(sid, alert)).forEach {
            out.getOrPut(Features.hashFeature(it).first) { ArrayList(1) }.add(it)
        }
        return out
    }

    @Test
    fun `explanations name the sender's wording`() {
        // Weights that push the message toward spam (w·v > 0) by these strings' amounts.
        fun toward(v: SparseVector, strength: Map<String, Float>) = FloatArray(Features.BUCKETS).also { w ->
            for ((f, x) in strength) Features.hashFeature(f).first.let { k -> w[k] = x * v.values[v.indices.indexOf(k)] }
        }
        val meta = FloatArray(Features.N_META)
        for (sid in listOf("ICICIT", "AX|B:C")) { // a key with '|' and ':' still splits at the word's prefix
            val snd = Features.snd(sid, alert)
            val wSnd = toward(snd, mapOf("s:$sid|b:spent using" to 40f, "s:$sid|u:icici.com" to 20f))
            val cues = head(wSnd).explain(null, meta, SparseVector.EMPTY, snd, stringsByIndex(sid))
            assertEquals(listOf("“spent using” from $sid", "“icici.com” from $sid"), cues.map { it.text })
        }
        // The same buckets in the n-gram vector are word cues, not sender cues.
        val sparse = Features.sparse(alert, FloatArray(Features.BUCKETS) { 1f })
        val h = head(FloatArray(Features.BUCKETS), toward(sparse, mapOf("b:spent using" to 40f)))
        val cues = h.explain(null, meta, sparse, Features.snd("ICICIT", alert), stringsByIndex("ICICIT"))
        assertEquals(listOf("Contains “spent using”"), cues.map { it.text })
    }

    @Test
    fun `a promotional suffix with a positive weight is a cue`() {
        val wMeta = FloatArray(Features.N_META).also { it[Features.META_NAMES.indexOf("dlt_promotional")] = 1.5f }
        val h = LinearHead(
            bias = 0.0, wEmbed = FloatArray(LinearHead.EMBED_DIM), muEmbed = FloatArray(LinearHead.EMBED_DIM), wMeta = wMeta,
            muMeta = FloatArray(Features.N_META), wHash = FloatArray(Features.BUCKETS), wSnd = FloatArray(Features.BUCKETS),
            idf = FloatArray(Features.BUCKETS) { 1f }, fbBias = 0.0, fbScale = 1.0, wMetaFb = wMeta,
        )
        val promo = Features.metadata(alert, SenderFeatures(dlt = 'P'))
        val cues = h.explain(null, promo, SparseVector.EMPTY, SparseVector.EMPTY, emptyMap())
        assertEquals(listOf("Registered as a promotional message (-P)"), cues.map { it.text })
        assertTrue(h.explain(null, Features.metadata(alert, SenderFeatures(dlt = 'S')), SparseVector.EMPTY, SparseVector.EMPTY, emptyMap()).isEmpty())
    }
}
