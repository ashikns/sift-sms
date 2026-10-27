package dev.ashik.sift.spamcore

import dev.ashik.sift.spamcore.features.Features
import dev.ashik.sift.spamcore.normalize.Resources
import dev.ashik.sift.spamcore.normalize.TextNormalizer
import dev.ashik.sift.spamcore.normalize.Url
import dev.ashik.sift.spamcore.tokenize.WordPieceTokenizer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.abs

/** Parity with the Python reference on ml/scripts/s10_golden.py --prelim fixtures (ml/SPEC.md). */
class GoldenNormV1Test {
    private fun open(path: String) = javaClass.classLoader.getResourceAsStream(path)

    private val resources = Resources.load { open("golden/resources/$it") }
    private val normalizer = TextNormalizer(resources)
    private val tokenizer = WordPieceTokenizer.load(open("golden/vocab.txt")!!)
    private val idfOnes = FloatArray(Features.BUCKETS) { 1f }

    private fun cases(): List<JsonObject> =
        java.util.zip.GZIPInputStream(open("golden/norm_v1_prelim.jsonl.gz")!!).bufferedReader().readLines().filter { it.isNotBlank() }
            .map { Json.parseToJsonElement(it).jsonObject }

    private fun urlOf(o: JsonObject) = Url(
        host = o["host"]!!.jsonPrimitive.content,
        tld = o["tld"]!!.jsonPrimitive.content,
        domain = o["domain"]!!.jsonPrimitive.content,
        ipHost = o["ip_host"]!!.jsonPrimitive.boolean,
        punycode = o["punycode"]!!.jsonPrimitive.boolean,
        atSign = o["at_sign"]!!.jsonPrimitive.boolean,
        shortener = o["shortener"]!!.jsonPrimitive.boolean,
        riskyTld = o["risky_tld"]!!.jsonPrimitive.boolean,
        brandLookalike = o["brand_lookalike"]!!.jsonPrimitive.boolean,
        official = o["official"]!!.jsonPrimitive.boolean,
    )

    @Test
    fun `normalizer tokenizer sender identity and features match python`() {
        val all = cases()
        assertTrue(all.size > 1000, "expected the full fixture, got ${all.size}")
        assertTrue(all.count { it["sid"]!!.jsonPrimitive.contentOrNull != null } > 1000, "expected cases with a sender")
        val failures = ArrayList<String>()
        for ((i, c) in all.withIndex()) {
            val raw = c["raw"]!!.jsonPrimitive.content
            val n = normalizer.normalize(raw)
            fun fail(what: String, expected: Any?, actual: Any?) {
                failures += "case $i ${what}: expected=<$expected> actual=<$actual> raw=${raw.take(80).escape()}"
            }
            val clean = c["clean"]!!.jsonPrimitive.content
            if (n.clean != clean) { fail("clean", clean.escape(), n.clean.escape()); continue }
            val enc = c["enc"]!!.jsonPrimitive.content
            if (n.enc != enc) fail("enc", enc, n.enc)
            val ng = c["ng"]!!.jsonPrimitive.content
            if (n.ng != ng) fail("ng", ng, n.ng)
            val urls = c["urls"]!!.jsonArray.map { urlOf(it.jsonObject) }
            if (n.urls != urls) fail("urls", urls, n.urls)
            val urlTokens = c["url_tokens"]!!.jsonArray.map { it.jsonPrimitive.int }.toSet()
            if (n.urlTokens != urlTokens) fail("url_tokens", urlTokens, n.urlTokens)
            val counts = c["counts"]!!.jsonArray.map { it.jsonPrimitive.int }
            val actualCounts = listOf(n.cpCount, n.emojiCount, n.invisibleCount, n.lookalikeCount, n.droppedCount, if (n.nonGsm7) 1 else 0)
            if (counts != actualCounts) fail("counts", counts, actualCounts)
            val ids = c["ids"]!!.jsonArray.map { it.jsonPrimitive.long }
            val actualIds = tokenizer.encode(n.enc).toList()
            if (ids != actualIds) fail("ids", ids, actualIds)
            val idx = c["idx"]!!.jsonArray.map { it.jsonPrimitive.int }
            val v = Features.sparse(n, idfOnes)
            if (v.indices.toList() != idx) {
                fail("idx", idx.size, v.indices.size)
            } else {
                val vals = c["val"]!!.jsonArray.map { it.jsonPrimitive.double }
                val worst = vals.indices.maxOfOrNull { abs(vals[it] - v.values[it]) } ?: 0.0
                if (worst > 1e-6) fail("val", "max|d|<=1e-6", worst)
            }
            val sid = c["sid"]!!.jsonPrimitive.contentOrNull
            val actualSid = Features.senderId(c["sender"]!!.jsonPrimitive.contentOrNull)
            if (actualSid != sid) fail("sid", sid, actualSid)
            // Integer sums and one division per value: the SND vector is bit-identical, not just close.
            val snd = Features.snd(sid, n)
            val sndIdx = c["snd_idx"]!!.jsonArray.map { it.jsonPrimitive.int }
            val sndVal = c["snd_val"]!!.jsonArray.map { it.jsonPrimitive.double.toFloat() }
            if (snd.indices.toList() != sndIdx) {
                fail("snd_idx", sndIdx, snd.indices.toList())
            } else if (snd.values.toList() != sndVal) {
                fail("snd_val", sndVal, snd.values.toList())
            }
            val meta = c["meta"]!!.jsonArray.map { it.jsonPrimitive.double }
            val m = Features.metadata(n)
            val bad = meta.indices.filter { abs(meta[it] - m[it]) > 1e-6 }
            if (bad.isNotEmpty()) fail("meta", bad.map { Features.META_NAMES[it] to meta[it] }, bad.map { m[it] })
        }
        assertTrue(failures.isEmpty(), "${failures.size} mismatches:\n" + failures.take(25).joinToString("\n"))
    }

    private fun String.escape(): String = buildString {
        for (ch in this@escape) if (ch.code in 0x20..0x7e) append(ch) else append("\\u%04x".format(ch.code))
    }
}
