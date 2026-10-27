package dev.ashik.sift

import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.ashik.sift.spam.AssetBundleSource
import dev.ashik.sift.spam.SpamEngineFactory
import dev.ashik.sift.spamcore.encoder.OrtTextEncoder
import dev.ashik.sift.spamcore.features.Features
import dev.ashik.sift.spamcore.head.LinearHead
import dev.ashik.sift.spamcore.model.ModelBundle
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * On-device parity with the Python reference (ml/scripts/s10_golden.py) using the bundle exactly as the app loads it,
 * plus encoder latency. Device int8 kernels round differently from the desktop's (ARM sdot/i8mm; the x86 emulator's
 * virtual CPU lacks VNNI), by about as much as int8 differs from fp32, so tolerances are looser than the JVM test's:
 * embedding cosine ≥ 0.99, p99 |Δp| ≤ 0.01, and at most 0.3% of Balanced decisions flipped.
 */
@RunWith(AndroidJUnit4::class)
class BundleParityTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val testAssets = InstrumentationRegistry.getInstrumentation().context.assets

    @Test
    fun bundleMatchesPythonOnDevice() {
        val t0 = SystemClock.elapsedRealtimeNanos()
        val bundle = ModelBundle.load(AssetBundleSource(context.assets, SpamEngineFactory.BUNDLE_DIR))
        val t1 = SystemClock.elapsedRealtimeNanos()
        val encoder = OrtTextEncoder.create(bundle) // includes one warm-up run
        val t2 = SystemClock.elapsedRealtimeNanos()
        val head = bundle.baseHead
        // The asset packager stores .gz assets decompressed, under the name without ".gz".
        val fixture = runCatching { testAssets.open("golden/bundle_v1.jsonl") }.getOrNull()
            ?: java.util.zip.GZIPInputStream(testAssets.open("golden/bundle_v1.jsonl.gz"))
        val cases = fixture.bufferedReader().readLines()
            .filter { it.isNotBlank() }.map { Json.parseToJsonElement(it).jsonObject }
        val dps = ArrayList<Double>()
        var minCos = 1.0
        var flips = 0
        val threshold = bundle.config.thresholds.full.balanced
        val runMs = ArrayList<Double>()
        val failures = ArrayList<String>()
        for ((i, c) in cases.withIndex()) {
            val n = bundle.normalizer.normalize(c["raw"]!!.jsonPrimitive.content)
            if (n.enc != c["enc"]!!.jsonPrimitive.content) { failures += "case $i: enc differs (ICU vs Python normalization)"; continue }
            val ids = bundle.tokenizer.encode(n.enc)
            if (ids.toList() != c["ids"]!!.jsonArray.map { it.jsonPrimitive.long }) { failures += "case $i: ids"; continue }
            val sparse = Features.sparse(n, head.idf)
            val snd = Features.snd(Features.senderId(c["sender"]!!.jsonPrimitive.contentOrNull), n)
            val meta = Features.metadata(n)
            val s = SystemClock.elapsedRealtimeNanos()
            val emb = encoder.embed(ids)
            runMs += (SystemClock.elapsedRealtimeNanos() - s) / 1e6
            val py = FloatArray(LinearHead.EMBED_DIM) { c["emb"]!!.jsonArray[it].jsonPrimitive.double.toFloat() }
            var dot = 0.0
            var na = 0.0
            var nb = 0.0
            for (k in py.indices) { dot += py[k].toDouble() * emb[k]; na += py[k].toDouble() * py[k]; nb += emb[k].toDouble() * emb[k] }
            minCos = minOf(minCos, dot / sqrt(na * nb))
            val p = LinearHead.sigmoid(head.z(emb, meta, sparse, snd))
            val pPy = c["p"]!!.jsonPrimitive.double
            dps += abs(p - pPy)
            if ((p >= threshold) != (pPy >= threshold)) flips++
        }
        encoder.close()
        dps.sort()
        runMs.sort()
        val p99 = dps[(dps.size * 0.99).toInt().coerceAtMost(dps.size - 1)]
        val summary = "cases=${cases.size} minCos=$minCos p99|dp|=$p99 max|dp|=${dps.last()} flips=$flips " +
            "load=${(t1 - t0) / 1e6}ms session+warmup=${(t2 - t1) / 1e6}ms " +
            "run p50=${runMs[runMs.size / 2]}ms p95=${runMs[(runMs.size * 0.95).toInt()]}ms"
        Log.i("SiftParity", summary)
        println(summary)
        assertTrue(failures.take(10).joinToString("\n"), failures.isEmpty())
        assertTrue("min embedding cosine $minCos", minCos >= 0.99)
        assertTrue("p99 |dp| $p99", p99 <= 0.01)
        assertTrue("decision flips $flips of ${cases.size}", flips <= maxOf(1, (cases.size * 0.003).toInt()))
    }
}
