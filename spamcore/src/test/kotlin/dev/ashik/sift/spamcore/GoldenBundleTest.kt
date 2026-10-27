package dev.ashik.sift.spamcore

import dev.ashik.sift.spamcore.encoder.OrtTextEncoder
import dev.ashik.sift.spamcore.features.Features
import dev.ashik.sift.spamcore.features.SparseVector
import dev.ashik.sift.spamcore.head.LinearHead
import dev.ashik.sift.spamcore.model.DirectoryBundleSource
import dev.ashik.sift.spamcore.model.ModelBundle
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test
import java.io.File
import kotlin.math.abs

/**
 * End-to-end parity with the packaged bundle (app/src/main/assets/spam_model) and the real int8 encoder run through
 * the desktop ONNX Runtime jar: tokens, features, embeddings, logits and probabilities vs ml/scripts/s10_golden.py.
 */
class GoldenBundleTest {
    private val bundleDir = File(System.getProperty("sift.bundleDir") ?: "../app/src/main/assets/spam_model")
    private val fixture = File(System.getProperty("sift.bundleGolden") ?: "../app/src/androidTest/assets/golden/bundle_v1.jsonl.gz")

    @Test
    fun `packaged bundle matches python end to end`() {
        assumeTrue(fixture.isFile && File(bundleDir, "model.json").isFile, "bundle or bundle fixture not generated yet")
        val bundle = ModelBundle.load(DirectoryBundleSource(bundleDir), verifyEncoder = true)
        val encoder = OrtTextEncoder.create(bundle)
        val head = bundle.baseHead
        var worstEmb = 0.0
        var worstZ = 0.0
        var worstP = 0.0
        val failures = ArrayList<String>()
        val cases = java.util.zip.GZIPInputStream(fixture.inputStream()).bufferedReader().readLines()
            .filter { it.isNotBlank() }.map { Json.parseToJsonElement(it).jsonObject }
        for ((i, c) in cases.withIndex()) {
            val n = bundle.normalizer.normalize(c["raw"]!!.jsonPrimitive.content)
            if (n.enc != c["enc"]!!.jsonPrimitive.content) { failures += "case $i enc"; continue }
            val ids = bundle.tokenizer.encode(n.enc)
            if (ids.toList() != c["ids"]!!.jsonArray.map { it.jsonPrimitive.long }) { failures += "case $i ids"; continue }
            val sparse = Features.sparse(n, head.idf)
            val idx = c["idx"]!!.jsonArray.map { it.jsonPrimitive.int }
            if (sparse.indices.toList() != idx) { failures += "case $i idx"; continue }
            val vals = (c["val"] as JsonArray).map { it.jsonPrimitive.double }
            if (vals.indices.any { abs(vals[it] - sparse.values[it]) > 1e-6 }) failures += "case $i val"
            val meta = Features.metadata(n)
            val pyMeta = c["meta"]!!.jsonArray.map { it.jsonPrimitive.double }
            if (pyMeta.indices.any { abs(pyMeta[it] - meta[it]) > 1e-6 }) failures += "case $i meta"

            val pyEmb = FloatArray(LinearHead.EMBED_DIM) { c["emb"]!!.jsonArray[it].jsonPrimitive.double.toFloat() }
            val ktEmb = encoder.embed(ids)
            worstEmb = maxOf(worstEmb, pyEmb.indices.maxOf { abs(pyEmb[it] - ktEmb[it]).toDouble() })
            // Head math with Python's embedding isolates the head from encoder numerics.
            val zPy = c["z"]!!.jsonPrimitive.double
            worstZ = maxOf(worstZ, abs(head.z(pyEmb, meta, sparse) - zPy))
            if (abs(head.zFallback(meta, sparse) - c["z_fb"]!!.jsonPrimitive.double) > 1e-5) failures += "case $i z_fb"
            worstP = maxOf(worstP, abs(LinearHead.sigmoid(head.z(ktEmb, meta, sparse)) - c["p"]!!.jsonPrimitive.double))
        }
        encoder.close()
        println("bundle parity over ${cases.size} cases: max|emb|=$worstEmb max|z|=$worstZ max|p|=$worstP")
        assertTrue(failures.isEmpty(), failures.take(20).joinToString("\n"))
        assertTrue(worstZ <= 1e-5, "head logit differs from python by $worstZ")
        assertTrue(worstEmb <= 1e-4, "embedding differs from python by $worstEmb")
        assertTrue(worstP <= 1e-4, "probability differs from python by $worstP")
        assertTrue(SparseVector.EMPTY.size == 0)
    }
}
