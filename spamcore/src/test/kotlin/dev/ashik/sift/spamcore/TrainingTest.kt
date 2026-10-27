package dev.ashik.sift.spamcore

import dev.ashik.sift.spamcore.features.Features
import dev.ashik.sift.spamcore.features.SparseVector
import dev.ashik.sift.spamcore.head.HeadCodec
import dev.ashik.sift.spamcore.head.LinearHead
import dev.ashik.sift.spamcore.train.ExampleWeighting
import dev.ashik.sift.spamcore.train.MapLogisticTrainer
import dev.ashik.sift.spamcore.train.PreparedExample
import dev.ashik.sift.spamcore.train.Priors
import dev.ashik.sift.spamcore.train.Weighting
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.abs
import kotlin.random.Random

class TrainingTest {
    private val fixture: JsonObject by lazy {
        Json.parseToJsonElement(javaClass.classLoader.getResource("golden/trainer_v1.json")!!.readText()).jsonObject
    }

    private fun JsonArray.floats() = FloatArray(size) { this[it].jsonPrimitive.double.toFloat() }
    private fun JsonArray.doubles() = DoubleArray(size) { this[it].jsonPrimitive.double }
    private fun JsonArray.ints() = IntArray(size) { this[it].jsonPrimitive.int }

    private fun baseHead(): LinearHead {
        val b = fixture["base"]!!.jsonObject
        val wHash = FloatArray(Features.BUCKETS)
        val idx = b["hash_idx"]!!.jsonArray.ints()
        val vals = b["hash_val"]!!.jsonArray.floats()
        idx.forEachIndexed { i, k -> wHash[k] = vals[i] }
        return LinearHead(
            bias = b["bias"]!!.jsonPrimitive.double,
            wEmbed = b["w_embed"]!!.jsonArray.floats(),
            muEmbed = b["mu_embed"]!!.jsonArray.floats(),
            wMeta = b["w_meta"]!!.jsonArray.floats(),
            muMeta = b["mu_meta"]!!.jsonArray.floats(),
            wHash = wHash,
            idf = FloatArray(Features.BUCKETS) { 1f },
            fbBias = 0.0, fbScale = 1.0, wMetaFb = FloatArray(Features.N_META),
        )
    }

    private fun examples(): List<PreparedExample> = fixture["examples"]!!.jsonArray.map { el ->
        val o = el.jsonObject
        PreparedExample(
            key = o["key"]!!.jsonPrimitive.long,
            label = o["label"]!!.jsonPrimitive.int,
            createdAt = o["created_at"]!!.jsonPrimitive.long,
            embedding = (o["emb"] as? JsonArray)?.floats().takeIf { o["emb"] !is JsonNull },
            meta = o["meta"]!!.jsonArray.floats(),
            sparse = SparseVector(o["idx"]!!.jsonArray.ints(), o["val"]!!.jsonArray.floats()),
        )
    }

    private fun weighting() = fixture["weighting"]!!.jsonObject.let {
        Weighting(
            spam = it["spam"]!!.jsonPrimitive.double, ham = it["ham"]!!.jsonPrimitive.double,
            clusterCap = it["cluster_cap"]!!.jsonPrimitive.double, clusterCos = it["cluster_cos"]!!.jsonPrimitive.double,
            maxExamples = it["max_examples"]!!.jsonPrimitive.int,
        )
    }

    private fun priors() = fixture["priors"]!!.jsonObject.let {
        Priors(it["embed"]!!.jsonPrimitive.double, it["meta"]!!.jsonPrimitive.double, it["hash"]!!.jsonPrimitive.double)
    }

    @Test
    fun `example weights match python`() {
        val expected = fixture["expected"]!!.jsonObject["weights"]!!.jsonArray.doubles()
        assertArrayEquals(expected, ExampleWeighting.weights(examples(), weighting()), 1e-12)
    }

    @Test
    fun `trainer reaches the same optimum as scipy`() {
        val exp = fixture["expected"]!!.jsonObject
        val ex = examples()
        val weights = ExampleWeighting.weights(ex, weighting())
        val result = MapLogisticTrainer().train(baseHead(), ex, weights, priors())
        assertTrue(result.gradNorm < 1e-6, "gradient at optimum ${result.gradNorm} after ${result.iterations} iterations")
        val worst = maxOf(
            maxDiff(exp["w_embed"]!!.jsonArray.floats(), result.head.wEmbed),
            maxDiff(exp["w_meta"]!!.jsonArray.floats(), result.head.wMeta),
            exp["active_idx"]!!.jsonArray.ints().let { idx ->
                maxDiff(exp["active_val"]!!.jsonArray.floats(), FloatArray(idx.size) { result.head.wHash[idx[it]] })
            },
        )
        assertTrue(worst < 1e-4, "max |w_kotlin - w_python| = $worst")
    }

    @Test
    fun `sender-type weights never move`() {
        val base = baseHead()
        val ex = examples()
        val r = MapLogisticTrainer().train(base, ex, ExampleWeighting.weights(ex, weighting()), priors())
        for (j in Features.SENDER_RANGE) assertEquals(base.wMeta[j], r.head.wMeta[j], "sender weight ${Features.META_NAMES[j]}")
    }

    @Test
    fun `no examples returns the base head and huge priors stay near it`() {
        val base = baseHead()
        assertTrue(MapLogisticTrainer().train(base, emptyList(), DoubleArray(0), Priors()).head === base)
        val ex = examples()
        val r = MapLogisticTrainer().train(base, ex, ExampleWeighting.weights(ex, weighting()), Priors().scaled(1e7))
        assertTrue(maxDiff(base.wEmbed, r.head.wEmbed) < 1e-4)
        assertTrue(maxDiff(base.wMeta, r.head.wMeta) < 1e-4)
    }

    @Test
    fun `training is deterministic and raises the score of the labeled examples`() {
        val base = baseHead()
        val ex = examples()
        val w = ExampleWeighting.weights(ex, weighting())
        val a = MapLogisticTrainer().train(base, ex, w, priors()).head
        val b = MapLogisticTrainer().train(base, ex, w, priors()).head
        assertArrayEquals(a.wEmbed, b.wEmbed)
        assertArrayEquals(a.wHash, b.wHash)
        // On average the personalized model agrees more with the labels than the base model does.
        fun agreement(h: LinearHead) = ex.sumOf { e ->
            val p = LinearHead.sigmoid(h.z(e.embedding, e.meta, e.sparse))
            if (e.label == 1) p else 1 - p
        }
        assertTrue(agreement(a) > agreement(base))
    }

    @Test
    fun `five thousand examples train in under a second`() {
        // Realistic structure: messages come from campaigns (shared n-grams + noise) whose label is consistent.
        val base = baseHead()
        val rnd = Random(3)
        val campaigns = (0 until 60).map { c ->
            Triple((0 until 150).map { rnd.nextInt(Features.BUCKETS) }, FloatArray(LinearHead.EMBED_DIM) { rnd.nextFloat() - 0.5f }, c % 3 == 0)
        }
        val ex = (0 until 5000).map { i ->
            val (grams, center, spam) = campaigns[rnd.nextInt(campaigns.size)]
            val idx = (grams.filter { rnd.nextFloat() < 0.8f } + (0 until 20).map { rnd.nextInt(Features.BUCKETS) })
                .distinct().sorted().toIntArray()
            val v = FloatArray(idx.size) { 0.1f + rnd.nextFloat() * 0.1f }
            val emb = FloatArray(LinearHead.EMBED_DIM) { center[it] + (rnd.nextFloat() - 0.5f) * 0.2f }
            PreparedExample(i.toLong(), if (spam) 1 else 0, i.toLong(), emb, FloatArray(Features.N_META) { rnd.nextFloat() }, SparseVector(idx, v))
        }
        val w = DoubleArray(ex.size) { 1.0 }
        MapLogisticTrainer(maxIterations = 5).train(base, ex.take(200), w, Priors()) // JIT warm-up
        val t0 = System.nanoTime()
        val r = MapLogisticTrainer().train(base, ex, w, Priors())
        val ms = (System.nanoTime() - t0) / 1e6
        assertTrue(ms < 1000, "took $ms ms (${r.iterations} iterations)")
    }

    @Test
    fun `codec round trips and rejects corruption`() {
        val head = baseHead()
        val bytes = HeadCodec.encode(head.toBlocks())
        val back = LinearHead.fromBlocks(HeadCodec.decode(bytes))
        assertEquals(head.bias, back.bias)
        assertArrayEquals(head.wHash, back.wHash)
        assertArrayEquals(head.muMeta, back.muMeta)
        bytes[100] = (bytes[100] + 1).toByte()
        assertThrows(HeadCodec.FormatException::class.java) { HeadCodec.decode(bytes) }
    }

    private fun maxDiff(a: FloatArray, b: FloatArray): Double {
        assertEquals(a.size, b.size)
        return a.indices.maxOf { abs(a[it].toDouble() - b[it]) }
    }
}
