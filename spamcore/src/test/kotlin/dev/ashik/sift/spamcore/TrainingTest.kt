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
        fun sparseBlock(idxName: String, valName: String) = FloatArray(Features.BUCKETS).also { w ->
            val vals = b[valName]!!.jsonArray.floats()
            b[idxName]!!.jsonArray.ints().forEachIndexed { i, k -> w[k] = vals[i] }
        }
        return LinearHead(
            bias = b["bias"]!!.jsonPrimitive.double,
            wEmbed = b["w_embed"]!!.jsonArray.floats(),
            muEmbed = b["mu_embed"]!!.jsonArray.floats(),
            wMeta = b["w_meta"]!!.jsonArray.floats(),
            muMeta = b["mu_meta"]!!.jsonArray.floats(),
            wHash = sparseBlock("hash_idx", "hash_val"),
            wSnd = sparseBlock("snd_idx", "snd_val"),
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
            snd = SparseVector(o["snd_idx"]!!.jsonArray.ints(), o["snd_val"]!!.jsonArray.floats()),
        )
    }

    private fun weighting() = fixture["weighting"]!!.jsonObject.let {
        Weighting(
            spam = it["spam"]!!.jsonPrimitive.double, ham = it["ham"]!!.jsonPrimitive.double,
            clusterCap = it["cluster_cap"]!!.jsonPrimitive.double, clusterCos = it["cluster_cos"]!!.jsonPrimitive.double,
        )
    }

    private fun priors() = fixture["priors"]!!.jsonObject.let {
        Priors(
            it["embed"]!!.jsonPrimitive.double, it["meta"]!!.jsonPrimitive.double, it["hash"]!!.jsonPrimitive.double,
            it["snd"]!!.jsonPrimitive.double,
        )
    }

    @Test
    fun `example weights match python`() {
        val expected = fixture["expected"]!!.jsonObject["weights"]!!.jsonArray.doubles()
        assertArrayEquals(expected, ExampleWeighting.weights(examples(), weighting()), 1e-12)
    }

    @Test
    fun `base logits with the sender term match python`() {
        val expected = fixture["expected"]!!.jsonObject["z"]!!.jsonArray.doubles()
        val head = baseHead()
        val ex = examples()
        assertTrue(ex.count { it.snd.size > 0 } > 40 && ex.any { it.snd.size == 0 })
        val worst = ex.indices.maxOf { abs(head.z(ex[it].embedding, ex[it].meta, ex[it].sparse, ex[it].snd) - expected[it]) }
        assertTrue(worst < 1e-9, "max |z_kotlin - z_python| = $worst")
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
            exp["snd_active_idx"]!!.jsonArray.ints().let { idx ->
                maxDiff(exp["snd_active_val"]!!.jsonArray.floats(), FloatArray(idx.size) { result.head.wSnd[idx[it]] })
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
    fun `no examples returns the base head and frozen priors pin every block`() {
        val base = baseHead()
        assertTrue(MapLogisticTrainer().train(base, emptyList(), DoubleArray(0), Priors()).head === base)
        val ex = examples()
        val frozen = Priors(Priors.FROZEN, Priors.FROZEN, Priors.FROZEN, Priors.FROZEN)
        val r = MapLogisticTrainer().train(base, ex, ExampleWeighting.weights(ex, weighting()), frozen).head
        assertArrayEquals(base.wEmbed, r.wEmbed)
        assertArrayEquals(base.wMeta, r.wMeta)
        assertArrayEquals(base.wHash, r.wHash)
        assertArrayEquals(base.wSnd, r.wSnd)
    }

    @Test
    fun `the sender block learns only from examples with a sender and a frozen prior pins it`() {
        val base = baseHead()
        val ex = examples()
        val w = ExampleWeighting.weights(ex, weighting())
        val learned = MapLogisticTrainer().train(base, ex, w, priors()).head
        val sndActive = ex.flatMap { it.snd.indices.asIterable() }.toSet()
        val moved = (0 until Features.BUCKETS).filter { learned.wSnd[it] != base.wSnd[it] }
        assertTrue(moved.isNotEmpty() && sndActive.containsAll(moved), "moved ${moved.size} SND weights")
        val withoutSenders = ex.filter { it.snd.size == 0 }
        val none = MapLogisticTrainer().train(base, withoutSenders, DoubleArray(withoutSenders.size) { 1.0 }, priors()).head
        assertTrue(none.wSnd === base.wSnd)
        val pinned = MapLogisticTrainer().train(base, ex, w, priors().copy(snd = Priors.FROZEN)).head
        assertArrayEquals(base.wSnd, pinned.wSnd)
        assertTrue((0 until Features.BUCKETS).any { pinned.wHash[it] != base.wHash[it] })
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
            val p = LinearHead.sigmoid(h.z(e.embedding, e.meta, e.sparse, e.snd))
            if (e.label == 1) p else 1 - p
        }
        assertTrue(agreement(a) > agreement(base))
    }

    @Test
    fun `five thousand examples train in under a second`() {
        // Realistic structure: messages come from campaigns (shared n-grams + noise, one sender each) whose label is
        // consistent.
        val base = baseHead()
        val rnd = Random(3)
        class Campaign(val grams: List<Int>, val sender: List<Int>, val center: FloatArray, val spam: Boolean)
        val campaigns = (0 until 60).map { c ->
            Campaign(
                (0 until 150).map { rnd.nextInt(Features.BUCKETS) }, (0 until 60).map { rnd.nextInt(Features.BUCKETS) },
                FloatArray(LinearHead.EMBED_DIM) { rnd.nextFloat() - 0.5f }, c % 3 == 0,
            )
        }
        fun sparse(idx: List<Int>): SparseVector {
            val sorted = idx.distinct().sorted().toIntArray()
            return SparseVector(sorted, FloatArray(sorted.size) { 0.1f + rnd.nextFloat() * 0.1f })
        }
        val ex = (0 until 5000).map { i ->
            val c = campaigns[rnd.nextInt(campaigns.size)]
            val grams = sparse(c.grams.filter { rnd.nextFloat() < 0.8f } + (0 until 20).map { rnd.nextInt(Features.BUCKETS) })
            val snd = sparse(c.sender.filter { rnd.nextFloat() < 0.8f })
            val emb = FloatArray(LinearHead.EMBED_DIM) { c.center[it] + (rnd.nextFloat() - 0.5f) * 0.2f }
            PreparedExample(i.toLong(), if (c.spam) 1 else 0, i.toLong(), emb, FloatArray(Features.N_META) { rnd.nextFloat() }, grams, snd)
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
        assertArrayEquals(head.wSnd, back.wSnd)
        assertArrayEquals(head.muMeta, back.muMeta)
        bytes[100] = (bytes[100] + 1).toByte()
        assertThrows(HeadCodec.FormatException::class.java) { HeadCodec.decode(bytes) }
    }

    private fun maxDiff(a: FloatArray, b: FloatArray): Double {
        assertEquals(a.size, b.size)
        return a.indices.maxOf { abs(a[it].toDouble() - b[it]) }
    }
}
