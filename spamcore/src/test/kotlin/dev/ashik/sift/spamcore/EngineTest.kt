package dev.ashik.sift.spamcore

import dev.ashik.sift.spamcore.encoder.TextEncoder
import dev.ashik.sift.spamcore.engine.Featurizer
import dev.ashik.sift.spamcore.engine.ModelSpamEngine
import dev.ashik.sift.spamcore.features.Features
import dev.ashik.sift.spamcore.features.SparseVector
import dev.ashik.sift.spamcore.head.HeadCodec
import dev.ashik.sift.spamcore.head.LinearHead
import dev.ashik.sift.spamcore.model.DirectoryBundleSource
import dev.ashik.sift.spamcore.model.ModelBundle
import dev.ashik.sift.spamcore.normalize.Resources
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.File
import kotlin.math.sqrt

class EngineTest {
    @TempDir lateinit var tmp: File

    private val scope = CoroutineScope(SupervisorJob())

    @AfterEach
    fun tearDown() = scope.cancel()

    class MemoryStore : ExampleStore {
        val rows = LinkedHashMap<Long, TrainingExample>()
        override suspend fun loadAll() = rows.values.toList()
        override suspend fun upsert(examples: List<TrainingExample>) { examples.forEach { rows[it.messageId] = it } }
        override suspend fun delete(messageIds: Collection<Long>) { messageIds.forEach { rows.remove(it) } }
    }

    /** Deterministic stand-in for the transformer: a normalized bag of token ids ([shift] makes a "different model"). */
    class FakeEncoder(private val shift: Int = 0) : TextEncoder {
        override fun embed(ids: LongArray): FloatArray {
            val v = FloatArray(LinearHead.EMBED_DIM)
            for (id in ids) v[((id + shift) % LinearHead.EMBED_DIM).toInt()] += 1f
            val n = sqrt(v.sumOf { it.toDouble() * it }).toFloat()
            return FloatArray(v.size) { v[it] / n }
        }
        override fun close() = Unit
    }

    private fun writeBundle(dir: File, version: String = "test-1", encoderBytes: ByteArray = ByteArray(8), tauHam: Double = 0.9): File {
        dir.mkdirs()
        val cl = javaClass.classLoader
        File(dir, "resources").mkdirs()
        val resourceShas = Resources.FILES.mapNotNull { name ->
            val bytes = cl.getResourceAsStream("golden/resources/$name")?.readBytes() ?: return@mapNotNull null
            File(dir, "resources/$name").writeBytes(bytes)
            name to ModelBundle.sha256(bytes)
        }.toMap()
        val vocab = cl.getResourceAsStream("golden/vocab.txt")!!.readBytes()
        File(dir, "vocab.txt").writeBytes(vocab)
        val wHash = FloatArray(Features.BUCKETS)
        for (w in listOf("w:winner", "w:prize", "w:claim", "w:gift", "b:gift card", "w:urgent", "w:toll", "w:unpaid")) {
            val (idx, sign) = Features.hashFeature(w)
            wHash[idx] = 6f * sign
        }
        val head = LinearHead(
            bias = -3.0, wEmbed = FloatArray(LinearHead.EMBED_DIM), muEmbed = FloatArray(LinearHead.EMBED_DIM),
            wMeta = FloatArray(Features.N_META).also { it[Features.META_NAMES.indexOf("url_count")] = 1.5f },
            muMeta = FloatArray(Features.N_META), wHash = wHash, wSnd = FloatArray(Features.BUCKETS),
            idf = FloatArray(Features.BUCKETS) { 1f }, fbBias = -3.0, fbScale = 1.0, wMetaFb = FloatArray(Features.N_META),
        )
        val headBytes = HeadCodec.encode(head.toBlocks())
        File(dir, "head_base.bin").writeBytes(headBytes)
        File(dir, "encoder.int8.ort").writeBytes(encoderBytes)
        val metaNames = Features.META_NAMES.joinToString(",") { "\"$it\"" }
        val resources = resourceShas.entries.joinToString(",") { "\"${it.key}\":\"${it.value}\"" }
        File(dir, "model.json").writeText(
            """
            {"schema":1,"bundle_version":"$version",
             "encoder":{"file":"encoder.int8.ort","sha256":"${ModelBundle.sha256(encoderBytes)}",
                        "inputs":["input_ids","attention_mask"],"output":"embedding"},
             "tokenizer":{"file":"vocab.txt","sha256":"${ModelBundle.sha256(vocab)}"},
             "normalizer":{"version":"norm_v1","resources":{$resources}},
             "features":{"version":"${Features.VERSION}","hash_bits":18,"hash_seed":${Features.SEED},"meta_names":[$metaNames]},
             "head":{"file":"head_base.bin","sha256":"${ModelBundle.sha256(headBytes)}"},
             "thresholds":{"full":{"conservative":0.97,"balanced":0.9,"aggressive":0.7,"warn":0.5},
                           "fallback":{"conservative":0.97,"balanced":0.9,"aggressive":0.7,"warn":0.5}},
             "personalization":{"priors":{"embed":1.0,"meta":4.0,"hash":0.5,"snd":0.1},
                "weights":{"spam":1.0,"ham":1.0,"cluster_cap":2.0,"cluster_cos":0.8},
                "debounce_ms":10},
             "knn":{"tau_spam":0.8,"tau_ham":$tauHam}}
            """.trimIndent(),
        )
        return dir
    }

    private fun engine(store: ExampleStore = MemoryStore(), bundleDir: File = writeBundle(File(tmp, "bundle")),
                       encoder: (ModelBundle) -> TextEncoder? = { FakeEncoder() }) =
        ModelSpamEngine(DirectoryBundleSource(bundleDir), store, File(tmp, "state"), scope, encoder)

    private fun sender(key: String = "+15550100") = SenderInfo(key, SenderKind.PHONE)

    private fun input(body: String, s: SenderInfo = sender()) = SmsInput(body, s)

    private fun example(id: Long, body: String, label: UserLabel, createdAt: Long = id, senderKey: String = "+15550100") =
        TrainingExample(id, body, SenderKind.PHONE, false, false, label, createdAt = createdAt, embedding = null, senderKey = senderKey)

    private fun dlt(key: String) = SenderInfo(key, SenderKind.ALPHANUMERIC)

    @Test
    fun `classifies with rules model and fails open`() = runBlocking {
        val e = engine()
        val spam = e.classify(input("WINNER! Claim your prize gift card now"), DecisionPolicy()).verdict
        assertEquals(SpamLabel.SPAM, spam.label)
        assertEquals(Reason.MODEL, spam.reason)
        assertTrue(spam.cues.isNotEmpty())
        assertEquals(SpamLabel.HAM, e.classify(input("Are we still on for dinner?"), DecisionPolicy()).verdict.label)
    }

    @Test
    fun `verdicts are per message even after marking a message from the same sender`() = runBlocking {
        // One sender (a bank short code) sends a promo the user calls spam and a passcode they need.
        val bank = SenderInfo("72975", SenderKind.SHORT_CODE)
        val e = engine()
        e.learn(listOf(example(40, "BANK: Unpaid toll offer! Claim your prize gift card today", UserLabel.SPAM)))
        e.awaitIdle()
        val promo = e.classify(input("BANK: Unpaid toll offer! Claim your prize gift card now", bank), DecisionPolicy()).verdict
        assertEquals(SpamLabel.SPAM, promo.label)
        val otp = e.classify(input("BANK: Your sign-in code is 482913. Don't share it.", bank), DecisionPolicy()).verdict
        assertEquals(SpamLabel.HAM, otp.label)
        assertEquals(Reason.MODEL, otp.reason)
    }

    @Test
    fun `labels take effect immediately and the model retrains`() = runBlocking {
        val store = MemoryStore()
        val e = engine(store)
        val novel = "Your parcel is held at the depot, schedule redelivery at parcel-help.top/r?id=88"
        val related = "Parcel held at depot: schedule redelivery today"
        val before = e.classify(input(novel), DecisionPolicy()).verdict
        assertNotEquals(SpamLabel.SPAM, before.label)
        val p0 = e.classify(input(related), DecisionPolicy()).verdict.probability

        e.learn(listOf(example(10, novel, UserLabel.SPAM)))
        val variant = e.classify(input("Your parcel is held at the depot, schedule redelivery at parcel-help.top/r?id=91"), DecisionPolicy()).verdict
        assertEquals(SpamLabel.SPAM to Reason.SIMILAR_TO_SPAM, variant.label to variant.reason)
        assertEquals(10L, variant.similarMessageId)

        e.awaitIdle()
        assertEquals(1, e.info.value.modelVersion)
        assertEquals(1, e.info.value.examples)
        // A related but not near-duplicate message now scores higher than it did with the base model.
        val p1 = e.classify(input(related), DecisionPolicy()).verdict.probability
        assertTrue(p1 > p0, "p before=$p0 after=$p1")

        e.forget(listOf(10L))
        e.awaitIdle()
        assertNotEquals(Reason.SIMILAR_TO_SPAM, e.classify(input(novel), DecisionPolicy()).verdict.reason)
        assertEquals(0, store.rows.size)
    }

    @Test
    fun `learned state survives a restart`() = runBlocking {
        val store = MemoryStore()
        val bundle = writeBundle(File(tmp, "bundle"))
        val first = engine(store, bundle)
        first.learn(listOf(example(20, "Final notice: unpaid toll balance, pay at toll-pay.top", UserLabel.SPAM)))
        first.awaitIdle()
        assertEquals(1, first.info.value.modelVersion)

        // The first start verified the bundle's checksums and recorded it, so the restart skips re-hashing.
        assertEquals("test-1", File(tmp, "state/bundle_verified").readText())
        val second = engine(store, bundle)
        second.warmUp()
        second.awaitIdle()
        assertEquals(1, second.info.value.modelVersion)
        val v = second.classify(input("Final notice: unpaid toll balance, pay at toll-pay.top"), DecisionPolicy()).verdict
        assertEquals(Reason.SIMILAR_TO_SPAM, v.reason)
    }

    @Test
    fun `learning is rebuilt after an update ships a new bundle`() = runBlocking {
        val store = MemoryStore()
        val first = engine(store, writeBundle(File(tmp, "v1")))
        first.learn(listOf(example(50, "Final notice: unpaid toll balance, pay at toll-pay.top", UserLabel.SPAM)))
        first.awaitIdle()
        assertEquals(1, first.info.value.modelVersion)

        // Same examples, new bundle: the old personalized weights are discarded, so they must be retrained.
        val updated = engine(store, writeBundle(File(tmp, "v2"), version = "test-2"))
        updated.warmUp()
        updated.awaitIdle()
        assertEquals(1, updated.info.value.modelVersion)
        assertEquals(1, updated.info.value.examples)
    }

    @Test
    fun `stored embeddings are recomputed when an update changes the encoder`() = runBlocking {
        val store = MemoryStore()
        val first = engine(store, writeBundle(File(tmp, "e1")), encoder = { FakeEncoder(shift = 0) })
        first.learn(listOf(example(60, "Your parcel is held, schedule redelivery at parcel-help.top", UserLabel.SPAM)))
        first.awaitIdle()
        val before = store.rows.getValue(60L).embedding!!

        val newModel = FakeEncoder(shift = 7)
        val updated = engine(store, writeBundle(File(tmp, "e2"), version = "test-2", encoderBytes = ByteArray(8) { 1 }),
            encoder = { newModel })
        updated.warmUp()
        updated.awaitIdle()
        val after = store.rows.getValue(60L).embedding!!
        assertTrue(!before.contentEquals(after), "embedding should change with the encoder")
        val bundle = ModelBundle.load(DirectoryBundleSource(File(tmp, "e2")))
        val ids = bundle.tokenizer.encode(bundle.normalizer.normalize(store.rows.getValue(60L).body).enc)
        assertTrue(after.contentEquals(newModel.embed(ids)), "embedding should come from the new encoder")
    }

    @Test
    fun `model json carries the sender prior and the feature version`() {
        val bundle = ModelBundle.load(DirectoryBundleSource(writeBundle(File(tmp, "bundle"))))
        assertEquals(0.1, bundle.config.personalization.priors.toPriors().snd)
        val other = writeBundle(File(tmp, "other"))
        File(other, "model.json").writeText(File(other, "model.json").readText().replace(Features.VERSION, "feat_other"))
        assertThrows(ModelBundle.Companion.BundleException::class.java) { ModelBundle.load(DirectoryBundleSource(other)) }
    }

    @Test
    fun `the featurizer derives the sid and the sender vector from the key`() {
        val f = Featurizer(ModelBundle.load(DirectoryBundleSource(writeBundle(File(tmp, "bundle")))))
        val body = "INR 640.00 spent using ICICI Bank Card XX1234 at SWIGGY. Not you? Call 18002662"
        val a = f.prepare(body, dlt("AX-ICICIT-S"))
        val b = f.prepare(body, dlt("JD-ICICIT-T"))
        val c = f.prepare(body, sender("+915550100123"))
        assertEquals("ICICIT", a.sid)
        assertEquals("+915550100123", c.sid)
        assertArrayEquals(Features.snd("ICICIT", a.normalized).indices, a.snd.indices)
        assertArrayEquals(a.snd.values, b.snd.values)
        // The n-gram vector (and so near-duplicate similarity) doesn't depend on the sender.
        assertArrayEquals(a.sparse.indices, c.sparse.indices)
        assertArrayEquals(a.sparse.values, c.sparse.values)
        assertTrue(SparseVector.cosine(a.snd, c.snd) < 0.2)
        assertEquals(0, f.prepare(body, SenderInfo("unknown", SenderKind.UNKNOWN)).snd.size)
        val strings = f.stringsByIndex(a.normalized, a.sid)
        assertTrue(strings.getValue(Features.hashFeature("s:ICICIT|b:spent using").first).contains("s:ICICIT|b:spent using"))
        assertTrue(strings.getValue(Features.hashFeature("w:spent").first).contains("w:spent"))
    }

    @Test
    fun `a not-spam mark covers every route of a DLT sender, and only that sender`() = runBlocking {
        // A low "not spam" cutoff, so a message with a new link domain still counts as a near-duplicate and only the
        // same-sender rule decides.
        val e = engine(bundleDir = writeBundle(File(tmp, "bundle"), tauHam = 0.5))
        val marked = "ICICI Bank: your reward gift card is credited. Details: icici.com/rw?id=4410"
        e.learn(listOf(example(70, marked, UserLabel.HAM, senderKey = "AX-ICICIT-S")))
        val again = "ICICI Bank: your reward gift card is credited. Details: icici-rewards.top/rw?id=4411"
        val sameSender = e.classify(input(again, dlt("JD-ICICIT-T")), DecisionPolicy()).verdict
        assertEquals(Reason.SIMILAR_TO_HAM, sameSender.reason)
        assertEquals(70L, sameSender.similarMessageId)
        // Another sender with a link domain the marked message didn't have gets no "not spam" match.
        assertNotEquals(Reason.SIMILAR_TO_HAM, e.classify(input(again, dlt("VM-HDFCBK-S")), DecisionPolicy()).verdict.reason)
    }

    @Test
    fun `marks teach the sender's wording, not the sender and not other senders`() = runBlocking {
        val store = MemoryStore()
        val e = engine(store)
        val related = "ICICI Bank: claim the gift card reward credited to your card XX9876 this week"
        val before = e.classify(input(related, dlt("JD-ICICIT-T")), DecisionPolicy()).verdict.probability
        e.learn(listOf(example(80, "ICICI Bank: you are a winner! Claim your gift card prize credited to card XX1234", UserLabel.HAM,
            senderKey = "AX-ICICIT-S")))
        e.awaitIdle()
        val sameSender = e.classify(input(related, dlt("JD-ICICIT-T")), DecisionPolicy()).verdict
        val otherSender = e.classify(input(related, dlt("VM-HDFCBK-S")), DecisionPolicy()).verdict
        assertEquals(Reason.MODEL, sameSender.reason)
        assertTrue(sameSender.probability < otherSender.probability && otherSender.probability < before,
            "before=$before same sender=${sameSender.probability} other sender=${otherSender.probability}")
        // The same sender's unrelated wording is unaffected by its sender weights.
        val unrelated = "Urgent: unpaid toll, pay now"
        val a = e.classify(input(unrelated, dlt("AX-ICICIT-S")), DecisionPolicy()).verdict.probability
        val b = e.classify(input(unrelated, dlt("AX-HDFCBK-S")), DecisionPolicy()).verdict.probability
        assertEquals(b, a, 1e-6f)

        // The sender weights are saved with the personalized model and come back after a restart.
        val blocks = HeadCodec.decode(File(tmp, "state/personal_head.spmh").readBytes())
        val sndIdx = (blocks["snd_idx"] as HeadCodec.Block.I32).data
        val sndVal = (blocks["snd_val"] as HeadCodec.Block.F32).data
        assertTrue(sndIdx.isNotEmpty() && sndIdx.size == sndVal.size)
        val restarted = engine(store)
        restarted.warmUp()
        restarted.awaitIdle()
        assertEquals(1, restarted.info.value.modelVersion)
        assertEquals(sameSender.probability, restarted.classify(input(related, dlt("JD-ICICIT-T")), DecisionPolicy()).verdict.probability)
    }

    @Test
    fun `broken bundle and broken encoder fail open`() = runBlocking {
        val broken = File(tmp, "broken").apply { mkdirs(); File(this, "model.json").writeText("{") }
        val v = engine(bundleDir = broken).classify(input("WINNER claim your prize"), DecisionPolicy()).verdict
        assertEquals(SpamLabel.HAM to Reason.UNAVAILABLE, v.label to v.reason)

        val noEncoder = engine(encoder = { error("no ORT") })
        val f = noEncoder.classify(input("WINNER claim your prize gift card"), DecisionPolicy())
        assertEquals(Reason.MODEL_FALLBACK, f.verdict.reason)
        assertEquals(null, f.embedding)
    }
}
