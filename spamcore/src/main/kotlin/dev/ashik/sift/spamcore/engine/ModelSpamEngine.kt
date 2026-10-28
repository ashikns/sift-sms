package dev.ashik.sift.spamcore.engine

import dev.ashik.sift.spamcore.DecisionPolicy
import dev.ashik.sift.spamcore.EngineInfo
import dev.ashik.sift.spamcore.ExampleStore
import dev.ashik.sift.spamcore.Reason
import dev.ashik.sift.spamcore.Scored
import dev.ashik.sift.spamcore.SenderInfo
import dev.ashik.sift.spamcore.SenderKind
import dev.ashik.sift.spamcore.SmsInput
import dev.ashik.sift.spamcore.SpamEngine
import dev.ashik.sift.spamcore.SpamLabel
import dev.ashik.sift.spamcore.TrainingExample
import dev.ashik.sift.spamcore.UserLabel
import dev.ashik.sift.spamcore.Verdict
import dev.ashik.sift.spamcore.combine.SpamDecider
import dev.ashik.sift.spamcore.combine.Thresholds
import dev.ashik.sift.spamcore.encoder.OrtTextEncoder
import dev.ashik.sift.spamcore.encoder.TextEncoder
import dev.ashik.sift.spamcore.features.Features
import dev.ashik.sift.spamcore.features.SenderFeatures
import dev.ashik.sift.spamcore.features.SparseVector
import dev.ashik.sift.spamcore.head.HeadCodec
import dev.ashik.sift.spamcore.head.HeadCodec.Block
import dev.ashik.sift.spamcore.head.LinearHead
import dev.ashik.sift.spamcore.memory.LabeledMemory
import dev.ashik.sift.spamcore.model.BundleSource
import dev.ashik.sift.spamcore.model.ModelBundle
import dev.ashik.sift.spamcore.normalize.Normalized
import dev.ashik.sift.spamcore.train.ExampleWeighting
import dev.ashik.sift.spamcore.train.MapLogisticTrainer
import dev.ashik.sift.spamcore.train.PreparedExample
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.util.concurrent.atomic.AtomicReference
import kotlin.coroutines.coroutineContext

/** Turns message text + sender facts into model inputs (norm_v1 → tok_v1 + feat_v3). */
class Featurizer(private val bundle: ModelBundle) {
    class Prepared(
        val normalized: Normalized,
        val ids: LongArray,
        val sparse: SparseVector,
        val meta: FloatArray,
        /** The sender's identity (ml/SPEC.md sid), or null. */
        val sid: String?,
        /** The sender × content vector: the sid paired with the message's words and link domains. */
        val snd: SparseVector,
    ) {
        val domains: Set<String> = normalized.urls.mapTo(HashSet()) { it.domain }
        val otpWithoutLink: Boolean get() = meta[OTP_INDEX] == 1f && normalized.urls.isEmpty()
    }

    fun prepare(body: String, sender: SenderInfo): Prepared {
        val n = bundle.normalizer.normalize(body)
        val sid = Features.senderId(sender.key)
        return Prepared(
            normalized = n,
            ids = bundle.tokenizer.encode(n.enc),
            sparse = Features.sparse(n, bundle.baseHead.idf),
            meta = Features.metadata(n, senderFeatures(sender)),
            sid = sid,
            snd = Features.snd(sid, n),
        )
    }

    /** Bucket index → the feature strings of both sparse vectors that hashed there, for explanations. */
    fun stringsByIndex(n: Normalized, sid: String?): Map<Int, List<String>> {
        val out = HashMap<Int, MutableList<String>>()
        fun add(f: String) = out.getOrPut(Features.hashFeature(f).first) { ArrayList(1) }.add(f)
        for (family in Features.featureStrings(n)) family.forEach(::add)
        Features.sndStrings(sid, n).forEach(::add)
        return out
    }

    private companion object {
        val OTP_INDEX = Features.META_NAMES.indexOf("otp_pattern")

        fun senderFeatures(s: SenderInfo) = SenderFeatures(
            kind = s.kind,
            nanp = s.kind == SenderKind.PHONE && s.key.startsWith("+1") && s.key.length == 12,
            international = s.international,
            firstContact = s.firstContact,
            dlt = Features.dltCategory(s.key),
        )
    }
}

/**
 * The on-device spam engine: bundle model + personalized head + near-duplicate memory.
 *
 * Classification reads an immutable [PersonalState] snapshot and takes no locks. All learning runs on a single
 * writer (examples map, memory updates, retrains); retraining is debounced, it learns only from the user's marks, and the
 * personalized weights are saved with an atomic rename. Every failure path fails open (HAM, [Reason.UNAVAILABLE]).
 */
@OptIn(ExperimentalCoroutinesApi::class)
class ModelSpamEngine(
    private val source: BundleSource,
    private val store: ExampleStore,
    private val stateDir: File,
    private val scope: CoroutineScope,
    private val encoderFactory: (ModelBundle) -> TextEncoder? = { OrtTextEncoder.create(it) },
    private val clock: () -> Long = System::currentTimeMillis,
    private val log: (String, Throwable?) -> Unit = { _, _ -> },
) : SpamEngine {

    private class Loaded(val bundle: ModelBundle, val featurizer: Featurizer, val encoder: TextEncoder?) {
        val full: Thresholds = bundle.config.thresholds.full.toThresholds()
        val fallback: Thresholds = bundle.config.thresholds.fallback.toThresholds()
    }

    private class PersonalState(val head: LinearHead, val memory: LabeledMemory, val version: Int, val lastTrainedAt: Long?)

    @Serializable
    private data class PersonalMeta(
        val bundleVersion: String,
        val featureVersion: String,
        val encoderSha: String,
        val version: Int,
        val trainedAt: Long,
        val examplesDigest: String,
    )

    private val loadMutex = Mutex()
    @Volatile private var loaded: Loaded? = null
    @Volatile private var loadError: Throwable? = null
    private val state = AtomicReference<PersonalState?>(null)
    private val writer = Dispatchers.Default.limitedParallelism(1)

    // Writer-confined.
    private val examples = LinkedHashMap<Long, PreparedExample>()
    private var examplesLoaded = false
    private var retrainJob: Job? = null
    @Volatile private var loadJob: Job? = null

    private val trainer = MapLogisticTrainer()
    private val personalFile get() = File(stateDir, "personal_head.spmh")
    /** The encoder (SHA-256) that produced the stored examples' embeddings. */
    private val embeddingsMarker get() = File(stateDir, "examples_encoder")

    private val _info = MutableStateFlow(EngineInfo("", 0, 0, null, encoderAvailable = false))
    override val info: StateFlow<EngineInfo> = _info

    override suspend fun warmUp() {
        ensureLoaded()
    }

    private suspend fun ensureLoaded(): Loaded? {
        loaded?.let { return it }
        loadError?.let { return null }
        return loadMutex.withLock {
            loaded ?: loadError?.let { null } ?: withContext(Dispatchers.Default) {
                try {
                    val verifiedMarker = File(stateDir, "bundle_verified")
                    val verified = runCatching { verifiedMarker.readText().trim() }.getOrNull()
                    val bundle = ModelBundle.load(source, alreadyVerified = verified)
                    if (verified != bundle.config.bundleVersion) {
                        runCatching { stateDir.mkdirs(); verifiedMarker.writeText(bundle.config.bundleVersion) }
                    }
                    val encoder = try {
                        encoderFactory(bundle)
                    } catch (t: Throwable) {
                        log("encoder unavailable; using the lexical fallback head", t)
                        null
                    }
                    val l = Loaded(bundle, Featurizer(bundle), encoder)
                    val cfg = bundle.config.knn
                    val (head, version, trainedAt) = loadPersonalHead(bundle) ?: Triple(bundle.baseHead, 0, null)
                    state.set(PersonalState(head, LabeledMemory(cfg.tauSpam, cfg.tauHam), version, trainedAt))
                    loaded = l
                    publishInfo()
                    // Examples and the memory load in the background so the first classification isn't delayed.
                    loadJob = scope.launch(writer) { loadExamples(l) }
                    l
                } catch (c: CancellationException) {
                    throw c
                } catch (t: Throwable) {
                    log("model bundle failed to load; failing open", t)
                    loadError = t
                    null
                }
            }
        }
    }

    override suspend fun classify(input: SmsInput, policy: DecisionPolicy, embedding: FloatArray?): Scored = try {
        withContext(Dispatchers.Default) { classifyInternal(input, policy, embedding) }
    } catch (c: CancellationException) {
        throw c
    } catch (t: Throwable) {
        log("classification failed; failing open", t)
        Scored(Verdict(SpamLabel.HAM, Float.NaN, Reason.UNAVAILABLE), embedding)
    }

    private suspend fun classifyInternal(input: SmsInput, policy: DecisionPolicy, cached: FloatArray?): Scored {
        val l = ensureLoaded() ?: return Scored(Verdict(SpamLabel.HAM, Float.NaN, Reason.UNAVAILABLE), cached)
        val s = state.get()!!
        val prep = l.featurizer.prepare(input.body, input.sender)
        val embedding = cached?.takeIf { it.size == LinearHead.EMBED_DIM } ?: l.encoder?.let { enc ->
            try {
                enc.embed(prep.ids)
            } catch (t: Throwable) {
                log("encoder run failed; using the fallback head", t)
                null
            }
        }
        val fallback = embedding == null
        val z = if (fallback) {
            s.head.zFallback(prep.meta, prep.sparse, prep.snd)
        } else {
            s.head.z(embedding, prep.meta, prep.sparse, prep.snd)
        }
        val p = LinearHead.sigmoid(z)
        val match = s.memory.match(prep.sparse, prep.sid, prep.domains)
        val (label, reason) = SpamDecider.memoryRule(match)
            ?: SpamDecider.modelRule(p, if (fallback) l.fallback else l.full, policy, fallback, prep.otpWithoutLink)
        val cues = if (label != SpamLabel.HAM) {
            val strings = l.featurizer.stringsByIndex(prep.normalized, prep.sid)
            s.head.explain(embedding, prep.meta, prep.sparse, prep.snd, strings)
        } else {
            emptyList()
        }
        return Scored(
            Verdict(label, p.toFloat(), reason, cues, similarMessageId = match?.key, modelVersion = s.version),
            embedding,
        )
    }

    override suspend fun learn(examples: List<TrainingExample>) {
        if (examples.isEmpty()) return
        store.upsert(examples) // the label is durable before any model work
        val l = ensureLoaded() ?: return
        withContext(writer) {
            val prepared = examples.map { prepare(l, it) }
            // Keep embeddings computed here, so a restart doesn't recompute them.
            val filled = examples.zip(prepared).filter { (ex, p) -> ex.embedding == null && p.embedding != null }
                .map { (ex, p) -> ex.copy(embedding = p.embedding) }
            if (filled.isNotEmpty()) store.upsert(filled)
            prepared.forEach { this@ModelSpamEngine.examples[it.key] = it }
            updateMemory { mem ->
                mem.withAll(prepared.map { it.toEntry() })
            }
            scheduleRetrain(l)
        }
    }

    override suspend fun forget(messageIds: Collection<Long>) {
        if (messageIds.isEmpty()) return
        store.delete(messageIds)
        val l = ensureLoaded() ?: return
        withContext(writer) {
            messageIds.forEach { examples.remove(it) }
            updateMemory { it.without(messageIds) }
            scheduleRetrain(l)
        }
    }

    override suspend fun awaitIdle() {
        ensureLoaded() ?: return
        loadJob?.join()
        while (true) {
            val job = withContext(writer) { retrainJob?.takeIf { it.isActive } } ?: return
            job.join()
        }
    }

    // ---- writer-confined helpers ----

    private suspend fun loadExamples(l: Loaded) {
        try {
            val stored = store.loadAll()
            // Stored embeddings come from the encoder that was current when each mark was made. If an update changed
            // the encoder, recompute them from the stored text, once, and write them back.
            val encoderSha = l.bundle.config.encoder.sha256
            val embeddedWith = runCatching { embeddingsMarker.readText().trim() }.getOrNull()
            val reembed = embeddedWith != null && embeddedWith != encoderSha
            val refreshed = ArrayList<TrainingExample>()
            for (ex in stored) {
                coroutineContext.ensureActive()
                if (ex.messageId in examples) continue
                val p = prepare(l, ex, useStoredEmbedding = !reembed)
                examples[ex.messageId] = p
                if (reembed && p.embedding != null) refreshed += ex.copy(embedding = p.embedding)
            }
            if (refreshed.isNotEmpty()) store.upsert(refreshed)
            if (l.encoder != null && embeddedWith != encoderSha) {
                runCatching { stateDir.mkdirs(); embeddingsMarker.writeText(encoderSha) }
            }
            examplesLoaded = true
            updateMemory { mem -> mem.withAll(examples.values.map { it.toEntry() }) }
            // Rebuild the personalized model when the saved one doesn't match the examples, or when it was discarded
            // (an update shipped a new bundle) while there are examples to learn from.
            val current = state.get()
            val stale = readPersonalMeta()?.examplesDigest != digest(examples.values) || current?.version == 0 || reembed
            if (examples.isNotEmpty() && stale) scheduleRetrain(l)
            else if (examples.isEmpty() && current?.version != 0) scheduleRetrain(l)
            publishInfo()
        } catch (c: CancellationException) {
            throw c
        } catch (t: Throwable) {
            log("loading learned examples failed", t)
        }
    }

    private fun prepare(l: Loaded, ex: TrainingExample, useStoredEmbedding: Boolean = true): PreparedExample {
        val prep = l.featurizer.prepare(ex.body, SenderInfo(ex.senderKey, ex.senderKind, ex.international, ex.firstContact))
        val emb = ex.embedding?.takeIf { useStoredEmbedding && it.size == LinearHead.EMBED_DIM }
            ?: l.encoder?.let { runCatching { it.embed(prep.ids) }.getOrNull() }
        return PreparedExample(
            key = ex.messageId,
            label = if (ex.label == UserLabel.SPAM) 1 else 0,
            createdAt = ex.createdAt,
            embedding = emb,
            meta = prep.meta,
            sparse = prep.sparse,
            snd = prep.snd,
            sid = prep.sid,
            domains = prep.domains,
        )
    }

    private fun PreparedExample.toEntry() = LabeledMemory.Entry(key, label, sparse, sid, domains)

    private fun updateMemory(f: (LabeledMemory) -> LabeledMemory) {
        while (true) {
            val cur = state.get() ?: return
            val next = PersonalState(cur.head, f(cur.memory), cur.version, cur.lastTrainedAt)
            if (state.compareAndSet(cur, next)) break
        }
        publishInfo()
    }

    private fun scheduleRetrain(l: Loaded) {
        retrainJob?.cancel()
        retrainJob = scope.launch(writer) {
            delay(l.bundle.config.personalization.debounceMs)
            retrain(l)
        }
    }

    private suspend fun retrain(l: Loaded) {
        if (!examplesLoaded) return // loadExamples() schedules a retrain once it has everything
        val cfg = l.bundle.config.personalization
        val base = l.bundle.baseHead
        val list = examples.values.sortedWith(compareBy<PreparedExample> { it.createdAt }.thenBy { it.key })
        val ctx = coroutineContext
        val head: LinearHead = if (list.isEmpty()) {
            base
        } else {
            val weights = ExampleWeighting.weights(list, cfg.weights.toWeighting())
            trainer.train(base, list, weights, cfg.priors.toPriors()) { !ctx.isActive }.head.also { ctx.ensureActive() }
        }
        val prev = state.get() ?: return
        val version = prev.version + 1
        val now = clock()
        while (true) {
            val cur = state.get() ?: return
            if (state.compareAndSet(cur, PersonalState(head, cur.memory, version, now))) break
        }
        try {
            savePersonalHead(l.bundle, head, version, now, digest(list))
        } catch (t: Throwable) {
            log("saving the personalized model failed", t)
        }
        publishInfo()
    }

    private fun digest(list: Collection<PreparedExample>): String =
        ModelBundle.sha256(list.sortedBy { it.key }.joinToString(";") { "${it.key}:${it.label}" }.toByteArray())

    private fun savePersonalHead(bundle: ModelBundle, head: LinearHead, version: Int, trainedAt: Long, digest: String) {
        stateDir.mkdirs()
        val base = bundle.baseHead
        val changed = (0 until Features.BUCKETS).filter { head.wHash[it] != base.wHash[it] }
        val sndChanged = (0 until Features.BUCKETS).filter { head.wSnd[it] != base.wSnd[it] }
        val meta = PersonalMeta(bundle.config.bundleVersion, bundle.config.features.version, bundle.config.encoder.sha256, version, trainedAt, digest)
        HeadCodec.writeAtomic(
            personalFile,
            linkedMapOf(
                "meta" to Block.U8(JSON.encodeToString(PersonalMeta.serializer(), meta).toByteArray()),
                "w_embed" to Block.F32(head.wEmbed),
                "w_meta" to Block.F32(head.wMeta),
                "hash_idx" to Block.I32(changed.toIntArray()),
                "hash_val" to Block.F32(FloatArray(changed.size) { head.wHash[changed[it]] }),
                "snd_idx" to Block.I32(sndChanged.toIntArray()),
                "snd_val" to Block.F32(FloatArray(sndChanged.size) { head.wSnd[sndChanged[it]] }),
            ),
        )
    }

    private fun readPersonal(): Map<String, Block>? = try {
        if (personalFile.isFile) HeadCodec.decode(personalFile.readBytes()) else null
    } catch (t: Throwable) {
        log("ignoring an unreadable personalized model", t)
        null
    }

    private fun readPersonalMeta(): PersonalMeta? = (readPersonal()?.get("meta") as? Block.U8)?.let {
        runCatching { JSON.decodeFromString(PersonalMeta.serializer(), it.data.decodeToString()) }.getOrNull()
    }

    private fun loadPersonalHead(bundle: ModelBundle): Triple<LinearHead, Int, Long?>? {
        val blocks = readPersonal() ?: return null
        val meta = (blocks["meta"] as? Block.U8)?.let {
            runCatching { JSON.decodeFromString(PersonalMeta.serializer(), it.data.decodeToString()) }.getOrNull()
        } ?: return null
        val cfg = bundle.config
        if (meta.bundleVersion != cfg.bundleVersion || meta.featureVersion != cfg.features.version || meta.encoderSha != cfg.encoder.sha256) {
            log("personalized model is for another bundle; it will be retrained", null)
            return null
        }
        val wEmbed = (blocks["w_embed"] as? Block.F32)?.data ?: return null
        val wMeta = (blocks["w_meta"] as? Block.F32)?.data ?: return null
        val idx = (blocks["hash_idx"] as? Block.I32)?.data ?: return null
        val vals = (blocks["hash_val"] as? Block.F32)?.data ?: return null
        val sndIdx = (blocks["snd_idx"] as? Block.I32)?.data ?: return null
        val sndVals = (blocks["snd_val"] as? Block.F32)?.data ?: return null
        if (wEmbed.size != LinearHead.EMBED_DIM || wMeta.size != Features.N_META) return null
        if (idx.size != vals.size || sndIdx.size != sndVals.size) return null
        val base = bundle.baseHead
        fun patched(weights: FloatArray, at: IntArray, values: FloatArray) = weights.copyOf().also {
            for (i in at.indices) if (at[i] in 0 until Features.BUCKETS) it[at[i]] = values[i]
        }
        val head = base.copyWith(
            wEmbed = wEmbed,
            wMeta = wMeta,
            wHash = patched(base.wHash, idx, vals),
            wSnd = patched(base.wSnd, sndIdx, sndVals),
        )
        return Triple(head, meta.version, meta.trainedAt)
    }

    private fun publishInfo() {
        val l = loaded
        val s = state.get()
        _info.value = EngineInfo(
            bundleVersion = l?.bundle?.config?.bundleVersion ?: "",
            modelVersion = s?.version ?: 0,
            examples = examples.size,
            lastTrainedAt = s?.lastTrainedAt,
            encoderAvailable = l?.encoder != null,
        )
    }

    private companion object {
        val JSON = Json { ignoreUnknownKeys = true }
    }
}
