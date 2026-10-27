package dev.ashik.sift.spam

import android.os.SystemClock
import android.util.Log
import androidx.room3.withWriteTransaction
import dev.ashik.sift.data.db.AppDatabase
import dev.ashik.sift.data.db.ClassState
import dev.ashik.sift.data.db.ClassificationEntity
import dev.ashik.sift.data.db.LabelCode
import dev.ashik.sift.data.db.MessageEmbeddingEntity
import dev.ashik.sift.data.db.MessageEntity
import dev.ashik.sift.data.db.MsgType
import dev.ashik.sift.data.db.Source
import dev.ashik.sift.data.repo.ConversationProjector
import dev.ashik.sift.data.settings.SettingsRepository
import dev.ashik.sift.spamcore.Cue
import dev.ashik.sift.spamcore.DecisionPolicy
import dev.ashik.sift.spamcore.Reason
import dev.ashik.sift.spamcore.SenderInfo
import dev.ashik.sift.spamcore.SenderKind
import dev.ashik.sift.spamcore.SmsInput
import dev.ashik.sift.spamcore.SpamEngine
import dev.ashik.sift.spamcore.SpamLabel
import dev.ashik.sift.spamcore.Verdict
import dev.ashik.sift.telephony.AddressNormalizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * Builds the engine's [SenderInfo]: facts about the channel a message arrived on (sender type, international,
 * first contact). They feed that message's own score; nothing about the sender decides a verdict by itself.
 */
class SenderInfoProvider(
    private val db: AppDatabase,
    private val normalizer: AddressNormalizer,
) {
    suspend fun build(m: MessageEntity): SenderInfo = SenderInfo(
        key = m.senderKey,
        kind = runCatching { SenderKind.valueOf(m.senderKind) }.getOrDefault(SenderKind.UNKNOWN),
        international = normalizer.normalize(m.address).international,
        firstContact = db.messages().earlierIncomingCount(m.senderKey, m.id, m.date) == 0,
    )
}

@Serializable
private data class StoredCue(val text: String, val weight: Float)

object CueCodec {
    private val json = Json { ignoreUnknownKeys = true }

    fun encode(cues: List<Cue>): String? =
        if (cues.isEmpty()) null else json.encodeToString(cues.map { StoredCue(it.text, it.weight) })

    fun decode(text: String?): List<Cue> =
        if (text.isNullOrBlank()) emptyList()
        else runCatching { json.decodeFromString<List<StoredCue>>(text).map { Cue(it.text, it.weight) } }.getOrDefault(emptyList())
}

/**
 * Runs the engine for stored messages and records verdicts. All engine calls are serialized on one dispatcher
 * behind a fair mutex, one message at a time, so a real-time classification never waits behind a whole batch.
 */
class ClassificationCoordinator(
    private val db: AppDatabase,
    private val engine: SpamEngine,
    private val senders: SenderInfoProvider,
    private val settings: SettingsRepository,
    private val projector: ConversationProjector,
    private val latency: LatencyStats,
) {
    @OptIn(ExperimentalCoroutinesApi::class)
    private val engineDispatcher = Dispatchers.Default.limitedParallelism(1)
    private val mutex = Mutex()

    /** A computed verdict for a message, not yet applied. */
    data class Computed(val message: MessageEntity, val verdict: Verdict, val previous: ClassificationEntity?)

    /** Runs the engine for an incoming message (uses the cached embedding when it's still valid). */
    suspend fun compute(m: MessageEntity, policy: DecisionPolicy? = null): Computed? {
        if (m.type != MsgType.INBOX) return null
        val previous = db.classifications().get(m.id)
        val sender = senders.build(m)
        val effectivePolicy = policy ?: settings.current().policy
        val bundle = engine.info.value.bundleVersion
        val cached = db.embeddings().get(m.id)
            ?.takeIf { it.fingerprint == m.fingerprint && it.bundleVersion == bundle }
            ?.let { EmbeddingCodec.decode(it.vector) }
        val started = SystemClock.elapsedRealtime()
        val scored = withContext(engineDispatcher) {
            mutex.withLock {
                runCatching { engine.classify(SmsInput(m.id, m.body, sender, m.date), effectivePolicy, cached) }
                    .onFailure { Log.e(TAG, "engine.classify failed for ${m.id}", it) }
                    .getOrNull()
            }
        }
        if (scored == null) return Computed(m, failOpen(), previous)
        if (cached == null && scored.embedding != null) {
            latency.record(SystemClock.elapsedRealtime() - started)
            db.embeddings().upsert(
                MessageEmbeddingEntity(m.id, m.fingerprint, bundle, EmbeddingCodec.encode(scored.embedding!!), System.currentTimeMillis()),
            )
        }
        return Computed(m, scored.verdict, previous)
    }

    /** Stores [c]'s verdict as-is (the user's own label still wins) and rebuilds the conversation. */
    suspend fun apply(c: Computed, effectiveOverride: Int? = null): ClassificationEntity {
        val entity = entityFor(c, effectiveOverride)
        db.withWriteTransaction {
            db.classifications().upsert(entity)
            projector.rebuild(listOf(c.message.threadId))
        }
        return entity
    }

    fun entityFor(c: Computed, effectiveOverride: Int? = null): ClassificationEntity {
        val v = c.verdict
        val userLabel = c.previous?.userLabel
        val effective = when {
            userLabel != null -> FolderPolicy.effectiveLabel(userLabel, null)
            effectiveOverride != null -> effectiveOverride
            else -> LabelCode.of(v.label)
        }
        val source = when {
            userLabel != null -> Source.USER
            v.reason == Reason.UNAVAILABLE -> Source.FAIL_OPEN
            else -> Source.ENGINE
        }
        return ClassificationEntity(
            messageId = c.message.id,
            state = ClassState.DONE,
            modelLabel = LabelCode.of(v.label),
            modelScore = v.probability.takeUnless { it.isNaN() },
            reason = v.reason.name,
            cues = CueCodec.encode(v.cues),
            similarMessageId = v.similarMessageId,
            modelVersion = v.modelVersion,
            userLabel = userLabel,
            userLabeledAt = c.previous?.userLabeledAt,
            effectiveLabel = effective,
            effectiveSource = source,
            notified = c.previous?.notified ?: false,
            classifiedAt = System.currentTimeMillis(),
        )
    }

    /** Classifies and applies in one go; returns null for outgoing or missing messages. */
    suspend fun classifyAndApply(messageId: Long): ClassificationEntity? {
        val m = db.messages().get(messageId) ?: return null
        val computed = compute(m) ?: return null
        return apply(computed)
    }

    private fun failOpen() = Verdict(SpamLabel.HAM, Float.NaN, Reason.UNAVAILABLE)

    private companion object {
        const val TAG = "SiftClassify"
    }
}
