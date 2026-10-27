package dev.ashik.sift.spam

import androidx.room3.withWriteTransaction
import dev.ashik.sift.data.db.AppDatabase
import dev.ashik.sift.data.db.ClassState
import dev.ashik.sift.data.db.ClassificationEntity
import dev.ashik.sift.data.db.FeedbackEventEntity
import dev.ashik.sift.data.db.FeedbackKind
import dev.ashik.sift.data.db.LabelCode
import dev.ashik.sift.data.db.MessageEntity
import dev.ashik.sift.data.db.MsgType
import dev.ashik.sift.data.db.Source
import dev.ashik.sift.data.db.UserLabelCode
import dev.ashik.sift.data.repo.ConversationProjector
import dev.ashik.sift.notifications.MessageNotifier
import dev.ashik.sift.spamcore.Reason
import dev.ashik.sift.spamcore.SenderKind
import dev.ashik.sift.spamcore.SpamEngine
import dev.ashik.sift.spamcore.TrainingExample
import dev.ashik.sift.spamcore.UserLabel
import dev.ashik.sift.spamcore.Verdict
import dev.ashik.sift.telephony.AddressNormalizer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Re-scores recent unlabeled messages after the model or the user's labels change.
 * User-driven verdicts (same sender, near-duplicates, trust) move messages; model-only flips never move an inbox
 * message to Spam retroactively (it only gets a "Possible spam" badge), but do rescue Spam-folder false positives.
 */
class RescoreService(
    private val db: AppDatabase,
    private val coordinator: ClassificationCoordinator,
    private val projector: ConversationProjector,
) {
    data class Outcome(val moved: Int, val threads: Set<Long>)

    /**
     * Serializes re-scores with the user's label changes (FeedbackService takes it too). A re-score reads a message's
     * classification, computes, then writes; without the lock a background re-score that started before a mark or an
     * Undo could write back the stale state over it.
     */
    val lock = Mutex()

    /**
     * Right after an explicit mark: re-scores the recent window (and the sender's other messages) by content, so
     * near-duplicates of the marked message move with it and the snackbar can count them.
     */
    suspend fun afterFeedback(batchId: Long, marked: List<MessageEntity>): Outcome {
        val markedIds = marked.map { it.id }.toSet()
        val senderKeys = marked.map { it.senderKey }.distinct()
        val sameSender = db.messages().unlabeledIncomingFrom(senderKeys, 200)
        val recent = db.messages().recentUnlabeledIncoming(System.currentTimeMillis() - WINDOW_MS, 200)
        val candidates = (sameSender + recent).distinctBy { it.id }.filter { it.id !in markedIds }
        return rescore(candidates, batchId)
    }

    /** After a retrain finished (or on demand): the full recent window, no undo records. */
    suspend fun afterRetrain(): Outcome {
        val candidates = db.messages().recentUnlabeledIncoming(System.currentTimeMillis() - WINDOW_MS, 2000)
        return rescore(candidates, batchId = null)
    }

    private suspend fun rescore(candidates: List<MessageEntity>, batchId: Long?): Outcome = lock.withLock {
        val updates = ArrayList<ClassificationEntity>()
        val events = ArrayList<FeedbackEventEntity>()
        val threads = HashSet<Long>()
        var moved = 0
        val now = System.currentTimeMillis()
        for (m in candidates) {
            if (m.type != MsgType.INBOX) continue
            val computed = coordinator.compute(m) ?: continue
            val previous = computed.previous
            if (previous?.userLabel != null) continue
            val oldEffective = previous?.effectiveLabel ?: LabelCode.HAM
            val newEffective = decide(oldEffective, previous?.modelScore, computed.verdict)
            val entity = coordinator.entityFor(computed, newEffective)
            if (previous == null || entity.effectiveLabel != previous.effectiveLabel || entity.reason != previous.reason ||
                entity.modelScore != previous.modelScore || previous.state != ClassState.DONE
            ) {
                updates += entity
                threads += m.threadId
            }
            val folderChanged = (oldEffective == LabelCode.SPAM) != (newEffective == LabelCode.SPAM)
            if (folderChanged) {
                moved++
                if (batchId != null) {
                    events += FeedbackEventEntity(
                        batchId = batchId, kind = FeedbackKind.AUTO_MOVE, messageId = m.id, senderKey = m.senderKey,
                        label = newEffective, prevUserLabel = null, prevEffectiveLabel = oldEffective, prevEffectiveSource = previous?.effectiveSource ?: Source.PENDING,
                        engineLabelAtMark = previous?.modelLabel, createdAt = now,
                    )
                }
            }
        }
        if (updates.isNotEmpty() || events.isNotEmpty()) {
            db.withWriteTransaction {
                val relabeled = updates.filter { db.classifications().get(it.messageId)?.userLabel != null }.map { it.messageId }.toSet()
                updates.removeAll { it.messageId in relabeled }
                db.classifications().upsertAll(updates)
                if (events.isNotEmpty()) db.feedback().insertAll(events)
                projector.rebuild(threads)
            }
        }
        Outcome(moved, threads)
    }

    companion object {
        const val WINDOW_MS = 30L * 24 * 60 * 60 * 1000
        const val HYSTERESIS = 0.05f

        /** The new effective label for an unlabeled message given its old one and a fresh verdict. */
        fun decide(oldEffective: Int, oldScore: Float?, v: Verdict): Int {
            val newLabel = LabelCode.of(v.label)
            return when (v.reason) {
                Reason.SIMILAR_TO_SPAM, Reason.SIMILAR_TO_HAM -> newLabel
                Reason.UNAVAILABLE -> oldEffective
                Reason.MODEL, Reason.MODEL_FALLBACK, Reason.OTP_GUARDRAIL -> when {
                    oldEffective == LabelCode.SPAM && newLabel != LabelCode.SPAM -> {
                        val clear = oldScore == null || v.probability.isNaN() || oldScore - v.probability >= HYSTERESIS
                        if (clear) newLabel else LabelCode.SPAM
                    }
                    oldEffective != LabelCode.SPAM && newLabel == LabelCode.SPAM -> LabelCode.SUSPICIOUS
                    else -> newLabel
                }
            }
        }
    }
}

/** Applies the user's spam / not-spam marks (the engine's only training signal), and supports Undo. */
class FeedbackService(
    private val db: AppDatabase,
    private val engine: SpamEngine,
    private val rescore: RescoreService,
    private val projector: ConversationProjector,
    private val notifier: MessageNotifier,
    private val normalizer: AddressNormalizer,
    private val appScope: CoroutineScope,
) {
    data class Result(val batchId: Long, val label: UserLabel, val marked: Int, val moved: Int)

    /**
     * Marks a conversation, which means its latest incoming message (the one shown in the list). Labels are always
     * per message: the same sender can also send messages that aren't spam, e.g. a bank's promo and its passcodes.
     */
    suspend fun markThreads(threadIds: Collection<Long>, label: UserLabel): Result {
        val ids = threadIds.flatMap { t -> db.messages().latestIncoming(t, 1).map { it.id } }.distinct()
        return markMessages(ids, label)
    }

    suspend fun markMessages(ids: Collection<Long>, label: UserLabel): Result {
        val now = System.currentTimeMillis()
        val batchId = now
        val messages = db.messages().getAll(ids.toList()).filter { it.type == MsgType.INBOX }
        if (messages.isEmpty()) return Result(batchId, label, 0, 0)
        val userCode = UserLabelCode.of(label)
        val effective = LabelCode.of(label)
        rescore.lock.withLock { db.withWriteTransaction {
            val events = ArrayList<FeedbackEventEntity>()
            for (m in messages) {
                val previous = db.classifications().get(m.id)
                events += FeedbackEventEntity(
                    batchId = batchId, kind = FeedbackKind.EXPLICIT, messageId = m.id, senderKey = m.senderKey,
                    label = userCode, prevUserLabel = previous?.userLabel, prevEffectiveLabel = previous?.effectiveLabel ?: LabelCode.HAM,
                    prevEffectiveSource = previous?.effectiveSource ?: Source.PENDING,
                    engineLabelAtMark = previous?.modelLabel, createdAt = now,
                )
                db.classifications().upsert(
                    (previous ?: ClassificationEntity(messageId = m.id, notified = true)).copy(
                        userLabel = userCode, userLabeledAt = now, effectiveLabel = effective, effectiveSource = Source.USER,
                    ),
                )
            }
            db.feedback().insertAll(events)
            projector.rebuild(messages.map { it.threadId }.toSet())
        } }
        engine.learn(messages.map { example(it, label, now) })
        if (label == UserLabel.SPAM) messages.map { it.threadId }.distinct().forEach(notifier::cancelThread)
        val outcome = rescore.afterFeedback(batchId, messages)
        scheduleRetrainRescore()
        return Result(batchId, label, messages.size, outcome.moved)
    }

    suspend fun undo(batchId: Long) {
        val events = db.feedback().activeBatch(batchId)
        if (events.isEmpty()) return
        val now = System.currentTimeMillis()
        val threads = db.messages().threadsOf(events.map { it.messageId })
        rescore.lock.withLock { db.withWriteTransaction {
            for (e in events) {
                when (e.kind) {
                    FeedbackKind.EXPLICIT -> db.classifications().setUserLabel(
                        e.messageId, e.prevUserLabel, if (e.prevUserLabel == null) null else now,
                        e.prevEffectiveLabel, e.prevEffectiveSource,
                    )
                    FeedbackKind.AUTO_MOVE -> db.classifications().setEffective(e.messageId, e.prevEffectiveLabel, e.prevEffectiveSource)
                }
            }
            db.feedback().markUndone(batchId)
            projector.rebuild(threads)
        } }
        val explicit = events.filter { it.kind == FeedbackKind.EXPLICIT }
        engine.forget(explicit.filter { it.prevUserLabel == null }.map { it.messageId })
        val relearn = explicit.filter { it.prevUserLabel != null }
        if (relearn.isNotEmpty()) {
            val byId = db.messages().getAll(relearn.map { it.messageId }).associateBy { it.id }
            engine.learn(relearn.mapNotNull { e ->
                byId[e.messageId]?.let { example(it, UserLabelCode.toUserLabel(e.prevUserLabel!!), now) }
            })
        }
        scheduleRetrainRescore()
    }

    private suspend fun example(m: MessageEntity, label: UserLabel, now: Long): TrainingExample {
        val cached = db.embeddings().get(m.id)?.takeIf { it.fingerprint == m.fingerprint && it.bundleVersion == engine.info.value.bundleVersion }
        return TrainingExample(
            messageId = m.id,
            body = m.body,
            senderKind = runCatching { SenderKind.valueOf(m.senderKind) }.getOrDefault(SenderKind.UNKNOWN),
            international = normalizer.normalize(m.address).international,
            firstContact = db.messages().earlierIncomingCount(m.senderKey, m.id, m.date) == 0,
            label = label,
            createdAt = now,
            embedding = cached?.vector?.let(EmbeddingCodec::decode),
            senderKey = m.senderKey,
        )
    }

    private fun scheduleRetrainRescore() {
        appScope.launch {
            engine.awaitIdle()
            rescore.afterRetrain()
        }
    }
}
