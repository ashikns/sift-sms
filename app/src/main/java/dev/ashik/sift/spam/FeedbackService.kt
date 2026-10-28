package dev.ashik.sift.spam

import androidx.room3.withWriteTransaction
import dev.ashik.sift.data.db.AppDatabase
import dev.ashik.sift.data.db.ClassState
import dev.ashik.sift.data.db.ClassificationEntity
import dev.ashik.sift.data.db.FeedbackEventEntity
import dev.ashik.sift.data.db.LabelCode
import dev.ashik.sift.data.db.MessageEntity
import dev.ashik.sift.data.db.MsgType
import dev.ashik.sift.data.db.Source
import dev.ashik.sift.data.db.UserLabelCode
import dev.ashik.sift.data.repo.ConversationProjector
import dev.ashik.sift.data.repo.MessageRepository
import dev.ashik.sift.notifications.MessageNotifier
import dev.ashik.sift.spam.work.WorkScheduler
import dev.ashik.sift.spamcore.Reason
import dev.ashik.sift.spamcore.SenderKind
import dev.ashik.sift.spamcore.SpamEngine
import dev.ashik.sift.spamcore.TrainingExample
import dev.ashik.sift.spamcore.UserLabel
import dev.ashik.sift.spamcore.Verdict
import dev.ashik.sift.telephony.AddressNormalizer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Re-scores unlabeled messages after the model or the user's labels change.
 * User-driven verdicts (near-duplicates of a marked message) move messages; model-only flips never move an inbox
 * message to Spam retroactively (it only gets a "Possible spam" badge), but do rescue Spam-folder false positives.
 * A message whose label changes is marked unread.
 */
class RescoreService(
    private val db: AppDatabase,
    private val coordinator: ClassificationCoordinator,
    private val projector: ConversationProjector,
    private val repository: MessageRepository,
) {
    /**
     * Serializes re-scores with the user's label changes (FeedbackService takes it too). A re-score reads a message's
     * classification, computes, then writes; without the lock a background re-score that started before a mark could
     * write back the stale state over it.
     */
    val lock = Mutex()

    /**
     * After a retrain: re-scores these messages (the marked messages' threads first, then RecheckWorker goes through the
     * rest of history in chunks). Near-duplicates of a marked message move with it here.
     */
    suspend fun afterRetrain(messageIds: List<Long>) = rescore(db.messages().getAll(messageIds))

    /**
     * Messages whose mark was deleted: each gets the model's own verdict, applied as it comes (as a rescan does, without
     * the re-score's move rules), unless it was marked again meanwhile.
     */
    suspend fun reclassify(messageIds: List<Long>) = lock.withLock {
        for (chunk in messageIds.chunked(500)) {
            for (m in db.messages().getAll(chunk)) {
                val computed = coordinator.compute(m) ?: continue
                if (computed.previous?.userLabel == null) coordinator.apply(computed)
            }
        }
    }

    private suspend fun rescore(candidates: List<MessageEntity>) = lock.withLock {
        val updates = ArrayList<ClassificationEntity>()
        val threads = HashSet<Long>()
        val changed = ArrayList<MessageEntity>()
        for (m in candidates) {
            if (m.type != MsgType.INBOX) continue
            val computed = coordinator.compute(m) ?: continue
            val previous = computed.previous
            if (previous?.userLabel != null) continue
            val oldEffective = previous?.effectiveLabel ?: LabelCode.HAM
            val newEffective = decide(oldEffective, previous?.modelScore, computed.verdict, previous?.reason)
            val entity = coordinator.entityFor(computed, newEffective)
            if (previous == null || entity.effectiveLabel != previous.effectiveLabel || entity.reason != previous.reason ||
                entity.modelScore != previous.modelScore || previous.state != ClassState.DONE
            ) {
                updates += entity
                threads += m.threadId
                if (ClassificationCoordinator.changesLabel(previous, entity)) changed += m
            }
        }
        if (updates.isNotEmpty()) {
            val relabeled = db.withWriteTransaction {
                val relabeled = updates.filter { db.classifications().get(it.messageId)?.userLabel != null }.map { it.messageId }.toSet()
                updates.removeAll { it.messageId in relabeled }
                db.classifications().upsertAll(updates)
                projector.rebuild(threads)
                relabeled
            }
            repository.markUnread(changed.filterNot { it.id in relabeled })
        }
    }

    companion object {
        const val HYSTERESIS = 0.05f

        /**
         * The new effective label for an unlabeled message given its old one and a fresh verdict. A Spam verdict that
         * came from a near-duplicate match ([oldReason] SIMILAR_TO_SPAM) has lost its basis when the fresh verdict is
         * the model's (the learning entry was deleted), so the model's label applies without the hysteresis margin.
         */
        fun decide(oldEffective: Int, oldScore: Float?, v: Verdict, oldReason: String? = null): Int {
            val newLabel = LabelCode.of(v.label)
            return when (v.reason) {
                Reason.SIMILAR_TO_SPAM, Reason.SIMILAR_TO_HAM -> newLabel
                Reason.UNAVAILABLE -> oldEffective
                Reason.MODEL, Reason.MODEL_FALLBACK, Reason.OTP_GUARDRAIL -> when {
                    oldEffective == LabelCode.SPAM && newLabel != LabelCode.SPAM -> {
                        val clear = oldReason == Reason.SIMILAR_TO_SPAM.name || oldScore == null || v.probability.isNaN() ||
                            oldScore - v.probability >= HYSTERESIS
                        if (clear) newLabel else LabelCode.SPAM
                    }
                    oldEffective != LabelCode.SPAM && newLabel == LabelCode.SPAM -> LabelCode.SUSPICIOUS
                    else -> newLabel
                }
            }
        }
    }
}

/** Applies the user's spam / not-spam marks (the engine's only training signal). */
class FeedbackService(
    private val db: AppDatabase,
    private val engine: SpamEngine,
    private val rescore: RescoreService,
    private val projector: ConversationProjector,
    private val notifier: MessageNotifier,
    private val normalizer: AddressNormalizer,
    private val scheduler: WorkScheduler,
    private val appScope: CoroutineScope,
) {
    /**
     * Marks these messages and learns from them. Returns the job that re-checks their threads once the model has
     * retrained (the rest of history follows in RecheckWorker), or null if none of them is incoming.
     */
    suspend fun markMessages(ids: Collection<Long>, label: UserLabel): Job? {
        val now = System.currentTimeMillis()
        val messages = db.messages().getAll(ids.toList()).filter { it.type == MsgType.INBOX }
        if (messages.isEmpty()) return null
        val userCode = UserLabelCode.of(label)
        val effective = LabelCode.of(label)
        val threads = messages.map { it.threadId }.toSet()
        rescore.lock.withLock { db.withWriteTransaction {
            val events = ArrayList<FeedbackEventEntity>()
            for (m in messages) {
                val previous = db.classifications().get(m.id)
                events += FeedbackEventEntity(messageId = m.id, label = userCode, engineLabelAtMark = previous?.modelLabel, createdAt = now)
                db.classifications().upsert(
                    (previous ?: ClassificationEntity(messageId = m.id, notified = true)).copy(
                        userLabel = userCode, userLabeledAt = now, effectiveLabel = effective, effectiveSource = Source.USER,
                    ),
                )
            }
            db.feedback().insertAll(events)
            projector.rebuild(threads)
        } }
        engine.learn(messages.map { example(it, label, now) })
        if (label == UserLabel.SPAM) threads.forEach(notifier::cancelThread)
        return recheckAfterRetrain(threads)
    }

    /**
     * Deletes learning entries and the marks behind them. The messages lose the user's label at once (pending, so a
     * restart's recovery pass judges them if the process dies first) and the marks stop counting; the engine forgets
     * them (its near-duplicate memory at once) and retrains. Then each message gets the retrained model's own verdict,
     * and every other message is re-checked.
     */
    suspend fun forgetLearned(messageIds: Collection<Long>) {
        if (messageIds.isEmpty()) return
        val ids = messageIds.toList()
        rescore.lock.withLock {
            db.withWriteTransaction {
                for (chunk in ids.chunked(500)) {
                    db.classifications().clearUserLabels(chunk)
                    db.feedback().retractMarks(chunk)
                }
            }
        }
        engine.forget(ids)
        appScope.launch {
            engine.awaitIdle()
            rescore.reclassify(ids)
            scheduler.enqueueRecheck()
        }
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

    /**
     * Once the engine has retrained (after any mark that came in meanwhile, too): re-checks the unlabeled messages in
     * [first] here, so the open thread shows the new verdicts first, then hands the rest of history to RecheckWorker.
     */
    private fun recheckAfterRetrain(first: Set<Long>): Job = appScope.launch {
        engine.awaitIdle()
        db.messages().unlabeledIncomingIdsIn(first.toList()).chunked(64).forEach { rescore.afterRetrain(it) }
        scheduler.enqueueRecheck(skipThreads = first)
    }
}
