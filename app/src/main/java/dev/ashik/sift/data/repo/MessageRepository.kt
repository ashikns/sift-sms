package dev.ashik.sift.data.repo

import androidx.room3.withWriteTransaction
import dev.ashik.sift.data.db.AppDatabase
import dev.ashik.sift.data.db.ClassState
import dev.ashik.sift.data.db.ClassificationEntity
import dev.ashik.sift.data.db.MessageEntity
import dev.ashik.sift.data.db.MsgType
import dev.ashik.sift.data.db.Source
import dev.ashik.sift.telephony.AddressNormalizer
import dev.ashik.sift.telephony.ProviderSms
import dev.ashik.sift.telephony.SmsProviderGateway
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** Pure helpers shared by the mirror and the reconciler. */
object MessageMapping {
    /** Identifies the message an `_id` points at; a mismatch means the provider reused the id. */
    fun fingerprint(address: String, date: Long): Long {
        var h = 1125899906842597L
        for (c in address) h = 31 * h + c.code
        return mix64(h xor date)
    }

    fun mutableHash(row: ProviderSms): Int = listOf(row.type, row.read, row.status, row.threadId).hashCode()

    private fun mix64(x0: Long): Long {
        var x = x0
        x = (x xor (x ushr 33)) * -0xae502812aa7333L
        x = (x xor (x ushr 33)) * -0x3b314601e57a13adL
        return x xor (x ushr 33)
    }
}

/** Rebuilds the denormalized `conversations` rows. Call inside a write transaction. */
class ConversationProjector(private val db: AppDatabase) {
    suspend fun rebuild(threadIds: Collection<Long>) {
        for (chunk in threadIds.distinct().chunked(500)) {
            db.conversations().deleteThreads(chunk)
            db.conversations().insertProjection(chunk)
        }
    }
}

/** Mirrors provider rows into Room. The provider stays the source of truth for message content. */
class MessageRepository(
    private val db: AppDatabase,
    private val gateway: SmsProviderGateway,
    private val normalizer: AddressNormalizer,
    private val projector: ConversationProjector,
) {
    fun toEntity(row: ProviderSms, body: String): MessageEntity {
        val sender = normalizer.normalize(row.address)
        return MessageEntity(
            id = row.id,
            threadId = row.threadId,
            address = row.address,
            senderKey = sender.key,
            senderKind = sender.kind.name,
            body = body,
            date = row.date,
            type = row.type,
            status = row.status,
            read = row.read,
            fingerprint = MessageMapping.fingerprint(row.address, row.date),
            mutableHash = MessageMapping.mutableHash(row),
        )
    }

    /**
     * Inserts or refreshes provider rows (with bodies) in one transaction. With [notify] (a message that just
     * arrived), incoming rows get a PENDING classification that may notify; imported or reconciled rows get no
     * classification row until the backfill classifies them, and never notify.
     */
    suspend fun mirror(rows: List<ProviderSms>, notify: Boolean): List<MessageEntity> {
        if (rows.isEmpty()) return emptyList()
        val entities = rows.map { toEntity(it, it.body.orEmpty()) }
        db.withWriteTransaction {
            val previous = db.messages().getAll(entities.map { it.id }).associateBy { it.id }
            val stale = previous.values.filter { old ->
                val new = entities.first { it.id == old.id }
                old.fingerprint != new.fingerprint
            }.map { it.id }
            if (stale.isNotEmpty()) {
                db.classifications().deleteIds(stale)
                db.embeddings().deleteIds(stale)
            }
            db.messages().upsertAll(entities)
            if (notify) {
                val existing = db.classifications().getAll(entities.map { it.id }).map { it.messageId }.toSet()
                val pending = entities.filter { it.type == MsgType.INBOX && it.id !in existing }.map {
                    ClassificationEntity(messageId = it.id, state = ClassState.PENDING, notified = false, effectiveSource = Source.PENDING)
                }
                if (pending.isNotEmpty()) db.classifications().upsertAll(pending)
            }
            val threads = entities.map { it.threadId } + previous.values.map { it.threadId }
            projector.rebuild(threads.toSet())
        }
        return entities
    }

    suspend fun mirrorById(id: Long, notify: Boolean): MessageEntity? {
        val row = withContext(Dispatchers.IO) { gateway.query(id) } ?: return null
        return mirror(listOf(row), notify).firstOrNull()
    }

    suspend fun markThreadRead(threadId: Long) {
        withContext(Dispatchers.IO) { runCatching { gateway.markThreadRead(threadId) } }
        db.withWriteTransaction {
            if (db.messages().markThreadRead(threadId) > 0) projector.rebuild(listOf(threadId))
        }
    }

    suspend fun deleteMessages(ids: List<Long>) {
        if (ids.isEmpty()) return
        withContext(Dispatchers.IO) { ids.forEach { gateway.deleteMessage(it) } }
        removeLocal(ids)
    }

    suspend fun deleteThreads(threadIds: List<Long>) {
        for (threadId in threadIds) {
            withContext(Dispatchers.IO) { gateway.deleteThreadSms(threadId) }
            removeLocal(db.messages().idsInThread(threadId))
        }
    }

    /** Drops Room rows (and their verdicts and cached embeddings) for messages gone from the provider. */
    suspend fun removeLocal(ids: List<Long>) {
        if (ids.isEmpty()) return
        db.withWriteTransaction {
            val threads = ids.chunked(500).flatMap { db.messages().threadsOf(it) }
            ids.chunked(500).forEach { chunk ->
                db.messages().deleteIds(chunk)
                db.classifications().deleteIds(chunk)
                db.embeddings().deleteIds(chunk)
            }
            projector.rebuild(threads)
        }
    }
}
