package dev.ashik.sift.outgoing

import android.app.Activity
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.telephony.SmsManager
import android.telephony.SubscriptionManager
import android.util.Log
import dev.ashik.sift.data.db.AppDatabase
import dev.ashik.sift.data.db.MsgStatus
import dev.ashik.sift.data.db.MsgType
import dev.ashik.sift.data.db.PendingSendEntity
import dev.ashik.sift.data.repo.MessageRepository
import dev.ashik.sift.data.settings.SettingsRepository
import dev.ashik.sift.notifications.MessageNotifier
import dev.ashik.sift.telephony.SmsProviderGateway
import dev.ashik.sift.telephony.receiver.SmsSendResultReceiver
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/** Per-part send bookkeeping. Pure, so it's unit-tested. */
object PartAggregator {
    enum class Outcome { PENDING, SENT, FAILED }

    fun allBits(total: Int): Long = if (total >= 64) -1L else (1L shl total) - 1

    fun outcome(total: Int, okMask: Long, failedMask: Long): Outcome {
        val all = allBits(total)
        return when {
            (okMask or failedMask) and all != all -> Outcome.PENDING
            failedMask and all != 0L -> Outcome.FAILED
            else -> Outcome.SENT
        }
    }

    /** TP-Status (3GPP 23.040): 0x00-0x1F delivered, 0x20-0x3F still trying, 0x40+ failed. */
    fun deliveryStatus(tpStatus: Int): Int = when {
        tpStatus < 0x20 -> MsgStatus.COMPLETE
        tpStatus < 0x40 -> MsgStatus.PENDING
        else -> MsgStatus.FAILED
    }
}

/**
 * Sends SMS. The default app must persist its own sent messages: insert as OUTBOX first, send with one
 * explicit, mutable, one-shot PendingIntent per part (mutable so the system can fill in `errorCode`), then
 * [SendResultProcessor] flips the row to SENT or FAILED once every part has reported.
 */
class SmsSender(
    private val context: Context,
    private val gateway: SmsProviderGateway,
    private val repository: MessageRepository,
    private val db: AppDatabase,
    private val settings: SettingsRepository,
    private val results: SendResultProcessor,
) {
    /** Returns the provider id of the new message. Throws [dev.ashik.sift.telephony.RoleLostException]. */
    suspend fun send(address: String, body: String, subIdHint: Int? = null): Long {
        val now = System.currentTimeMillis()
        val deliveryReports = settings.current().deliveryReports
        val subId = subIdHint?.takeIf { it >= 0 } ?: SubscriptionManager.getDefaultSmsSubscriptionId()
        val id = withContext(Dispatchers.IO) { gateway.insertOutgoing(address, body, now, subId, deliveryReports) }
        val message = repository.mirrorById(id, notify = false)
        dispatch(id, address, body, subId, deliveryReports)
        return id
    }

    suspend fun retry(id: Long) {
        val m = db.messages().get(id) ?: return
        if (m.type != MsgType.FAILED) return
        val deliveryReports = settings.current().deliveryReports
        withContext(Dispatchers.IO) {
            gateway.updateType(id, MsgType.OUTBOX, 0)
            gateway.updateDate(id, System.currentTimeMillis())
        }
        repository.mirrorById(id, notify = false)
        dispatch(id, m.address, m.body, m.subId, deliveryReports)
    }

    private suspend fun dispatch(id: Long, address: String, body: String, subId: Int, deliveryReports: Boolean) {
        val manager = smsManager(subId)
        val parts = manager.divideMessage(body)
        db.pendingSends().upsert(
            PendingSendEntity(messageId = id, partsTotal = parts.size, deliveryReports = deliveryReports, createdAt = System.currentTimeMillis()),
        )
        val sent = ArrayList(parts.indices.map { resultIntent(SmsSendResultReceiver.ACTION_SENT, id, it) })
        val delivered = if (deliveryReports) ArrayList(parts.indices.map { resultIntent(SmsSendResultReceiver.ACTION_DELIVERED, id, it) }) else null
        try {
            manager.sendMultipartTextMessage(address, null, parts, sent, delivered, id)
        } catch (e: Exception) {
            Log.e(TAG, "sendMultipartTextMessage failed for $id", e)
            results.markFailed(id, SmsManager.RESULT_ERROR_GENERIC_FAILURE)
        }
    }

    private fun smsManager(subId: Int): SmsManager {
        val manager = context.getSystemService(SmsManager::class.java)
        return if (SubscriptionManager.isValidSubscriptionId(subId)) manager.createForSubscriptionId(subId) else manager
    }

    private fun resultIntent(action: String, id: Long, part: Int): PendingIntent {
        val intent = Intent(action, Uri.parse("smsresult://$id/$part"), context, SmsSendResultReceiver::class.java)
        return PendingIntent.getBroadcast(context, 0, intent, PendingIntent.FLAG_MUTABLE or PendingIntent.FLAG_ONE_SHOT)
    }

    private companion object {
        const val TAG = "SiftSend"
    }
}

/** Collects per-part sent/delivered results and finalizes the provider row. Serialized by a mutex. */
class SendResultProcessor(
    private val db: AppDatabase,
    private val gateway: SmsProviderGateway,
    private val repository: MessageRepository,
    private val notifier: MessageNotifier,
) {
    private val mutex = Mutex()

    suspend fun onSent(id: Long, part: Int, resultCode: Int, errorCode: Int) = mutex.withLock {
        val ok = resultCode == Activity.RESULT_OK
        if (!ok) Log.w(TAG, "part $part of $id failed: result=$resultCode errorCode=$errorCode")
        val row = db.pendingSends().get(id)
        if (row == null) {
            finalize(id, if (ok) PartAggregator.Outcome.SENT else PartAggregator.Outcome.FAILED, resultCode, keepForDelivery = false)
            return@withLock
        }
        val bit = 1L shl part.coerceIn(0, 63)
        val updated = row.copy(
            sentMask = if (ok) row.sentMask or bit else row.sentMask,
            failedMask = if (!ok) row.failedMask or bit else row.failedMask,
            lastError = if (!ok) resultCode else row.lastError,
        )
        db.pendingSends().upsert(updated)
        val outcome = PartAggregator.outcome(updated.partsTotal, updated.sentMask, updated.failedMask)
        if (outcome != PartAggregator.Outcome.PENDING) {
            finalize(id, outcome, updated.lastError, keepForDelivery = outcome == PartAggregator.Outcome.SENT && updated.deliveryReports)
        }
    }

    suspend fun onDelivered(id: Long, part: Int, tpStatus: Int) = mutex.withLock {
        val row = db.pendingSends().get(id) ?: return@withLock
        val status = PartAggregator.deliveryStatus(tpStatus)
        if (status == MsgStatus.PENDING) return@withLock
        val bit = 1L shl part.coerceIn(0, 63)
        val updated = row.copy(
            deliveredMask = if (status == MsgStatus.COMPLETE) row.deliveredMask or bit else row.deliveredMask,
            deliveryFailedMask = if (status == MsgStatus.FAILED) row.deliveryFailedMask or bit else row.deliveryFailedMask,
        )
        db.pendingSends().upsert(updated)
        when (PartAggregator.outcome(updated.partsTotal, updated.deliveredMask, updated.deliveryFailedMask)) {
            PartAggregator.Outcome.PENDING -> Unit
            PartAggregator.Outcome.SENT -> setStatus(id, MsgStatus.COMPLETE)
            PartAggregator.Outcome.FAILED -> setStatus(id, MsgStatus.FAILED)
        }
    }

    suspend fun markFailed(id: Long, resultCode: Int) = mutex.withLock {
        finalize(id, PartAggregator.Outcome.FAILED, resultCode, keepForDelivery = false)
    }

    private suspend fun setStatus(id: Long, status: Int) {
        withContext(Dispatchers.IO) { runCatching { gateway.updateStatus(id, status) } }
        db.pendingSends().delete(id)
        repository.mirrorById(id, notify = false)
    }

    private suspend fun finalize(id: Long, outcome: PartAggregator.Outcome, resultCode: Int, keepForDelivery: Boolean) {
        val type = if (outcome == PartAggregator.Outcome.SENT) MsgType.SENT else MsgType.FAILED
        withContext(Dispatchers.IO) {
            runCatching { gateway.updateType(id, type, if (type == MsgType.FAILED) resultCode else null) }
                .onFailure { Log.e(TAG, "could not finalize $id", it) }
        }
        if (!keepForDelivery) db.pendingSends().delete(id)
        val m = repository.mirrorById(id, notify = false)
        if (type == MsgType.FAILED && m != null) notifier.notifySendFailed(m.threadId, m.address)
    }

    private companion object {
        const val TAG = "SiftSend"
    }
}

/**
 * Force-stopping the app cancels its PendingIntents, so some send results never arrive. At startup, outbox rows
 * older than 10 minutes with no live send are marked FAILED so the user can retry them.
 */
class OutboxRecovery(
    private val db: AppDatabase,
    private val results: SendResultProcessor,
) {
    suspend fun run() {
        val cutoff = System.currentTimeMillis() - STALE_MS
        for (m in db.messages().staleOutbox(cutoff)) {
            val pending = db.pendingSends().get(m.id)
            if (pending == null || pending.createdAt < cutoff) results.markFailed(m.id, SmsManager.RESULT_ERROR_GENERIC_FAILURE)
        }
    }

    private companion object {
        const val STALE_MS = 10L * 60 * 1000
    }
}
