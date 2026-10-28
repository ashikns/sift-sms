package dev.ashik.sift.incoming

import android.content.Intent
import android.os.SystemClock
import android.provider.Telephony
import android.telephony.SmsMessage
import android.telephony.SubscriptionManager
import android.util.Log
import dev.ashik.sift.data.db.AppDatabase
import dev.ashik.sift.data.db.ClassState
import dev.ashik.sift.data.db.ClassificationEntity
import dev.ashik.sift.data.db.Folder
import dev.ashik.sift.data.db.LabelCode
import dev.ashik.sift.data.db.MessageEntity
import dev.ashik.sift.data.repo.MessageRepository
import dev.ashik.sift.data.settings.SettingsRepository
import dev.ashik.sift.notifications.MessageNotifier
import dev.ashik.sift.notifications.VisibleThreadTracker
import dev.ashik.sift.spam.ClassificationCoordinator
import dev.ashik.sift.spam.work.WorkScheduler
import dev.ashik.sift.telephony.SmsProviderGateway
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull

/** One SMS as delivered by SMS_DELIVER (the framework already joined multipart segments into one intent). */
data class IncomingSms(
    val address: String,
    val body: String,
    val dateSent: Long,
    val subId: Int,
    val protocolId: Int,
    val replyPathPresent: Boolean,
    val serviceCenter: String?,
    val isClass0: Boolean,
)

object IncomingSmsParser {
    fun parse(intent: Intent): IncomingSms? {
        val parts = Telephony.Sms.Intents.getMessagesFromIntent(intent)?.filterNotNull().orEmpty()
        if (parts.isEmpty()) return null
        val first = parts[0]
        val address = first.displayOriginatingAddress ?: first.originatingAddress ?: return null
        val body = parts.joinToString("") { it.displayMessageBody ?: it.messageBody ?: "" }
        val subId = intent.getIntExtra(
            SubscriptionManager.EXTRA_SUBSCRIPTION_INDEX,
            intent.getIntExtra("subscription", SubscriptionManager.INVALID_SUBSCRIPTION_ID),
        )
        return IncomingSms(
            address = address,
            body = body,
            dateSent = first.timestampMillis,
            subId = subId,
            protocolId = first.protocolIdentifier,
            replyPathPresent = first.isReplyPathPresent,
            serviceCenter = first.serviceCenterAddress,
            isClass0 = first.messageClass == SmsMessage.MessageClass.CLASS_0,
        )
    }
}

/**
 * Handles one SMS_DELIVER. Order matters: persist to the provider first (a message is never lost, even if
 * classification fails), mirror into Room (it shows in the inbox immediately), classify with a deadline, then
 * notify. A verdict that misses the deadline is applied when it lands (and by [ClassifyPendingWorker] as backup).
 */
class IncomingSmsHandler(
    private val db: AppDatabase,
    private val gateway: SmsProviderGateway,
    private val repository: MessageRepository,
    private val coordinator: ClassificationCoordinator,
    private val notifier: MessageNotifier,
    private val settings: SettingsRepository,
    private val visible: VisibleThreadTracker,
    private val scheduler: WorkScheduler,
    private val appScope: CoroutineScope,
) {
    suspend fun handle(intent: Intent, startedAt: Long) {
        val sms = IncomingSmsParser.parse(intent) ?: return
        val now = System.currentTimeMillis()
        val id = withContext(Dispatchers.IO) {
            gateway.findDuplicateIncoming(sms.address, sms.dateSent, sms.body)
                ?: gateway.insertIncoming(
                    sms.address, sms.body, now, sms.dateSent, sms.subId, sms.protocolId, sms.replyPathPresent, sms.serviceCenter,
                )
        }
        val message = repository.mirrorById(id, notify = true) ?: return
        if (db.classifications().get(id)?.state == ClassState.DONE) return // re-delivered duplicate, already handled

        val verdict: Deferred<ClassificationEntity?> = appScope.async {
            coordinator.compute(message)?.let { coordinator.apply(it) }
        }
        val elapsed = SystemClock.elapsedRealtime() - startedAt
        val wait = (BUDGET_MS - elapsed - SAFETY_MS).coerceIn(MIN_WAIT_MS, MAX_WAIT_MS)
        val applied = withTimeoutOrNull(wait) { verdict.await() }
        if (applied == null) {
            Log.w(TAG, "verdict for $id missed the ${wait}ms deadline; notifying fail-open")
            scheduler.enqueueClassifyPending()
            appScope.launch {
                val late = runCatching { verdict.await() }.getOrNull()
                if (late?.effectiveLabel == LabelCode.SPAM) notifier.cancelThread(message.threadId)
            }
        }
        postArrival(message, applied?.effectiveLabel ?: LabelCode.HAM)
    }

    /** Marks [m] read if its thread is open in the tab it landed in; otherwise notifies (spam only on request). */
    private suspend fun postArrival(m: MessageEntity, label: Int) {
        val folder = if (label == LabelCode.SPAM) Folder.SPAM else Folder.INBOX
        if (visible.isVisible(m.threadId, folder)) {
            repository.markRead(m.threadId, folder)
        } else if (label == LabelCode.SPAM) {
            if (settings.current().spamNotifications) notifier.notifySpam(m.threadId)
        } else {
            notifier.notifyThread(m.threadId, alert = true)
        }
        db.classifications().markNotified(m.id)
    }

    private companion object {
        const val TAG = "SiftIncoming"
        const val BUDGET_MS = 8_000L
        const val SAFETY_MS = 2_000L
        const val MIN_WAIT_MS = 250L
        const val MAX_WAIT_MS = 3_000L
    }
}
