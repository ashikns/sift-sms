package dev.ashik.sift.telephony.receiver

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.SystemClock
import android.provider.Telephony
import android.util.Log
import dev.ashik.sift.SiftApp
import dev.ashik.sift.di.AppGraph
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeout

private const val TAG = "SiftReceiver"

/** Runs [block] off the main thread within the broadcast's time budget, always releasing the broadcast. */
private fun BroadcastReceiver.runAsync(context: Context, timeoutMs: Long, block: suspend (AppGraph) -> Unit) {
    val graph = (context.applicationContext as SiftApp).graph
    val pending = goAsync()
    graph.appScope.launch {
        try {
            withTimeout(timeoutMs) { block(graph) }
        } catch (t: Throwable) {
            Log.e(TAG, "${this@runAsync.javaClass.simpleName} failed", t)
        } finally {
            pending.finish()
        }
    }
}

/** SMS_DELIVER: only the default SMS app receives it, and it must persist the message itself. */
class SmsDeliverReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Telephony.Sms.Intents.SMS_DELIVER_ACTION) return
        val started = SystemClock.elapsedRealtime()
        // finish() releases the next queued SMS broadcast, so the whole job is bounded at 8 s.
        runAsync(context, 8_000) { it.incoming.handle(intent, started) }
    }
}

/** Required to qualify for the SMS role. Sift doesn't support MMS, so it just says so. */
class MmsWapPushReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Telephony.Sms.Intents.WAP_PUSH_DELIVER_ACTION) return
        (context.applicationContext as SiftApp).graph.notifier.notifyMmsUnsupported()
    }
}

/** The system tells the old and new default SMS apps when the role moves. Extras aren't trusted: re-check. */
class DefaultSmsChangedReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Telephony.Sms.Intents.ACTION_DEFAULT_SMS_PACKAGE_CHANGED) return
        val graph = (context.applicationContext as SiftApp).graph
        if (graph.role.refresh()) {
            graph.observer.register()
            // Catch up right away on what arrived while another app was default; WorkManager is the fallback
            // for an inbox too big to reconcile within the broadcast's budget.
            graph.scheduler.enqueueSync()
            runAsync(context, 8_000) { it.sync.fullReconcile() }
        } else {
            graph.observer.unregister()
        }
    }
}

/** Another app (or the system) wrote to the SMS provider while Sift is the default app. */
class ExternalProviderChangeReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Telephony.Sms.Intents.ACTION_EXTERNAL_PROVIDER_CHANGE) return
        runAsync(context, 8_000) { it.sync.fullReconcile() }
    }
}

/** Mark read from a notification. */
class NotificationActionReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val threadId = intent.getLongExtra(EXTRA_THREAD_ID, -1L).takeIf { it > 0 } ?: return
        when (intent.action) {
            ACTION_MARK_READ -> runAsync(context, 8_000) { graph ->
                graph.repository.markThreadRead(threadId)
                graph.notifier.cancelThread(threadId)
            }
        }
    }

    companion object {
        const val ACTION_MARK_READ = "dev.ashik.sift.action.MARK_READ"
        const val EXTRA_THREAD_ID = "threadId"
    }
}
