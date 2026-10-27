package dev.ashik.sift.debug

import android.Manifest
import android.content.ClipboardManager
import android.content.ContentProvider
import android.content.ContentValues
import android.database.Cursor
import android.database.MatrixCursor
import android.net.Uri
import android.os.Bundle
import android.provider.Telephony
import android.telephony.SubscriptionManager
import android.util.Log
import dev.ashik.sift.SiftApp
import dev.ashik.sift.data.db.ClassState
import dev.ashik.sift.data.db.LabelCode
import dev.ashik.sift.di.AppGraph
import dev.ashik.sift.spamcore.UserLabel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull

/**
 * Debug-only window into app state for `scripts/e2e.sh`:
 *
 *     adb shell content query --uri content://dev.ashik.sift.debug/conversations
 *     adb shell content query --uri content://dev.ashik.sift.debug/messages
 *     adb shell content call --uri content://dev.ashik.sift.debug --method awaitIdle
 *     adb shell content call --uri content://dev.ashik.sift.debug --method markSpam --arg <messageId | thread:<id>>
 *     adb shell content call --uri content://dev.ashik.sift.debug --method resetLearning
 *     adb shell content call --uri content://dev.ashik.sift.debug --method seedHistory --arg <count>
 *     adb shell content call --uri content://dev.ashik.sift.debug --method seedStats
 *     adb shell content call --uri content://dev.ashik.sift.debug --method rescan
 *     adb shell content call --uri content://dev.ashik.sift.debug --method clipboard   (while Sift is in the foreground)
 *
 * Guarded by android.permission.DUMP (adb shell holds it; third-party apps can't).
 */
class DebugStateProvider : ContentProvider() {
    private val graph: AppGraph get() = (context!!.applicationContext as SiftApp).graph

    override fun onCreate() = true

    private fun enforce() = context!!.enforceCallingOrSelfPermission(Manifest.permission.DUMP, "DebugStateProvider")

    override fun query(uri: Uri, projection: Array<out String>?, selection: String?, args: Array<out String>?, sort: String?): Cursor? {
        enforce()
        return runBlocking(Dispatchers.IO) {
            when (uri.pathSegments.firstOrNull()) {
                "conversations" -> MatrixCursor(
                    arrayOf("threadId", "address", "senderKey", "folder", "latestLabel", "unreadCount", "messageCount", "latestIncomingId", "snippet"),
                ).apply {
                    graph.db.conversations().all().forEach {
                        addRow(arrayOf<Any?>(it.threadId, it.address, it.senderKey, it.folder, it.latestLabel, it.unreadCount, it.messageCount, it.latestIncomingId, it.snippet.take(60)))
                    }
                }
                "messages" -> MatrixCursor(
                    arrayOf("id", "threadId", "address", "type", "status", "effectiveLabel", "source", "reason", "score", "userLabel", "classState", "body"),
                ).apply {
                    graph.db.messages().debugMessages(300).forEach {
                        addRow(arrayOf<Any?>(it.id, it.threadId, it.address, it.type, it.status, it.effectiveLabel, it.effectiveSource, it.reason, it.modelScore, it.userLabel, it.classState, it.body.take(200)))
                    }
                }
                "engine" -> MatrixCursor(arrayOf("bundle", "modelVersion", "examples", "encoder")).apply {
                    val i = graph.engine.info.value
                    addRow(arrayOf<Any?>(i.bundleVersion, i.modelVersion, i.examples, i.encoderAvailable))
                }
                else -> null
            }
        }
    }

    override fun call(method: String, arg: String?, extras: Bundle?): Bundle {
        enforce()
        return runBlocking(Dispatchers.Default) {
            val out = Bundle()
            try {
                when (method) {
                    "awaitIdle" -> out.putBoolean("idle", awaitIdle())
                    "markSpam", "markNotSpam" -> {
                        val label = if (method == "markSpam") UserLabel.SPAM else UserLabel.HAM
                        val target = arg.orEmpty()
                        val result = if (target.startsWith("thread:")) {
                            graph.feedback.markThreads(listOf(target.removePrefix("thread:").toLong()), label)
                        } else {
                            graph.feedback.markMessages(listOf(target.toLong()), label)
                        }
                        out.putLong("batchId", result.batchId)
                        out.putInt("marked", result.marked)
                        out.putInt("moved", result.moved)
                    }
                    "undo" -> graph.feedback.undo(arg!!.toLong())
                    "resetLearning" -> graph.engine.reset()
                    "reconcile" -> graph.sync.fullReconcile()
                    "incremental" -> graph.sync.incremental()
                    "seedHistory" -> out.putInt("seeded", seedHistory(arg?.toIntOrNull() ?: 800))
                    "seedStats" -> seedStats(out)
                    "rescan" -> graph.scheduler.enqueueBackfill(rescan = true)
                    "clipboard" -> out.putString(
                        "text", context!!.getSystemService(ClipboardManager::class.java).primaryClip?.getItemAt(0)?.text?.toString(),
                    )
                    "counts" -> {
                        out.putInt("messages", graph.db.messages().count())
                        out.putInt("provider", graph.gateway.count())
                        out.putInt("pending", graph.db.classifications().pendingCount())
                    }
                    else -> out.putString("error", "unknown method $method")
                }
            } catch (t: Throwable) {
                Log.e("SiftDebug", "call $method failed", t)
                out.putString("error", t.toString())
            }
            out
        }
    }

    /** Waits (up to 20 s) until no classification is pending and the engine has finished retraining. */
    private suspend fun awaitIdle(): Boolean = withTimeoutOrNull(20_000) {
        while (graph.db.classifications().pendingCount() > 0) delay(100)
        graph.engine.awaitIdle()
        delay(300)
        true
    } ?: false

    /**
     * Replaces earlier seeds with [count] incoming messages from 100 to 400 days ago, each from its own sender, scam
     * texts at even indices and ordinary ones at odd. They're imported like history, so only a rescan classifies them.
     */
    private suspend fun seedHistory(count: Int): Int {
        val resolver = context!!.contentResolver
        withContext(Dispatchers.IO) {
            resolver.delete(Telephony.Sms.CONTENT_URI, "${Telephony.Sms.ADDRESS} LIKE ?", arrayOf("$SEED_PREFIX%"))
            val now = System.currentTimeMillis()
            val subId = SubscriptionManager.getDefaultSmsSubscriptionId()
            for (i in 0 until count) {
                val date = now - (100 + 300L * i / count) * DAY_MS
                graph.gateway.insertIncoming(SEED_PREFIX + "%04d".format(i), seedBody(i), date, date, subId, 0, false, null)
            }
        }
        graph.sync.fullReconcile()
        return count
    }

    private fun seedBody(i: Int): String {
        val n = i / 2
        return if (i % 2 == 0) SEED_SCAMS[n % SEED_SCAMS.size].format(n) else SEED_ORDINARY[n % SEED_ORDINARY.size].format(n)
    }

    /** How many seeded messages have a verdict, and how many of each kind are effectively spam. */
    private suspend fun seedStats(out: Bundle) {
        val ids = withContext(Dispatchers.IO) {
            context!!.contentResolver.query(
                Telephony.Sms.CONTENT_URI, arrayOf(Telephony.Sms._ID), "${Telephony.Sms.ADDRESS} LIKE ?", arrayOf("$SEED_PREFIX%"), null,
            )?.use { c -> buildList { while (c.moveToNext()) add(c.getLong(0)) } }.orEmpty()
        }
        val messages = ids.chunked(500).flatMap { graph.db.messages().getAll(it) }
        val verdicts = ids.chunked(500).flatMap { graph.db.classifications().getAll(it) }.associateBy { it.messageId }
        var classified = 0
        var scams = 0
        var scamsFlagged = 0
        var ordinaryFlagged = 0
        for (m in messages) {
            val v = verdicts[m.id]
            if (v?.state == ClassState.DONE) classified++
            val flagged = v?.effectiveLabel == LabelCode.SPAM
            if (m.address.takeLast(4).toInt() % 2 == 0) {
                scams++
                if (flagged) scamsFlagged++
            } else if (flagged) {
                ordinaryFlagged++
            }
        }
        out.putInt("seeded", messages.size)
        out.putInt("classified", classified)
        out.putInt("scams", scams)
        out.putInt("scamsFlagged", scamsFlagged)
        out.putInt("ordinaryFlagged", ordinaryFlagged)
    }

    override fun getType(uri: Uri): String? = null
    override fun insert(uri: Uri, values: ContentValues?): Uri? = null
    override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?) = 0
    override fun update(uri: Uri, values: ContentValues?, selection: String?, selectionArgs: Array<out String>?) = 0

    private companion object {
        const val SEED_PREFIX = "+1555019"
        const val DAY_MS = 24L * 60 * 60 * 1000
        val SEED_SCAMS = listOf(
            "USPS: your parcel is held due to an incomplete address. Update within 24h: https://usps-redeliver.top/t%d",
            "Toll Services: unpaid toll balance of \$6.99. Avoid a late fee: https://ezpass-billing.click/pay%d",
            "Congrats! You won a \$500 Walmart gift card. Claim now: https://bit.ly/3xYz%d",
        )
        val SEED_ORDINARY = listOf(
            "Your Chase verification code is %06d. Do not share it with anyone.",
            "CVS Pharmacy: your prescription #%d is ready for pickup at 123 Main St.",
            "Delta: flight DL%d to SEA now departs 6:45 PM from gate B12.",
        )
    }
}
