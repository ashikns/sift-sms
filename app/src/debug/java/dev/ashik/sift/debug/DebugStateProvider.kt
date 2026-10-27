package dev.ashik.sift.debug

import android.Manifest
import android.content.ContentProvider
import android.content.ContentValues
import android.database.Cursor
import android.database.MatrixCursor
import android.net.Uri
import android.os.Bundle
import android.util.Log
import dev.ashik.sift.SiftApp
import dev.ashik.sift.di.AppGraph
import dev.ashik.sift.spamcore.UserLabel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull

/**
 * Debug-only window into app state for `scripts/e2e.sh`:
 *
 *     adb shell content query --uri content://dev.ashik.sift.debug/conversations
 *     adb shell content query --uri content://dev.ashik.sift.debug/messages
 *     adb shell content call --uri content://dev.ashik.sift.debug --method awaitIdle
 *     adb shell content call --uri content://dev.ashik.sift.debug --method markSpam --arg <messageId | thread:<id>>
 *     adb shell content call --uri content://dev.ashik.sift.debug --method resetLearning
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

    override fun getType(uri: Uri): String? = null
    override fun insert(uri: Uri, values: ContentValues?): Uri? = null
    override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?) = 0
    override fun update(uri: Uri, values: ContentValues?, selection: String?, selectionArgs: Array<out String>?) = 0
}
