package dev.ashik.sift.spam.work

import android.content.Context
import android.content.pm.ServiceInfo
import android.os.SystemClock
import android.util.Log
import androidx.work.BackoffPolicy
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.ForegroundInfo
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.OutOfQuotaPolicy
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import dev.ashik.sift.R
import dev.ashik.sift.SiftApp
import dev.ashik.sift.data.db.ClassState
import dev.ashik.sift.data.db.LabelCode
import dev.ashik.sift.di.AppGraph
import dev.ashik.sift.spam.FolderMoves
import dev.ashik.sift.spamcore.Reason
import java.util.concurrent.TimeUnit
import kotlin.coroutines.cancellation.CancellationException

/**
 * Enqueues the app's background work (all unique). A rescan or re-check runs in the foreground; a backfill of a big
 * history can outlast a job's 10 minutes and continues where it stopped.
 */
class WorkScheduler(private val context: Context) {
    private val wm get() = WorkManager.getInstance(context)

    fun enqueueSync(markInitialDone: Boolean = false) {
        val request = OneTimeWorkRequestBuilder<SyncWorker>()
            .setInputData(workDataOf(SyncWorker.KEY_INITIAL to markInitialDone))
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
            .build()
        wm.enqueueUniqueWork("sync", ExistingWorkPolicy.KEEP, request)
    }

    fun enqueueBackfill(rescan: Boolean = false) {
        val request = OneTimeWorkRequestBuilder<BackfillClassifyWorker>()
            .setInputData(workDataOf(BackfillClassifyWorker.KEY_RESCAN to rescan, BackfillClassifyWorker.KEY_STARTED to System.currentTimeMillis()))
            .build()
        wm.enqueueUniqueWork(BACKFILL, if (rescan) ExistingWorkPolicy.REPLACE else ExistingWorkPolicy.KEEP, request)
    }

    /**
     * Re-checks all history with the model a retrain just produced, except [skipThreads] (already re-checked with it),
     * replacing a re-check still running for an older one.
     */
    fun enqueueRecheck(skipThreads: Collection<Long> = emptyList()) {
        val request = OneTimeWorkRequestBuilder<RecheckWorker>()
            .setInputData(workDataOf(RecheckWorker.KEY_SKIP_THREADS to skipThreads.toLongArray()))
            .build()
        wm.enqueueUniqueWork("recheck", ExistingWorkPolicy.REPLACE, request)
    }

    fun enqueueClassifyPending() {
        val request = OneTimeWorkRequestBuilder<ClassifyPendingWorker>()
            .setExpedited(OutOfQuotaPolicy.RUN_AS_NON_EXPEDITED_WORK_REQUEST)
            .build()
        wm.enqueueUniqueWork("classify-pending", ExistingWorkPolicy.APPEND_OR_REPLACE, request)
    }

    companion object {
        const val BACKFILL = "backfill-classify"
    }
}

class SyncWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val graph = (applicationContext as SiftApp).graph
        if (!graph.role.isHeldNow()) return Result.success()
        graph.sync.fullReconcile()
        if (inputData.getBoolean(KEY_INITIAL, false)) {
            graph.settings.setInitialImportDone(true)
            graph.contacts.refreshAll()
            graph.scheduler.enqueueBackfill()
        }
        return Result.success()
    }

    companion object {
        const val KEY_INITIAL = "initial"
    }
}

/**
 * Classifies every incoming message that has no verdict yet, newest first, or rescans every incoming message on
 * request. A rescan shows its progress in a notification, runs as a foreground service when Android
 * allows it (full speed, no job time limit), and picks up where it stopped if it's interrupted.
 */
class BackfillClassifyWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val graph = (applicationContext as SiftApp).graph
        if (!graph.role.isHeldNow()) return Result.success()
        return if (inputData.getBoolean(KEY_RESCAN, false)) rescan(graph) else backfill(graph)
    }

    private suspend fun backfill(graph: AppGraph): Result {
        val messages = graph.db.messages()
        var remaining = messages.unclassifiedIncomingIds()
        var total = remaining.size
        var done = 0
        val skipped = HashSet<Long>()
        while (remaining.isNotEmpty()) {
            for (chunk in remaining.chunked(32)) {
                for (id in chunk) if (classify(graph, id)) done++ else skipped += id
                setProgress(workDataOf(KEY_DONE to done, KEY_TOTAL to total))
                if (isStopped) return Result.retry()
            }
            // Messages imported meanwhile: their own backfill request was dropped because this work was running.
            remaining = messages.unclassifiedIncomingIds().filterNot { it in skipped }
            total += remaining.size
        }
        return Result.success(workDataOf(KEY_DONE to done, KEY_TOTAL to total))
    }

    /**
     * Every incoming message without a user label, newest first. One classified again after the rescan was requested
     * counts as done, so a retry continues where the last run stopped.
     */
    private suspend fun rescan(graph: AppGraph): Result {
        val started = inputData.getLong(KEY_STARTED, 0L)
        val messages = graph.db.messages()
        val notifier = graph.notifier
        notifier.cancelScanDone() // the previous rescan's summary is out of date now
        try {
            setForeground(
                ForegroundInfo(
                    notifier.scanNotificationId(id), notifier.scanProgress(id, R.string.scan_progress_title, 0, 0),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
                ),
            )
        } catch (e: CancellationException) {
            throw e
        } catch (e: IllegalStateException) {
            // Android won't start a foreground service while the app is in the background (say, a retry after the
            // process died). Carry on as a regular job; the progress updates below post the same notice directly.
            Log.w(TAG, "rescan running without a foreground service", e)
        }
        try {
            // Folder moves are only summarized for a rescan that ran in one go: the snapshot doesn't survive a retry.
            val before = if (runAttemptCount == 0) folders(graph) else null
            var total = messages.rescanCount()
            var remaining = messages.rescanRemainingIds(started)
            var done = (total - remaining.size).coerceAtLeast(0)
            val skipped = HashSet<Long>()
            var noticeAt = 0L
            while (remaining.isNotEmpty()) {
                for (chunk in remaining.chunked(32)) {
                    for (messageId in chunk) {
                        if (!classify(graph, messageId)) skipped += messageId
                        done++
                    }
                    setProgress(workDataOf(KEY_DONE to done, KEY_TOTAL to total))
                    val now = SystemClock.elapsedRealtime()
                    if (now - noticeAt >= NOTICE_INTERVAL_MS) {
                        notifier.updateScanProgress(id, R.string.scan_progress_title, done, total)
                        noticeAt = now
                    }
                    if (isStopped) return Result.retry()
                }
                // Messages imported meanwhile: their own backfill request was dropped because this work was running.
                remaining = messages.rescanRemainingIds(started).filterNot { it in skipped }
                total += remaining.size
            }
            notifier.notifyScanDone(done - skipped.size, before?.let { FolderMoves.between(it, folders(graph)) })
            return Result.success(workDataOf(KEY_DONE to done, KEY_TOTAL to total))
        } finally {
            notifier.cancelScanProgress(id)
        }
    }

    /**
     * Classifies one stored message; false if it's gone, outgoing, or the engine failed on a message that already had
     * a verdict. A failed run only says "not spam", so it never replaces a real verdict.
     */
    private suspend fun classify(graph: AppGraph, messageId: Long): Boolean {
        val message = graph.db.messages().get(messageId) ?: return false
        val computed = graph.coordinator.compute(message) ?: return false
        if (computed.verdict.reason == Reason.UNAVAILABLE && computed.previous?.state == ClassState.DONE) return false
        graph.coordinator.apply(computed)
        return true
    }

    private suspend fun folders(graph: AppGraph) = graph.db.messages().incomingFolders().associate { it.id to it.folder }

    companion object {
        const val KEY_RESCAN = "rescan"
        const val KEY_STARTED = "started"
        const val KEY_DONE = "done"
        const val KEY_TOTAL = "total"
        private const val NOTICE_INTERVAL_MS = 500L
        private const val TAG = "SiftRescan"
    }
}

/**
 * After a retrain: re-checks every incoming message without a user label, newest first, so what the user just taught
 * applies to all of history. The threads of the messages just marked were re-checked first, right after the retrain, and
 * are skipped. The same rules as any re-score apply (model-only verdicts never move an Inbox message to Spam). Shows its
 * progress in a notification and runs as a foreground service when Android allows it; the next retrain's re-check
 * replaces one still running.
 */
class RecheckWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val graph = (applicationContext as SiftApp).graph
        if (!graph.role.isHeldNow()) return Result.success()
        val notifier = graph.notifier
        try {
            setForeground(
                ForegroundInfo(
                    notifier.scanNotificationId(id), notifier.scanProgress(id, R.string.recheck_progress_title, 0, 0),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
                ),
            )
        } catch (e: CancellationException) {
            throw e
        } catch (e: IllegalStateException) {
            // The app went to the background before the work started; the progress updates below post the notice.
            Log.w(TAG, "re-check running without a foreground service", e)
        }
        try {
            val skip = inputData.getLongArray(KEY_SKIP_THREADS)?.toList().orEmpty()
            val ids = graph.db.messages().unlabeledIncomingIds(skip)
            var done = 0
            var noticeAt = 0L
            for (chunk in ids.chunked(64)) {
                graph.rescore.afterRetrain(chunk)
                done += chunk.size
                val now = SystemClock.elapsedRealtime()
                if (now - noticeAt >= NOTICE_INTERVAL_MS) {
                    notifier.updateScanProgress(id, R.string.recheck_progress_title, done, ids.size)
                    noticeAt = now
                }
                if (isStopped) return Result.retry()
            }
            return Result.success()
        } finally {
            notifier.cancelScanProgress(id)
        }
    }

    companion object {
        const val KEY_SKIP_THREADS = "skipThreads"
        private const val NOTICE_INTERVAL_MS = 500L
        private const val TAG = "SiftRecheck"
    }
}

/**
 * Finishes every real-time classification that missed the receiver's deadline (or was interrupted by process death).
 * A late spam verdict moves the conversation and withdraws its notification; a late ham verdict notifies if the
 * message never did and is still unread.
 */
class ClassifyPendingWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val graph = (applicationContext as SiftApp).graph
        val ids = graph.db.classifications().pendingIds()
        for (id in ids) {
            val message = graph.db.messages().get(id) ?: continue
            val computed = graph.coordinator.compute(message) ?: continue
            val applied = graph.coordinator.apply(computed)
            if (applied.effectiveLabel == LabelCode.SPAM) {
                graph.notifier.cancelThread(message.threadId)
            } else if (!applied.notified && !message.read) {
                graph.notifier.notifyThread(message.threadId, alert = true)
            }
            graph.db.classifications().markNotified(id)
        }
        return Result.success()
    }
}
