package dev.ashik.sift.spam.work

import android.content.Context
import androidx.work.BackoffPolicy
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.OutOfQuotaPolicy
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import dev.ashik.sift.SiftApp
import dev.ashik.sift.data.db.LabelCode
import java.util.concurrent.TimeUnit

/** Enqueues the app's background work (all unique, all chunked so each run stays well under 10 minutes). */
class WorkScheduler(private val context: Context) {
    private val wm get() = WorkManager.getInstance(context)

    fun enqueueSync(mode: String, markInitialDone: Boolean = false) {
        val request = OneTimeWorkRequestBuilder<SyncWorker>()
            .setInputData(workDataOf(SyncWorker.KEY_MODE to mode, SyncWorker.KEY_INITIAL to markInitialDone))
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
            .build()
        wm.enqueueUniqueWork("sync-$mode", ExistingWorkPolicy.KEEP, request)
    }

    fun enqueueBackfill(rescan: Boolean = false) {
        val request = OneTimeWorkRequestBuilder<BackfillClassifyWorker>()
            .setInputData(workDataOf(BackfillClassifyWorker.KEY_RESCAN to rescan, BackfillClassifyWorker.KEY_STARTED to System.currentTimeMillis()))
            .build()
        wm.enqueueUniqueWork(BACKFILL, if (rescan) ExistingWorkPolicy.REPLACE else ExistingWorkPolicy.KEEP, request)
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
        when (inputData.getString(KEY_MODE)) {
            MODE_FULL -> graph.sync.fullReconcile()
            else -> graph.sync.incremental()
        }
        if (inputData.getBoolean(KEY_INITIAL, false)) {
            graph.settings.setInitialImportDone(true)
            graph.contacts.refreshAll()
            graph.scheduler.enqueueBackfill()
        }
        return Result.success()
    }

    companion object {
        const val KEY_MODE = "mode"
        const val KEY_INITIAL = "initial"
        const val MODE_FULL = "full"
        const val MODE_INCREMENTAL = "incremental"
    }
}

/** Classifies history (last 90 days, at most 5,000 messages, newest first), or re-scans it on request. */
class BackfillClassifyWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val graph = (applicationContext as SiftApp).graph
        if (!graph.role.isHeldNow()) return Result.success()
        val since = System.currentTimeMillis() - WINDOW_MS
        val rescan = inputData.getBoolean(KEY_RESCAN, false)
        val ids = if (rescan) {
            graph.db.messages().rescanCandidateIds(since, LIMIT)
        } else {
            graph.db.messages().unclassifiedIncomingIds(since, LIMIT)
        }
        val total = ids.size
        var done = 0
        for (chunk in ids.chunked(32)) {
            for (id in chunk) {
                val message = graph.db.messages().get(id) ?: continue
                val computed = graph.coordinator.compute(message) ?: continue
                graph.coordinator.apply(computed)
                done++
            }
            setProgress(workDataOf(KEY_DONE to done, KEY_TOTAL to total))
            if (isStopped) return Result.retry()
        }
        return Result.success(workDataOf(KEY_DONE to done, KEY_TOTAL to total))
    }

    companion object {
        const val KEY_RESCAN = "rescan"
        const val KEY_STARTED = "started"
        const val KEY_DONE = "done"
        const val KEY_TOTAL = "total"
        const val WINDOW_MS = 90L * 24 * 60 * 60 * 1000
        const val LIMIT = 5000
    }
}

/**
 * Finishes real-time classifications that missed the receiver's deadline (or were interrupted by process death).
 * A late spam verdict moves the conversation and withdraws its notification; a late ham verdict notifies if the
 * message never did and is still fresh.
 */
class ClassifyPendingWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val graph = (applicationContext as SiftApp).graph
        val ids = graph.db.classifications().pendingIds(200)
        val now = System.currentTimeMillis()
        for (id in ids) {
            val message = graph.db.messages().get(id) ?: continue
            val computed = graph.coordinator.compute(message) ?: continue
            val applied = graph.coordinator.apply(computed)
            if (applied.effectiveLabel == LabelCode.SPAM) {
                graph.notifier.cancelThread(message.threadId)
            } else if (!applied.notified && !message.read && now - message.date < FRESH_MS) {
                graph.notifier.notifyThread(message.threadId, alert = true)
            }
            graph.db.classifications().markNotified(id)
        }
        return Result.success()
    }

    private companion object {
        const val FRESH_MS = 60L * 60 * 1000
    }
}
