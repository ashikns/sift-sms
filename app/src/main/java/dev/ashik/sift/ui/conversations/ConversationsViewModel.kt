package dev.ashik.sift.ui.conversations

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.work.WorkInfo
import androidx.work.WorkManager
import dev.ashik.sift.data.db.ConversationRow
import dev.ashik.sift.data.db.Folder
import dev.ashik.sift.di.AppGraph
import dev.ashik.sift.spam.work.BackfillClassifyWorker
import dev.ashik.sift.spam.work.WorkScheduler
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class ScanProgress(val done: Int, val total: Int)

class ConversationsViewModel(private val graph: AppGraph, context: Context) : ViewModel() {
    val inbox: StateFlow<List<ConversationRow>?> = graph.db.conversations().observeFolder(Folder.INBOX)
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)
    val spam: StateFlow<List<ConversationRow>?> = graph.db.conversations().observeFolder(Folder.SPAM)
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)

    val scan: StateFlow<ScanProgress?> = WorkManager.getInstance(context)
        .getWorkInfosForUniqueWorkFlow(WorkScheduler.BACKFILL)
        .map { infos ->
            infos.firstOrNull { it.state == WorkInfo.State.RUNNING }?.progress?.let { p ->
                val total = p.getInt(BackfillClassifyWorker.KEY_TOTAL, 0)
                if (total > 0) ScanProgress(p.getInt(BackfillClassifyWorker.KEY_DONE, 0), total) else null
            }
        }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)

    fun delete(threadIds: Set<Long>) = viewModelScope.launch {
        threadIds.forEach(graph.notifier::cancelThread)
        runCatching { graph.repository.deleteThreads(threadIds.toList()) }
    }
}
