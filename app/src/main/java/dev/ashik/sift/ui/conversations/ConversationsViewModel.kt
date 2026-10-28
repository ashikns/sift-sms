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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.conflate
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class ScanProgress(val done: Int, val total: Int)

@OptIn(ExperimentalCoroutinesApi::class)
class ConversationsViewModel(private val graph: AppGraph, context: Context) : ViewModel() {
    val inbox: StateFlow<List<ConversationRow>?> = graph.db.conversations().observeFolder(Folder.INBOX)
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)
    val spam: StateFlow<List<ConversationRow>?> = graph.db.conversations().observeFolder(Folder.SPAM)
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)

    private val query = MutableStateFlow("")

    /** Every message's search key: built only while there's a search, and again when messages or their tabs change. */
    private val index: StateFlow<SearchIndex?> = graph.db.messages().observeSearchable()
        .conflate()
        .map(::SearchIndex)
        .flowOn(Dispatchers.Default)
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)

    /** Each tab's conversations matching the search, or null when there's no search (or while it first loads). */
    val search: StateFlow<SearchResults?> = query.map(SearchText::key).distinctUntilChanged()
        .flatMapLatest { q ->
            if (q.isEmpty()) {
                flowOf(null)
            } else {
                combine(inbox.filterNotNull(), spam.filterNotNull(), index.filterNotNull()) { i, s, idx -> idx.search(i, s, q) }
                    .flowOn(Dispatchers.Default)
            }
        }
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

    /** What the search box holds ("" when it's closed). */
    fun setQuery(text: String) {
        query.value = text
    }

    /** Deletes these conversations' messages in [folder]: what that tab shows of them. */
    fun delete(threadIds: Set<Long>, folder: Int) = viewModelScope.launch {
        threadIds.forEach { graph.notifier.cancel(it, folder) }
        runCatching { graph.repository.deleteFolder(threadIds, folder) }
    }
}
