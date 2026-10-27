package dev.ashik.sift.ui.thread

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.ashik.sift.data.db.ConversationRow
import dev.ashik.sift.data.db.MsgType
import dev.ashik.sift.data.db.ThreadMessageRow
import dev.ashik.sift.di.AppGraph
import dev.ashik.sift.spamcore.UserLabel
import dev.ashik.sift.ui.navigation.ThreadKey
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

@OptIn(ExperimentalCoroutinesApi::class)
class ThreadViewModel(private val graph: AppGraph, key: ThreadKey) : ViewModel() {
    private val threadId = MutableStateFlow(key.threadId.takeIf { it > 0 })

    val conversation: StateFlow<ConversationRow?> = threadId
        .flatMapLatest { id -> if (id == null) flowOf(null) else graph.db.conversations().observe(id) }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)

    val messages: StateFlow<List<ThreadMessageRow>?> = threadId
        .flatMapLatest { id -> if (id == null) flowOf(emptyList()) else graph.db.messages().observeThread(id) }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)

    /** For each mark, the job that re-checks this thread with the retrained model. */
    private val _rechecks = MutableSharedFlow<Job>(extraBufferCapacity = 4)
    val rechecks: SharedFlow<Job> = _rechecks

    init {
        val address = key.address
        if (threadId.value == null && address != null) {
            viewModelScope.launch {
                val senderKey = graph.normalizer.normalize(address).key
                graph.db.messages().threadForSender(senderKey)?.let { threadId.value = it }
            }
        }
    }

    val currentThreadId: Long? get() = threadId.value

    fun markMessage(messageId: Long, label: UserLabel) = viewModelScope.launch {
        graph.feedback.markMessages(listOf(messageId), label)?.let { _rechecks.emit(it) }
    }

    fun deleteMessage(messageId: Long) = viewModelScope.launch { runCatching { graph.repository.deleteMessages(listOf(messageId)) } }

    fun deleteConversation(onDone: () -> Unit) {
        val id = threadId.value ?: return onDone()
        viewModelScope.launch {
            graph.notifier.cancelThread(id)
            runCatching { graph.repository.deleteThreads(listOf(id)) }
            onDone()
        }
    }

    fun markReadIfNeeded(rows: List<ThreadMessageRow>) {
        val id = threadId.value ?: return
        if (rows.any { it.type == MsgType.INBOX && !it.read }) {
            viewModelScope.launch {
                graph.repository.markThreadRead(id)
                graph.notifier.cancelThread(id)
            }
        }
    }
}
