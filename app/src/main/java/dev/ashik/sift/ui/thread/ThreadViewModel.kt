package dev.ashik.sift.ui.thread

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.ashik.sift.R
import dev.ashik.sift.data.db.ConversationRow
import dev.ashik.sift.data.db.MsgType
import dev.ashik.sift.data.db.ThreadMessageRow
import dev.ashik.sift.di.AppGraph
import dev.ashik.sift.spam.FeedbackService
import dev.ashik.sift.spamcore.UserLabel
import dev.ashik.sift.telephony.RoleLostException
import dev.ashik.sift.ui.navigation.ThreadKey
import kotlinx.coroutines.ExperimentalCoroutinesApi
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
    private val draftAddress = key.address
    val draft = MutableStateFlow(key.draft.orEmpty())

    val conversation: StateFlow<ConversationRow?> = threadId
        .flatMapLatest { id -> if (id == null) flowOf(null) else graph.db.conversations().observe(id) }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)

    val messages: StateFlow<List<ThreadMessageRow>?> = threadId
        .flatMapLatest { id -> if (id == null) flowOf(emptyList()) else graph.db.messages().observeThread(id, 500) }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)

    private val _errors = MutableSharedFlow<Int>(extraBufferCapacity = 4)
    val errors: SharedFlow<Int> = _errors
    private val _feedback = MutableSharedFlow<FeedbackService.Result>(extraBufferCapacity = 4)
    val feedback: SharedFlow<FeedbackService.Result> = _feedback

    init {
        if (threadId.value == null && draftAddress != null) {
            viewModelScope.launch {
                val senderKey = graph.normalizer.normalize(draftAddress).key
                graph.db.messages().threadForSender(senderKey)?.let { threadId.value = it }
            }
        }
    }

    val currentThreadId: Long? get() = threadId.value

    /** The address to send to: the conversation's, or the one this new conversation was started with. */
    val address: String? get() = conversation.value?.conversation?.address ?: draftAddress

    fun send() {
        val text = draft.value.trim()
        val to = address
        if (text.isEmpty() || to.isNullOrBlank()) return
        draft.value = ""
        viewModelScope.launch {
            try {
                val id = graph.sender.send(to, text)
                if (threadId.value == null) graph.db.messages().get(id)?.threadId?.let { threadId.value = it }
            } catch (_: RoleLostException) {
                graph.role.refresh()
                draft.value = text
                _errors.emit(R.string.error_role_lost)
            } catch (_: Exception) {
                draft.value = text
                _errors.emit(R.string.error_send)
            }
        }
    }

    fun retry(messageId: Long) = viewModelScope.launch { runCatching { graph.sender.retry(messageId) } }

    fun markMessage(messageId: Long, label: UserLabel) = viewModelScope.launch {
        _feedback.emit(graph.feedback.markMessages(listOf(messageId), label))
    }

    fun markConversation(label: UserLabel) {
        val id = threadId.value ?: return
        viewModelScope.launch { _feedback.emit(graph.feedback.markThreads(listOf(id), label)) }
    }

    fun undo(batchId: Long) = viewModelScope.launch { graph.feedback.undo(batchId) }

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
