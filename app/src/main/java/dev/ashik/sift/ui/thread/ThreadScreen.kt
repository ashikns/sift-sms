package dev.ashik.sift.ui.thread

import android.content.ClipData
import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.SnackbarResult
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.ClipEntry
import androidx.compose.ui.platform.LocalClipboard
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalResources
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.ashik.sift.R
import dev.ashik.sift.data.db.Folder
import dev.ashik.sift.data.db.LabelCode
import dev.ashik.sift.data.db.MsgStatus
import dev.ashik.sift.data.db.MsgType
import dev.ashik.sift.data.db.ThreadMessageRow
import dev.ashik.sift.data.db.UserLabelCode
import dev.ashik.sift.spam.CueCodec
import dev.ashik.sift.spamcore.Reason
import dev.ashik.sift.spamcore.SenderKind
import dev.ashik.sift.spamcore.UserLabel
import dev.ashik.sift.telephony.SegmentCounter
import dev.ashik.sift.ui.components.dayHeader
import dev.ashik.sift.ui.components.displayAddress
import dev.ashik.sift.ui.components.findLinks
import dev.ashik.sift.ui.components.messageText
import dev.ashik.sift.ui.components.rememberGraph
import dev.ashik.sift.ui.components.sameDay
import dev.ashik.sift.ui.components.timeOfDay
import dev.ashik.sift.ui.conversations.feedbackMessage
import dev.ashik.sift.ui.navigation.ThreadKey
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ThreadScreen(key: ThreadKey, onBack: () -> Unit) {
    val graph = rememberGraph()
    val context = LocalContext.current
    val vm = viewModel(key = "thread_${key.threadId}_${key.address}") { ThreadViewModel(graph, key) }
    val conversation by vm.conversation.collectAsStateWithLifecycle()
    val messages by vm.messages.collectAsStateWithLifecycle()
    val draft by vm.draft.collectAsStateWithLifecycle()
    val snackbar = remember { SnackbarHostState() }
    var why by remember { mutableStateOf<ThreadMessageRow?>(null) }
    var menu by remember { mutableStateOf(false) }
    var confirmDeleteThread by remember { mutableStateOf(false) }
    val lifecycleOwner = LocalLifecycleOwner.current
    val threadId = conversation?.conversation?.threadId ?: vm.currentThreadId

    DisposableEffect(lifecycleOwner, threadId) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_RESUME -> graph.visibleThread.threadId = threadId
                Lifecycle.Event.ON_PAUSE -> if (graph.visibleThread.threadId == threadId) graph.visibleThread.threadId = null
                else -> Unit
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        if (lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)) graph.visibleThread.threadId = threadId
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            if (graph.visibleThread.threadId == threadId) graph.visibleThread.threadId = null
        }
    }
    LaunchedEffect(messages) { messages?.let(vm::markReadIfNeeded) }
    val resources = LocalResources.current
    LaunchedEffect(vm) {
        vm.errors.collect { snackbar.showSnackbar(resources.getString(it)) }
    }
    LaunchedEffect(vm) {
        vm.feedback.collect { result ->
            if (result.marked == 0) return@collect
            val action = snackbar.showSnackbar(
                feedbackMessage(resources, result), actionLabel = resources.getString(R.string.action_undo),
                duration = SnackbarDuration.Long,
            )
            if (action == SnackbarResult.ActionPerformed) vm.undo(result.batchId)
        }
    }

    val conv = conversation?.conversation
    val address = conv?.address ?: key.address.orEmpty()
    val title = conversation?.displayName ?: displayAddress(address)
    val inSpam = conv?.folder == Folder.SPAM
    val senderKind = remember(address) { graph.normalizer.normalize(address).kind }
    val canReply = conv?.canReply ?: (senderKind == SenderKind.PHONE || senderKind == SenderKind.SHORT_CODE)

    Scaffold(
        topBar = {
            TopAppBar(
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.action_back)) }
                },
                title = {
                    Column {
                        Text(title, maxLines = 1)
                        if (conversation?.displayName != null) {
                            Text(displayAddress(address), style = MaterialTheme.typography.bodySmall)
                        }
                    }
                },
                actions = {
                    Box {
                        IconButton(onClick = { menu = true }) { Icon(Icons.Default.MoreVert, stringResource(R.string.action_more)) }
                        DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                            if (conv != null && conv.latestIncomingId != null) {
                                DropdownMenuItem(
                                    text = {
                                        Text(stringResource(if (inSpam) R.string.action_mark_conversation_not_spam else R.string.action_mark_conversation_spam))
                                    },
                                    onClick = {
                                        menu = false
                                        vm.markConversation(if (inSpam) UserLabel.HAM else UserLabel.SPAM)
                                    },
                                )
                            }
                            if (senderKind == SenderKind.PHONE) {
                                DropdownMenuItem(
                                    text = { Text(stringResource(R.string.action_call)) },
                                    onClick = {
                                        menu = false
                                        context.startActivity(Intent(Intent.ACTION_DIAL, Uri.parse("tel:$address")))
                                    },
                                )
                            }
                            if (conv != null) {
                                DropdownMenuItem(
                                    text = { Text(stringResource(R.string.action_delete_conversation)) },
                                    onClick = { menu = false; confirmDeleteThread = true },
                                )
                            }
                        }
                    }
                },
            )
        },
        bottomBar = {
            when {
                inSpam -> SpamBanner(onNotSpam = { vm.markConversation(UserLabel.HAM) })
                !canReply -> Surface(tonalElevation = 2.dp) {
                    Text(
                        stringResource(R.string.cant_reply), textAlign = TextAlign.Center,
                        modifier = Modifier.fillMaxWidth().navigationBarsPadding().padding(16.dp),
                    )
                }
                else -> Composer(text = draft, onText = { vm.draft.value = it }, onSend = vm::send)
            }
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { padding ->
        val rows = messages.orEmpty()
        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(padding).testTag("message_list"),
            reverseLayout = true,
            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            rows.forEachIndexed { index, m ->
                item(key = "m${m.id}") {
                    MessageBubble(
                        m = m,
                        onRetry = { vm.retry(m.id) },
                        onWhy = { why = m },
                        onMark = { label -> vm.markMessage(m.id, label) },
                        onDelete = { vm.deleteMessage(m.id) },
                    )
                }
                val older = rows.getOrNull(index + 1)
                if (older == null || !sameDay(older.date, m.date)) {
                    item(key = "d${m.id}") {
                        Text(
                            dayHeader(context, m.date), style = MaterialTheme.typography.labelMedium, textAlign = TextAlign.Center,
                            color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
                        )
                    }
                }
            }
        }
    }

    why?.let { m ->
        WhySheet(
            m = m,
            onDismiss = { why = null },
            onMark = { label -> why = null; vm.markMessage(m.id, label) },
        )
    }

    if (confirmDeleteThread) {
        AlertDialog(
            onDismissRequest = { confirmDeleteThread = false },
            title = { Text(stringResource(R.string.action_delete_conversation) + "?") },
            text = { Text(stringResource(R.string.delete_conversations_text)) },
            confirmButton = {
                TextButton(onClick = { confirmDeleteThread = false; vm.deleteConversation(onBack) }) { Text(stringResource(R.string.action_delete)) }
            },
            dismissButton = { TextButton(onClick = { confirmDeleteThread = false }) { Text(stringResource(R.string.action_cancel)) } },
        )
    }
}

@Composable
private fun SpamBanner(onNotSpam: () -> Unit) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
        modifier = Modifier.fillMaxWidth().navigationBarsPadding().padding(12.dp),
    ) {
        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Default.Warning, contentDescription = null, tint = MaterialTheme.colorScheme.onErrorContainer)
            Text(
                stringResource(R.string.spam_banner), color = MaterialTheme.colorScheme.onErrorContainer,
                modifier = Modifier.weight(1f).padding(horizontal = 12.dp),
            )
            Button(onClick = onNotSpam, modifier = Modifier.testTag("banner_not_spam")) { Text(stringResource(R.string.action_not_spam)) }
        }
    }
}

@Composable
fun Composer(text: String, onText: (String) -> Unit, onSend: () -> Unit) {
    val count = remember(text) { if (text.isEmpty()) null else runCatching { SegmentCounter.count(text) }.getOrNull() }
    Surface(tonalElevation = 3.dp) {
        Row(
            Modifier.fillMaxWidth().navigationBarsPadding().imePadding().padding(horizontal = 8.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            OutlinedTextField(
                value = text,
                onValueChange = onText,
                placeholder = { Text(stringResource(R.string.thread_composer_hint)) },
                maxLines = 6,
                supportingText = count?.let { c ->
                    {
                        Text(
                            if (c.segments > 1) stringResource(R.string.segments_counter, c.remainingInSegment, c.segments)
                            else stringResource(R.string.segments_counter_single, c.remainingInSegment),
                        )
                    }
                },
                modifier = Modifier.weight(1f).testTag("composer"),
            )
            IconButton(onClick = onSend, enabled = text.isNotBlank(), modifier = Modifier.testTag("send")) {
                Icon(Icons.AutoMirrored.Filled.Send, stringResource(R.string.action_send))
            }
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun MessageBubble(
    m: ThreadMessageRow,
    onRetry: () -> Unit,
    onWhy: () -> Unit,
    onMark: (UserLabel) -> Unit,
    onDelete: () -> Unit,
) {
    val incoming = m.type == MsgType.INBOX
    val flagged = incoming && m.effectiveLabel != LabelCode.HAM
    val failed = m.type == MsgType.FAILED
    var menu by remember { mutableStateOf(false) }
    var confirmDelete by remember { mutableStateOf(false) }
    val clipboard = LocalClipboard.current
    val scope = rememberCoroutineScope()
    Column(
        Modifier.fillMaxWidth().testTag("message_${m.id}"),
        horizontalAlignment = if (incoming) Alignment.Start else Alignment.End,
    ) {
        if (flagged) {
            AssistChip(
                onClick = onWhy,
                label = { Text(stringResource(if (m.effectiveLabel == LabelCode.SPAM) R.string.chip_spam else R.string.chip_possible_spam)) },
                leadingIcon = { Icon(Icons.Default.Warning, contentDescription = null, modifier = Modifier.size(18.dp)) },
                colors = AssistChipDefaults.assistChipColors(
                    containerColor = if (m.effectiveLabel == LabelCode.SPAM) MaterialTheme.colorScheme.errorContainer else MaterialTheme.colorScheme.tertiaryContainer,
                ),
                modifier = Modifier.testTag("chip_${m.id}"),
            )
        }
        Box {
            Surface(
                shape = RoundedCornerShape(18.dp),
                color = when {
                    failed -> MaterialTheme.colorScheme.errorContainer
                    incoming -> MaterialTheme.colorScheme.surfaceContainerHighest
                    else -> MaterialTheme.colorScheme.primaryContainer
                },
                modifier = Modifier
                    .widthIn(max = 320.dp)
                    .combinedClickable(onClick = { if (failed) onRetry() }, onLongClick = { menu = true }),
            ) {
                Text(
                    messageText(m.body, linksEnabled = !flagged, linkColor = MaterialTheme.colorScheme.primary),
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 9.dp),
                )
            }
            DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                if (incoming) {
                    val isSpam = m.effectiveLabel == LabelCode.SPAM
                    DropdownMenuItem(
                        text = { Text(stringResource(if (isSpam) R.string.action_not_spam else R.string.action_mark_conversation_spam)) },
                        onClick = { menu = false; onMark(if (isSpam) UserLabel.HAM else UserLabel.SPAM) },
                        modifier = Modifier.testTag("menu_mark"),
                    )
                    DropdownMenuItem(text = { Text(stringResource(R.string.action_details)) }, onClick = { menu = false; onWhy() })
                }
                DropdownMenuItem(
                    text = { Text(stringResource(R.string.action_copy)) },
                    onClick = {
                        menu = false
                        scope.launch { clipboard.setClipEntry(ClipEntry(ClipData.newPlainText("message", m.body))) }
                    },
                )
                DropdownMenuItem(text = { Text(stringResource(R.string.action_delete)) }, onClick = { menu = false; confirmDelete = true })
            }
        }
        if (flagged && findLinks(m.body).isNotEmpty()) {
            Text(
                stringResource(R.string.links_disabled), style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        val status = when {
            m.type == MsgType.OUTBOX || m.type == MsgType.QUEUED -> stringResource(R.string.status_sending)
            failed -> stringResource(R.string.status_failed)
            m.type == MsgType.SENT && m.status == MsgStatus.COMPLETE -> stringResource(R.string.status_delivered)
            m.type == MsgType.SENT && m.status == MsgStatus.FAILED -> stringResource(R.string.status_delivery_failed)
            else -> null
        }
        Text(
            listOfNotNull(timeOfDay(LocalContext.current, m.date), status).joinToString(" · "),
            style = MaterialTheme.typography.labelSmall,
            color = if (failed) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(horizontal = 6.dp),
        )
    }
    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text(stringResource(R.string.delete_message_title)) },
            confirmButton = { TextButton(onClick = { confirmDelete = false; onDelete() }) { Text(stringResource(R.string.action_delete)) } },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text(stringResource(R.string.action_cancel)) } },
        )
    }
}

fun reasonText(reason: String?): Int = when (runCatching { Reason.valueOf(reason ?: "") }.getOrNull()) {
    Reason.SIMILAR_TO_SPAM -> R.string.reason_SIMILAR_TO_SPAM
    Reason.SIMILAR_TO_HAM -> R.string.reason_SIMILAR_TO_HAM
    Reason.MODEL -> R.string.reason_MODEL
    Reason.MODEL_FALLBACK -> R.string.reason_MODEL_FALLBACK
    Reason.OTP_GUARDRAIL -> R.string.reason_OTP_GUARDRAIL
    Reason.UNAVAILABLE -> R.string.reason_UNAVAILABLE
    null -> R.string.why_pending
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun WhySheet(m: ThreadMessageRow, onDismiss: () -> Unit, onMark: (UserLabel) -> Unit) {
    ModalBottomSheet(onDismissRequest = onDismiss) {
        Column(Modifier.fillMaxWidth().padding(horizontal = 24.dp).padding(bottom = 32.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            val title = when (m.effectiveLabel) {
                LabelCode.SPAM -> R.string.why_title_spam
                LabelCode.SUSPICIOUS -> R.string.why_title_suspicious
                else -> R.string.why_title_ham
            }
            Text(stringResource(title), style = MaterialTheme.typography.titleLarge)
            val userLabel = m.userLabel?.let(UserLabelCode::toUserLabel)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Info, contentDescription = null)
                Text(
                    stringResource(
                        when (userLabel) {
                            UserLabel.SPAM -> R.string.reason_user_spam
                            UserLabel.HAM -> R.string.reason_user_ham
                            null -> reasonText(m.reason)
                        },
                    ),
                    modifier = Modifier.padding(start = 12.dp),
                )
            }
            m.modelScore?.let { score ->
                Text(stringResource(R.string.why_score, (score * 100).roundToInt()), style = MaterialTheme.typography.bodyLarge)
            }
            val cues = CueCodec.decode(m.cues)
            if (cues.isNotEmpty()) {
                HorizontalDivider()
                Text(stringResource(R.string.why_cues), style = MaterialTheme.typography.titleSmall)
                cues.take(5).forEach { Text("• ${it.text}", style = MaterialTheme.typography.bodyMedium) }
            }
            m.modelVersion?.let { v ->
                Text(
                    if (v > 0) stringResource(R.string.why_model_version, v) else stringResource(R.string.why_base_model),
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                if (m.effectiveLabel == LabelCode.SPAM) {
                    Button(onClick = { onMark(UserLabel.HAM) }) { Text(stringResource(R.string.action_not_spam)) }
                } else {
                    OutlinedButton(onClick = { onMark(UserLabel.SPAM) }) { Text(stringResource(R.string.action_mark_conversation_spam)) }
                    if (m.effectiveLabel == LabelCode.SUSPICIOUS) {
                        Button(onClick = { onMark(UserLabel.HAM) }) { Text(stringResource(R.string.action_not_spam)) }
                    }
                }
            }
        }
    }
}
