package dev.ashik.sift.ui.thread

import android.content.ClipData
import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.contextmenu.data.TextContextMenuKeys
import androidx.compose.foundation.text.contextmenu.modifier.filterTextContextMenuComponents
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.text.selection.rememberSelectionState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
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
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.pointerInput
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
import dev.ashik.sift.notifications.ThreadView
import dev.ashik.sift.spam.CueCodec
import dev.ashik.sift.spamcore.Reason
import dev.ashik.sift.spamcore.SenderKind
import dev.ashik.sift.spamcore.UserLabel
import dev.ashik.sift.ui.components.dayHeader
import dev.ashik.sift.ui.components.displayAddress
import dev.ashik.sift.ui.components.rememberGraph
import dev.ashik.sift.ui.components.sameDay
import dev.ashik.sift.ui.components.timeOfDay
import dev.ashik.sift.ui.navigation.ThreadKey
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ThreadScreen(key: ThreadKey, onBack: () -> Unit) {
    val graph = rememberGraph()
    val context = LocalContext.current
    val vm = viewModel(key = "thread_${key.threadId}_${key.folder}_${key.address}") { ThreadViewModel(graph, key) }
    val conversation by vm.conversation.collectAsStateWithLifecycle()
    val messages by vm.messages.collectAsStateWithLifecycle()
    val snackbar = remember { SnackbarHostState() }
    var why by remember { mutableStateOf<ThreadMessageRow?>(null) }
    var menu by remember { mutableStateOf(false) }
    var confirmDeleteThread by remember { mutableStateOf(false) }
    val lifecycleOwner = LocalLifecycleOwner.current
    val threadId = conversation?.conversation?.threadId ?: vm.currentThreadId
    val view = threadId?.let { ThreadView(it, vm.folder) }

    DisposableEffect(lifecycleOwner, view) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_RESUME -> graph.visibleThread.view = view
                Lifecycle.Event.ON_PAUSE -> if (graph.visibleThread.view == view) graph.visibleThread.view = null
                else -> Unit
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        if (lifecycleOwner.lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)) graph.visibleThread.view = view
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            if (graph.visibleThread.view == view) graph.visibleThread.view = null
        }
    }
    LaunchedEffect(messages) { messages?.let(vm::markReadIfNeeded) }
    val resources = LocalResources.current
    LaunchedEffect(vm) {
        // After a mark, "Re-checking now…" stays up until the retrained model has re-checked this thread; a newer mark
        // takes over the snackbar.
        vm.rechecks.collectLatest { rechecked ->
            coroutineScope {
                val shown = launch { snackbar.showSnackbar(resources.getString(R.string.snackbar_rechecking), duration = SnackbarDuration.Indefinite) }
                rechecked.join()
                shown.cancel() // cancelling showSnackbar removes the snackbar
            }
        }
    }

    val conv = conversation?.conversation
    val address = conv?.address ?: key.address.orEmpty()
    val title = conversation?.displayName ?: displayAddress(address)
    val senderKind = remember(address) { graph.normalizer.normalize(address).kind }

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
                            if (senderKind == SenderKind.PHONE) {
                                DropdownMenuItem(
                                    text = { Text(stringResource(R.string.action_call)) },
                                    onClick = {
                                        menu = false
                                        context.startActivity(Intent(Intent.ACTION_DIAL, Uri.parse("tel:$address")))
                                    },
                                )
                            }
                            if (messages?.isNotEmpty() == true) {
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
        snackbarHost = { SnackbarHost(snackbar) },
    ) { padding ->
        val rows = messages.orEmpty()
        if (messages?.isEmpty() == true) {
            val empty = when {
                // Opened from another app's sms: link to someone without a conversation; Sift can't start one.
                conv == null -> R.string.thread_empty
                // Everything this tab showed moved to the other one (say, the only spam message was marked not spam).
                vm.folder == Folder.SPAM -> R.string.thread_empty_spam
                else -> R.string.thread_empty_inbox
            }
            Box(Modifier.fillMaxSize().padding(padding).padding(32.dp), contentAlignment = Alignment.Center) {
                Text(
                    stringResource(empty), style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center,
                )
            }
        }
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

/**
 * One message. Its text is selectable (long-press a word, drag the handles for part of it, or Select all) and never has
 * links; a tap opens the message's menu (spam marks, details, copying the whole text, delete).
 */
@Composable
fun MessageBubble(
    m: ThreadMessageRow,
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
    val selection = rememberSelectionState()
    // A tap that dismisses a text selection shouldn't also open the menu. The selection is already gone when the click
    // fires, so note whether there was one when the finger went down (the initial pass reaches this before the text).
    var tapClearsSelection by remember { mutableStateOf(false) }
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
                    .pointerInput(selection) {
                        awaitEachGesture {
                            awaitFirstDown(requireUnconsumed = false, pass = PointerEventPass.Initial)
                            tapClearsSelection = selection.selectedTexts.isNotEmpty()
                        }
                    }
                    .clickable(onClickLabel = stringResource(R.string.action_message_options)) { if (!tapClearsSelection) menu = true },
            ) {
                // The selection toolbar offers only Copy and Select all: no smart "Open" or "Call" for a link or number in
                // the text, and no other apps' text actions.
                SelectionContainer(selection, Modifier.filterTextContextMenuComponents { it.key in COPY_ONLY }) {
                    Text(m.body, modifier = Modifier.padding(horizontal = 14.dp, vertical = 9.dp).testTag("body_${m.id}"))
                }
            }
            DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                if (incoming) {
                    val isSpam = m.effectiveLabel == LabelCode.SPAM
                    DropdownMenuItem(
                        text = { Text(stringResource(if (isSpam) R.string.action_not_spam else R.string.action_mark_spam)) },
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
        val status = when {
            // Left unsent by another app (Sift never sends), or failed.
            m.type == MsgType.OUTBOX || m.type == MsgType.QUEUED || failed -> stringResource(R.string.status_not_sent)
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

private val COPY_ONLY = setOf(TextContextMenuKeys.CopyKey, TextContextMenuKeys.SelectAllKey)

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
                    OutlinedButton(onClick = { onMark(UserLabel.SPAM) }) { Text(stringResource(R.string.action_mark_spam)) }
                    if (m.effectiveLabel == LabelCode.SUSPICIOUS) {
                        Button(onClick = { onMark(UserLabel.HAM) }) { Text(stringResource(R.string.action_not_spam)) }
                    }
                }
            }
        }
    }
}
