package dev.ashik.sift.ui.conversations

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Done
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Badge
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.ListItem
import androidx.compose.material3.ListItemDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.PrimaryTabRow
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SuggestionChip
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.ashik.sift.R
import dev.ashik.sift.data.db.ConversationRow
import dev.ashik.sift.data.db.LabelCode
import dev.ashik.sift.data.db.MsgType
import dev.ashik.sift.ui.components.Avatar
import dev.ashik.sift.ui.components.displayAddress
import dev.ashik.sift.ui.components.rememberGraph
import dev.ashik.sift.ui.components.shortTime

@Composable
fun ConversationsScreen(onOpenThread: (Long) -> Unit, onOpenSettings: () -> Unit) {
    val graph = rememberGraph()
    val context = LocalContext.current
    val vm = viewModel { ConversationsViewModel(graph, context.applicationContext) }
    val inbox by vm.inbox.collectAsStateWithLifecycle()
    val spam by vm.spam.collectAsStateWithLifecycle()
    val scan by vm.scan.collectAsStateWithLifecycle()
    ConversationsContent(
        inbox = inbox, spam = spam, scan = scan,
        onOpenThread = onOpenThread, onOpenSettings = onOpenSettings,
        onMarkRead = { vm.markRead(it) }, onDelete = { vm.delete(it) },
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConversationsContent(
    inbox: List<ConversationRow>?,
    spam: List<ConversationRow>?,
    scan: ScanProgress?,
    onOpenThread: (Long) -> Unit,
    onOpenSettings: () -> Unit,
    onMarkRead: (Set<Long>) -> Unit,
    onDelete: (Set<Long>) -> Unit,
) {
    var tab by rememberSaveable { mutableIntStateOf(0) }
    var selection by remember { mutableStateOf(emptySet<Long>()) }
    var confirmDelete by remember { mutableStateOf(false) }
    BackHandler(enabled = selection.isNotEmpty()) { selection = emptySet() }
    val rows = if (tab == 0) inbox else spam
    val inSpam = tab == 1
    // One list state per tab. A LazyColumn with stable keys keeps its first visible item in place when a new
    // conversation is inserted above it, so a new message would land just off-screen; follow it if we were at the top.
    val inboxState = rememberLazyListState()
    val spamState = rememberLazyListState()
    val listState = if (inSpam) spamState else inboxState
    val topKey = rows?.firstOrNull()?.conversation?.threadId
    LaunchedEffect(topKey, inSpam) {
        if (topKey != null && listState.firstVisibleItemIndex <= 1) listState.scrollToItem(0)
    }

    Scaffold(
        topBar = {
            if (selection.isEmpty()) {
                TopAppBar(
                    title = { Text(stringResource(R.string.app_name)) },
                    actions = {
                        IconButton(onClick = onOpenSettings) { Icon(Icons.Default.Settings, stringResource(R.string.settings)) }
                    },
                )
            } else {
                TopAppBar(
                    navigationIcon = {
                        IconButton(onClick = { selection = emptySet() }) {
                            Icon(Icons.Default.Close, stringResource(R.string.action_close_selection))
                        }
                    },
                    title = { Text(stringResource(R.string.selected_count, selection.size)) },
                    actions = {
                        if (!inSpam) {
                            IconButton(onClick = { onMarkRead(selection); selection = emptySet() }) {
                                Icon(Icons.Default.Done, stringResource(R.string.action_mark_read))
                            }
                        }
                        IconButton(onClick = { confirmDelete = true }) { Icon(Icons.Default.Delete, stringResource(R.string.action_delete)) }
                    },
                )
            }
        },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            PrimaryTabRow(selectedTabIndex = tab) {
                Tab(
                    selected = tab == 0, onClick = { tab = 0; selection = emptySet() },
                    text = { TabLabel(stringResource(R.string.tab_inbox), inbox?.count { it.conversation.unreadCount > 0 } ?: 0) },
                    modifier = Modifier.testTag("tab_inbox"),
                )
                Tab(
                    selected = tab == 1, onClick = { tab = 1; selection = emptySet() },
                    text = { TabLabel(stringResource(R.string.tab_spam), spam?.size ?: 0) },
                    modifier = Modifier.testTag("tab_spam"),
                )
            }
            if (scan != null) {
                Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp)) {
                    Text(stringResource(R.string.scanning, scan.done, scan.total), style = MaterialTheme.typography.bodySmall)
                    LinearProgressIndicator(progress = { scan.done.toFloat() / scan.total.coerceAtLeast(1) }, modifier = Modifier.fillMaxWidth())
                }
            }
            when {
                rows == null -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { CircularProgressIndicator() }
                rows.isEmpty() -> Box(Modifier.fillMaxSize().padding(32.dp), contentAlignment = Alignment.Center) {
                    Text(
                        stringResource(if (inSpam) R.string.empty_spam else R.string.empty_inbox),
                        style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                else -> LazyColumn(Modifier.fillMaxSize().testTag("conversation_list"), state = listState) {
                    items(rows, key = { it.conversation.threadId }) { row ->
                        val id = row.conversation.threadId
                        ConversationItem(
                            row = row,
                            selected = id in selection,
                            onClick = {
                                if (selection.isNotEmpty()) selection = selection.toggle(id) else onOpenThread(id)
                            },
                            onLongClick = { selection = selection.toggle(id) },
                        )
                    }
                }
            }
        }
    }

    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text(pluralStringResource(R.plurals.delete_conversations_title, selection.size, selection.size)) },
            text = { Text(stringResource(R.string.delete_conversations_text)) },
            confirmButton = {
                TextButton(onClick = { onDelete(selection); selection = emptySet(); confirmDelete = false }) {
                    Text(stringResource(R.string.action_delete))
                }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text(stringResource(R.string.action_cancel)) } },
        )
    }
}

private fun Set<Long>.toggle(id: Long) = if (id in this) this - id else this + id

@Composable
private fun TabLabel(text: String, count: Int) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(text)
        if (count > 0) Badge { Text(count.toString()) }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun ConversationItem(row: ConversationRow, selected: Boolean, onClick: () -> Unit, onLongClick: () -> Unit) {
    val c = row.conversation
    val name = row.displayName ?: displayAddress(c.address)
    val unread = c.unreadCount > 0
    val snippet = if (c.latestType != MsgType.INBOX) stringResource(R.string.you_prefix, c.snippet) else c.snippet
    ListItem(
        modifier = Modifier
            .combinedClickable(onClick = onClick, onLongClick = onLongClick)
            .testTag("conversation_${c.threadId}"),
        colors = ListItemDefaults.colors(
            containerColor = if (selected) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.surface,
        ),
        leadingContent = {
            if (selected) {
                Box(
                    Modifier.size(44.dp).background(MaterialTheme.colorScheme.primary, CircleShape),
                    contentAlignment = Alignment.Center,
                ) { Icon(Icons.Default.Check, contentDescription = null, tint = MaterialTheme.colorScheme.onPrimary) }
            } else {
                Avatar(name = name, key = c.senderKey, photoUri = row.photoUri)
            }
        },
        headlineContent = {
            Text(name, maxLines = 1, overflow = TextOverflow.Ellipsis, fontWeight = if (unread) FontWeight.Bold else FontWeight.Normal)
        },
        supportingContent = {
            Column {
                Text(
                    snippet, maxLines = 2, overflow = TextOverflow.Ellipsis,
                    fontWeight = if (unread) FontWeight.SemiBold else FontWeight.Normal,
                )
                if (c.latestLabel == LabelCode.SUSPICIOUS && c.folder == 0) {
                    SuggestionChip(onClick = onClick, label = { Text(stringResource(R.string.possible_spam)) })
                }
            }
        },
        trailingContent = {
            Column(horizontalAlignment = Alignment.End) {
                Text(shortTime(LocalContext.current, c.latestDate), style = MaterialTheme.typography.labelSmall)
                if (unread) Badge(Modifier.padding(top = 4.dp)) { Text(c.unreadCount.toString()) }
            }
        },
    )
}
