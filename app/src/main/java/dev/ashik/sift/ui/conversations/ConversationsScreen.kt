package dev.ashik.sift.ui.conversations

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Badge
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.IconToggleButton
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
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
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
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.ashik.sift.R
import dev.ashik.sift.data.db.ConversationRow
import dev.ashik.sift.data.db.Folder
import dev.ashik.sift.data.db.LabelCode
import dev.ashik.sift.data.db.MsgType
import dev.ashik.sift.data.db.SearchableMessage
import dev.ashik.sift.ui.components.Avatar
import dev.ashik.sift.ui.components.FilterListIcon
import dev.ashik.sift.ui.components.displayAddress
import dev.ashik.sift.ui.components.highlighted
import dev.ashik.sift.ui.components.rememberGraph
import dev.ashik.sift.ui.components.shortTime
import dev.ashik.sift.ui.navigation.ThreadKey

@Composable
fun ConversationsScreen(onOpenThread: (ThreadKey) -> Unit, onOpenSettings: () -> Unit) {
    val graph = rememberGraph()
    val context = LocalContext.current
    val vm = viewModel { ConversationsViewModel(graph, context.applicationContext) }
    val inbox by vm.inbox.collectAsStateWithLifecycle()
    val spam by vm.spam.collectAsStateWithLifecycle()
    val scan by vm.scan.collectAsStateWithLifecycle()
    val search by vm.search.collectAsStateWithLifecycle()
    ConversationsContent(
        inbox = inbox, spam = spam, scan = scan, search = search,
        onOpenThread = onOpenThread, onOpenSettings = onOpenSettings,
        onSearch = vm::setQuery, onDelete = vm::delete,
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConversationsContent(
    inbox: List<ConversationRow>?,
    spam: List<ConversationRow>?,
    scan: ScanProgress?,
    search: SearchResults?,
    onOpenThread: (ThreadKey) -> Unit,
    onOpenSettings: () -> Unit,
    onSearch: (query: String) -> Unit,
    onDelete: (threadIds: Set<Long>, folder: Int) -> Unit,
) {
    var tab by rememberSaveable { mutableIntStateOf(0) }
    // Kept while Sift is open (into a conversation and back), not across launches.
    var unreadOnly by rememberSaveable { mutableStateOf(false) }
    var searching by rememberSaveable { mutableStateOf(false) }
    var query by rememberSaveable { mutableStateOf("") }
    var selection by remember { mutableStateOf(emptySet<Long>()) }
    var confirmDelete by remember { mutableStateOf(false) }
    LaunchedEffect(searching, query) { onSearch(if (searching) query else "") }
    // Back cancels a selection first (the later handler wins), then closes the search.
    BackHandler(enabled = searching) { searching = false; query = "" }
    BackHandler(enabled = selection.isNotEmpty()) { selection = emptySet() }
    val inSpam = tab == 1
    val folder = if (inSpam) Folder.SPAM else Folder.INBOX
    // Only spaces or punctuation is no search: the tab lists everything.
    val highlight = query.takeIf { searching && SearchText.key(it).isNotEmpty() }
    val all = if (highlight == null) {
        val rows = if (inSpam) spam else inbox
        remember(rows) { rows?.map { SearchHit(it, null) } }
    } else {
        search?.let { if (inSpam) it.spam else it.inbox }
    }
    val rows = if (unreadOnly) all?.filter { it.row.conversation.unreadCount > 0 } else all
    // One list state per tab, and one for search results. A LazyColumn with stable keys keeps its first visible item in
    // place when a new conversation is inserted above it, so a new message would land just off-screen; follow it if we
    // were at the top.
    val inboxState = rememberLazyListState()
    val spamState = rememberLazyListState()
    val searchState = rememberLazyListState()
    val listState = when {
        highlight != null -> searchState
        inSpam -> spamState
        else -> inboxState
    }
    val topKey = rows?.firstOrNull()?.row?.conversation?.threadId
    LaunchedEffect(topKey, listState) {
        if (topKey != null && listState.firstVisibleItemIndex <= 1) listState.scrollToItem(0)
    }
    // A new search, or the other tab's results, start at the top.
    LaunchedEffect(highlight, inSpam) { if (highlight != null) searchState.scrollToItem(0) }

    Scaffold(
        topBar = {
            val filter = @Composable { UnreadFilter(checked = unreadOnly, onCheckedChange = { unreadOnly = it }) }
            when {
                selection.isNotEmpty() -> TopAppBar(
                    navigationIcon = {
                        IconButton(onClick = { selection = emptySet() }) {
                            Icon(Icons.Default.Close, stringResource(R.string.action_close_selection))
                        }
                    },
                    title = { Text(stringResource(R.string.selected_count, selection.size)) },
                    actions = {
                        IconButton(onClick = { confirmDelete = true }) { Icon(Icons.Default.Delete, stringResource(R.string.action_delete)) }
                    },
                )
                searching -> SearchBar(
                    query = query, onQueryChange = { query = it }, onClose = { searching = false; query = "" },
                    actions = { filter() },
                )
                else -> TopAppBar(
                    title = { Text(stringResource(R.string.app_name)) },
                    actions = {
                        IconButton(onClick = { searching = true }, modifier = Modifier.testTag("search")) {
                            Icon(Icons.Default.Search, stringResource(R.string.action_search))
                        }
                        filter()
                        IconButton(onClick = onOpenSettings) { Icon(Icons.Default.Settings, stringResource(R.string.settings)) }
                    },
                )
            }
        },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            PrimaryTabRow(selectedTabIndex = tab) {
                Tab(
                    selected = tab == 0, onClick = { tab = 0; selection = emptySet() },
                    text = { TabLabel(stringResource(R.string.tab_inbox), inbox.unreadConversations()) },
                    modifier = Modifier.testTag("tab_inbox"),
                )
                Tab(
                    selected = tab == 1, onClick = { tab = 1; selection = emptySet() },
                    text = { TabLabel(stringResource(R.string.tab_spam), spam.unreadConversations()) },
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
                        stringResource(
                            when {
                                highlight != null -> R.string.empty_search
                                unreadOnly -> R.string.empty_unread
                                inSpam -> R.string.empty_spam
                                else -> R.string.empty_inbox
                            },
                        ),
                        style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                else -> LazyColumn(Modifier.fillMaxSize().testTag("conversation_list"), state = listState) {
                    items(rows, key = { it.row.conversation.threadId }) { hit ->
                        val id = hit.row.conversation.threadId
                        ConversationItem(
                            row = hit.row,
                            selected = id in selection,
                            onClick = {
                                if (selection.isNotEmpty()) {
                                    selection = selection.toggle(id)
                                } else {
                                    onOpenThread(ThreadKey(id, folder, messageId = hit.match?.id, query = highlight))
                                }
                            },
                            onLongClick = { selection = selection.toggle(id) },
                            match = hit.match,
                            query = highlight,
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
                TextButton(onClick = { onDelete(selection, folder); selection = emptySet(); confirmDelete = false }) {
                    Text(stringResource(R.string.action_delete))
                }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text(stringResource(R.string.action_cancel)) } },
        )
    }
}

private fun Set<Long>.toggle(id: Long) = if (id in this) this - id else this + id

@Composable
private fun UnreadFilter(checked: Boolean, onCheckedChange: (Boolean) -> Unit) {
    IconToggleButton(
        checked = checked, onCheckedChange = onCheckedChange,
        colors = IconButtonDefaults.iconToggleButtonColors(
            checkedContainerColor = MaterialTheme.colorScheme.secondaryContainer,
            checkedContentColor = MaterialTheme.colorScheme.onSecondaryContainer,
        ),
        modifier = Modifier.testTag("filter_unread"),
    ) { Icon(FilterListIcon, stringResource(R.string.filter_unread_only)) }
}

/** The top bar while searching: the search box, Clear, then [actions]. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SearchBar(query: String, onQueryChange: (String) -> Unit, onClose: () -> Unit, actions: @Composable RowScope.() -> Unit) {
    val focus = remember { FocusRequester() }
    val keyboard = LocalSoftwareKeyboardController.current
    // Opening the search brings up the keyboard; coming back to a search from a conversation doesn't.
    LaunchedEffect(Unit) { if (query.isEmpty()) focus.requestFocus() }
    TopAppBar(
        navigationIcon = {
            IconButton(onClick = onClose) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.action_close_search)) }
        },
        title = {
            TextField(
                value = query, onValueChange = onQueryChange, singleLine = true,
                textStyle = MaterialTheme.typography.bodyLarge,
                placeholder = { Text(stringResource(R.string.search_hint), maxLines = 1, overflow = TextOverflow.Ellipsis) },
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
                keyboardActions = KeyboardActions(onSearch = { keyboard?.hide() }),
                colors = TextFieldDefaults.colors(
                    focusedContainerColor = Color.Transparent, unfocusedContainerColor = Color.Transparent,
                    focusedIndicatorColor = Color.Transparent, unfocusedIndicatorColor = Color.Transparent,
                ),
                modifier = Modifier.fillMaxWidth().focusRequester(focus).testTag("search_field"),
            )
        },
        actions = {
            if (query.isNotEmpty()) {
                IconButton(onClick = { onQueryChange(""); focus.requestFocus() }, modifier = Modifier.testTag("search_clear")) {
                    Icon(Icons.Default.Close, stringResource(R.string.action_clear_search))
                }
            }
            actions()
        },
    )
}

private fun List<ConversationRow>?.unreadConversations() = this?.count { it.conversation.unreadCount > 0 } ?: 0

@Composable
private fun TabLabel(text: String, count: Int) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(text)
        if (count > 0) Badge { Text(count.toString()) }
    }
}

/** A conversation row; from a search, [query] is highlighted and [match] (the message that matched) is its snippet. */
@OptIn(ExperimentalFoundationApi::class)
@Composable
fun ConversationItem(
    row: ConversationRow,
    selected: Boolean,
    onClick: () -> Unit,
    onLongClick: () -> Unit,
    match: SearchableMessage? = null,
    query: String? = null,
) {
    val c = row.conversation
    val name = row.displayName ?: displayAddress(c.address)
    val unread = c.unreadCount > 0
    val text = if (match != null) SearchText.excerpt(match.body, query) else c.snippet
    val snippet = if ((match?.type ?: c.latestType) != MsgType.INBOX) stringResource(R.string.you_prefix, text) else text
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
            Text(
                highlighted(name, query), maxLines = 1, overflow = TextOverflow.Ellipsis,
                fontWeight = if (unread) FontWeight.Bold else FontWeight.Normal,
            )
        },
        supportingContent = {
            Column {
                Text(
                    highlighted(snippet, query), maxLines = 2, overflow = TextOverflow.Ellipsis,
                    fontWeight = if (unread) FontWeight.SemiBold else FontWeight.Normal,
                )
                if (c.latestLabel == LabelCode.SUSPICIOUS) {
                    SuggestionChip(onClick = onClick, label = { Text(stringResource(R.string.possible_spam)) })
                }
            }
        },
        trailingContent = {
            Column(horizontalAlignment = Alignment.End) {
                Text(shortTime(LocalContext.current, match?.date ?: c.latestDate), style = MaterialTheme.typography.labelSmall)
                if (unread) Badge(Modifier.padding(top = 4.dp)) { Text(c.unreadCount.toString()) }
            }
        },
    )
}
