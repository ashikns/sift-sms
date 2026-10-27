package dev.ashik.sift.ui.learning

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.SnackbarResult
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalResources
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.compose.LifecycleEventEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.ashik.sift.R
import dev.ashik.sift.data.db.LearningEntryRow
import dev.ashik.sift.di.AppGraph
import dev.ashik.sift.spamcore.UserLabel
import dev.ashik.sift.ui.components.displayAddress
import dev.ashik.sift.ui.components.rememberGraph
import dev.ashik.sift.ui.components.shortTime
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

/**
 * The user's marks the engine learns from. Deleting one only hides it here; the engine forgets everything deleted, and
 * retrains once, when the page closes (or the app goes to the background, so a killed process can't lose deletions).
 */
class LearningViewModel(private val graph: AppGraph) : ViewModel() {
    private val deleted = MutableStateFlow<Set<Long>>(emptySet())
    private var committed = emptySet<Long>()

    val entries: StateFlow<List<LearningEntryRow>?> =
        combine(graph.db.trainingExamples().observeEntries(), deleted) { all, gone -> all.filterNot { it.messageId in gone } }
            .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)

    fun delete(ids: Collection<Long>) = deleted.update { it + ids }

    fun restore(ids: Collection<Long>) = deleted.update { it - ids.toSet() }

    /** Called on the main thread, when the page closes or the app goes to the background. */
    fun commit() {
        val ids = deleted.value - committed
        if (ids.isEmpty()) return
        committed = committed + ids
        graph.appScope.launch { graph.feedback.forgetLearned(ids) }
    }

    override fun onCleared() = commit()
}

@Composable
fun LearningScreen(onBack: () -> Unit) {
    val graph = rememberGraph()
    val vm = viewModel { LearningViewModel(graph) }
    val entries by vm.entries.collectAsStateWithLifecycle()
    val snackbar = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()
    val resources = LocalResources.current
    LifecycleEventEffect(Lifecycle.Event.ON_STOP) {
        snackbar.currentSnackbarData?.dismiss() // committed deletions can't be undone
        vm.commit()
    }
    fun deleteWithUndo(ids: List<Long>, message: Int) {
        vm.delete(ids)
        scope.launch {
            snackbar.currentSnackbarData?.dismiss()
            val action = snackbar.showSnackbar(resources.getString(message), actionLabel = resources.getString(R.string.action_undo), duration = SnackbarDuration.Long)
            if (action == SnackbarResult.ActionPerformed) vm.restore(ids)
        }
    }
    LearningContent(
        entries = entries, snackbar = snackbar, onBack = onBack,
        onDelete = { deleteWithUndo(listOf(it.messageId), R.string.learning_deleted) },
        onDeleteAll = { deleteWithUndo(entries.orEmpty().map { it.messageId }, R.string.learning_deleted_all) },
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LearningContent(
    entries: List<LearningEntryRow>?,
    snackbar: SnackbarHostState,
    onBack: () -> Unit,
    onDelete: (LearningEntryRow) -> Unit,
    onDeleteAll: () -> Unit,
) {
    var menu by remember { mutableStateOf(false) }
    var confirmDeleteAll by remember { mutableStateOf(false) }
    Scaffold(
        topBar = {
            TopAppBar(
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.action_back)) }
                },
                title = { Text(stringResource(R.string.learning_entries)) },
                actions = {
                    if (!entries.isNullOrEmpty()) {
                        Box {
                            IconButton(onClick = { menu = true }) { Icon(Icons.Default.MoreVert, stringResource(R.string.action_more)) }
                            DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                                DropdownMenuItem(
                                    text = { Text(stringResource(R.string.learning_delete_all)) },
                                    onClick = { menu = false; confirmDeleteAll = true },
                                )
                            }
                        }
                    }
                },
            )
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { padding ->
        when {
            entries == null -> Box(Modifier.padding(padding).fillMaxSize(), contentAlignment = Alignment.Center) { CircularProgressIndicator() }
            entries.isEmpty() -> Box(Modifier.padding(padding).fillMaxSize().padding(32.dp), contentAlignment = Alignment.Center) {
                Text(
                    stringResource(R.string.learning_empty), style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center,
                )
            }
            else -> LazyColumn(Modifier.padding(padding).fillMaxSize().testTag("learning_list")) {
                item(key = "intro") {
                    val spam = entries.count { it.label == UserLabel.SPAM.ordinal }
                    Column(Modifier.padding(horizontal = 16.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            stringResource(R.string.learning_intro), style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Text(
                            pluralStringResource(R.plurals.learning_count, entries.size, entries.size, spam, entries.size - spam),
                            style = MaterialTheme.typography.labelLarge,
                        )
                    }
                }
                items(entries, key = { it.messageId }) { entry ->
                    EntryRow(entry, onDelete = { onDelete(entry) }, Modifier.animateItem())
                }
            }
        }
    }

    if (confirmDeleteAll) {
        AlertDialog(
            onDismissRequest = { confirmDeleteAll = false },
            title = { Text(stringResource(R.string.learning_delete_all_title)) },
            text = { Text(stringResource(R.string.learning_delete_all_text)) },
            confirmButton = {
                TextButton(onClick = { confirmDeleteAll = false; onDeleteAll() }) { Text(stringResource(R.string.learning_delete_all)) }
            },
            dismissButton = { TextButton(onClick = { confirmDeleteAll = false }) { Text(stringResource(R.string.action_cancel)) } },
        )
    }
}

@Composable
private fun EntryRow(entry: LearningEntryRow, onDelete: () -> Unit, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val spam = entry.label == UserLabel.SPAM.ordinal
    val sender = entry.displayName ?: entry.address.takeIf { it.isNotEmpty() }?.let(::displayAddress)
    ListItem(
        overlineContent = {
            Text(
                stringResource(if (spam) R.string.learning_spam else R.string.learning_not_spam),
                color = if (spam) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
            )
        },
        headlineContent = { Text(entry.body, maxLines = 3, overflow = TextOverflow.Ellipsis) },
        supportingContent = { Text(listOfNotNull(sender, shortTime(context, entry.createdAt)).joinToString(" · ")) },
        trailingContent = {
            IconButton(onClick = onDelete) { Icon(Icons.Default.Delete, stringResource(R.string.learning_delete)) }
        },
        // One stop for screen readers: label, text, sender and date together; the delete button stays separate.
        modifier = modifier.semantics(mergeDescendants = true) {}.testTag("learning_${entry.messageId}"),
    )
}
