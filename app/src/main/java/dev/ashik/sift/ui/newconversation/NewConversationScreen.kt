package dev.ashik.sift.ui.newconversation

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import dev.ashik.sift.R
import dev.ashik.sift.data.contacts.ContactPhone
import dev.ashik.sift.ui.components.Avatar
import dev.ashik.sift.ui.components.rememberGraph
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/** "Looks like something you can text": at least 3 digits and only phone punctuation. */
fun looksLikeNumber(text: String): Boolean {
    val t = text.trim()
    return t.count { it.isDigit() } >= 3 && t.all { it.isDigit() || it in "+-(). " }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NewConversationScreen(initialBody: String?, onBack: () -> Unit, onPick: (threadId: Long, address: String) -> Unit) {
    val graph = rememberGraph()
    val scope = rememberCoroutineScope()
    var query by rememberSaveable { mutableStateOf("") }
    val focus = remember { FocusRequester() }
    val results by produceState(emptyList<ContactPhone>(), query) {
        delay(150)
        value = graph.contacts.search(query)
    }
    LaunchedEffect(Unit) { focus.requestFocus() }

    fun pick(address: String) = scope.launch {
        val key = graph.normalizer.normalize(address).key
        onPick(graph.db.messages().threadForSender(key) ?: 0L, address)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.action_back)) }
                },
                title = { Text(stringResource(R.string.new_conversation_title)) },
            )
        },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                placeholder = { Text(stringResource(R.string.new_conversation_hint)) },
                singleLine = true,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Go),
                modifier = Modifier.fillMaxWidth().padding(16.dp).focusRequester(focus).testTag("recipient"),
            )
            if (!graph.contacts.hasPermission()) {
                Text(
                    stringResource(R.string.contacts_permission_missing), style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(horizontal = 16.dp),
                )
            }
            LazyColumn(Modifier.fillMaxSize()) {
                if (looksLikeNumber(query)) {
                    item {
                        ListItem(
                            leadingContent = { Icon(Icons.AutoMirrored.Filled.Send, contentDescription = null) },
                            headlineContent = { Text(stringResource(R.string.send_to_number, query.trim())) },
                            modifier = Modifier.clickable { pick(query.trim()) }.testTag("send_to_number"),
                        )
                    }
                }
                items(results, key = { it.number + it.name }) { c ->
                    ListItem(
                        leadingContent = { Avatar(c.name, c.number, c.photoUri) },
                        headlineContent = { Text(c.name) },
                        supportingContent = { Text(listOfNotNull(c.number, c.label).joinToString(" · ")) },
                        modifier = Modifier.clickable { pick(c.number) },
                    )
                }
            }
        }
    }
}
