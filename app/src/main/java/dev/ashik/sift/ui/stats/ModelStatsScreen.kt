package dev.ashik.sift.ui.stats

import android.text.format.DateUtils
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.ashik.sift.R
import dev.ashik.sift.di.AppGraph
import dev.ashik.sift.spamcore.EngineInfo
import dev.ashik.sift.spamcore.UserLabel
import dev.ashik.sift.ui.components.rememberGraph
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class WeekStat(val weekStart: Long, val corrections: Int, val incoming: Int)

class ModelStatsViewModel(graph: AppGraph) : ViewModel() {
    val info: StateFlow<EngineInfo> = graph.engine.info
    val markedSpam = graph.db.feedback().observeExplicitCount(UserLabel.SPAM.ordinal)
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), 0)
    val markedHam = graph.db.feedback().observeExplicitCount(UserLabel.HAM.ordinal)
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), 0)
    val latency = MutableStateFlow(graph.latency.summary())
    val weeks = MutableStateFlow<List<WeekStat>>(emptyList())

    init {
        viewModelScope.launch {
            val since = System.currentTimeMillis() - 8 * WEEK_MS
            val corrections = graph.db.feedback().correctionsPerWeek(since).associate { it.week to it.count }
            val incoming = graph.db.messages().incomingPerWeek(since).associate { it.week to it.count }
            weeks.value = (corrections.keys + incoming.keys).distinct().sortedDescending().map { w ->
                WeekStat(w * WEEK_MS, corrections[w] ?: 0, incoming[w] ?: 0)
            }
        }
    }

    private companion object {
        const val WEEK_MS = 604_800_000L
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelStatsScreen(onBack: () -> Unit) {
    val graph = rememberGraph()
    val vm = viewModel { ModelStatsViewModel(graph) }
    val info by vm.info.collectAsStateWithLifecycle()
    val spam by vm.markedSpam.collectAsStateWithLifecycle()
    val ham by vm.markedHam.collectAsStateWithLifecycle()
    val latency by vm.latency.collectAsStateWithLifecycle()
    val weeks by vm.weeks.collectAsStateWithLifecycle()

    Scaffold(
        topBar = {
            TopAppBar(
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.action_back)) }
                },
                title = { Text(stringResource(R.string.model_stats)) },
            )
        },
    ) { padding ->
        Column(
            Modifier.padding(padding).fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Section(R.string.stats_model) {
                Text(stringResource(if (info.encoderAvailable) R.string.stats_encoder_on else R.string.stats_encoder_off))
                Text(stringResource(R.string.stats_bundle, info.bundleVersion))
                Text(stringResource(R.string.stats_version, info.modelVersion))
                Text(
                    info.lastTrainedAt?.let {
                        stringResource(R.string.stats_last_trained, DateUtils.getRelativeTimeSpanString(it).toString())
                    } ?: stringResource(R.string.stats_never_trained),
                )
            }
            Section(R.string.stats_feedback) {
                Text(stringResource(R.string.stats_marked_spam, spam))
                Text(stringResource(R.string.stats_marked_not_spam, ham))
            }
            Section(R.string.stats_speed) {
                val l = latency
                Text(if (l == null) stringResource(R.string.stats_latency_none) else stringResource(R.string.stats_latency, l.first, l.second, l.third))
            }
            Section(R.string.stats_corrections) {
                Text(stringResource(R.string.stats_corrections_desc), style = MaterialTheme.typography.bodySmall)
                if (weeks.none { it.corrections > 0 }) Text(stringResource(R.string.stats_no_corrections))
                weeks.forEach { w ->
                    val share = if (w.incoming > 0) "%.0f%%".format(100.0 * w.corrections / w.incoming) else "—"
                    Text(
                        stringResource(
                            R.string.stats_week,
                            DateUtils.formatDateTime(LocalContext.current, w.weekStart, DateUtils.FORMAT_SHOW_DATE or DateUtils.FORMAT_ABBREV_MONTH or DateUtils.FORMAT_NO_YEAR),
                            w.corrections, w.incoming, share,
                        ),
                    )
                }
            }
        }
    }
}

@Composable
private fun Section(title: Int, content: @Composable () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(stringResource(title), style = MaterialTheme.typography.titleMedium)
            content()
        }
    }
}
