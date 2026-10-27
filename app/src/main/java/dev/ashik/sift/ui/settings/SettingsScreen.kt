package dev.ashik.sift.ui.settings

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalResources
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.ashik.sift.R
import dev.ashik.sift.data.settings.AppSettings
import dev.ashik.sift.di.AppGraph
import dev.ashik.sift.spamcore.Sensitivity
import dev.ashik.sift.ui.components.rememberGraph
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class SettingsViewModel(private val graph: AppGraph) : ViewModel() {
    val settings: StateFlow<AppSettings> = graph.settings.flow.stateIn(viewModelScope, SharingStarted.Eagerly, AppSettings())

    fun setSensitivity(value: Sensitivity) = viewModelScope.launch { graph.settings.setSensitivity(value) }
    fun setSpamNotifications(value: Boolean) = viewModelScope.launch { graph.settings.setSpamNotifications(value) }
    fun setDeliveryReports(value: Boolean) = viewModelScope.launch { graph.settings.setDeliveryReports(value) }
    fun rescan() = graph.scheduler.enqueueBackfill(rescan = true)
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(onBack: () -> Unit, onOpenStats: () -> Unit, onOpenLearning: () -> Unit) {
    val graph = rememberGraph()
    val vm = viewModel { SettingsViewModel(graph) }
    val s by vm.settings.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val resources = LocalResources.current
    val scope = rememberCoroutineScope()
    val snackbar = remember { SnackbarHostState() }
    val version = remember { runCatching { context.packageManager.getPackageInfo(context.packageName, 0).versionName }.getOrNull().orEmpty() }

    Scaffold(
        topBar = {
            TopAppBar(
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.action_back)) }
                },
                title = { Text(stringResource(R.string.settings)) },
            )
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize().verticalScroll(rememberScrollState())) {
            Header(R.string.settings_filtering)
            Text(stringResource(R.string.sensitivity_title), style = MaterialTheme.typography.titleSmall, modifier = Modifier.padding(horizontal = 16.dp))
            listOf(
                Triple(Sensitivity.CONSERVATIVE, R.string.sensitivity_conservative, R.string.sensitivity_conservative_desc),
                Triple(Sensitivity.BALANCED, R.string.sensitivity_balanced, R.string.sensitivity_balanced_desc),
                Triple(Sensitivity.AGGRESSIVE, R.string.sensitivity_aggressive, R.string.sensitivity_aggressive_desc),
            ).forEach { (value, title, desc) ->
                ListItem(
                    leadingContent = { RadioButton(selected = s.sensitivity == value, onClick = null) },
                    headlineContent = { Text(stringResource(title)) },
                    supportingContent = { Text(stringResource(desc)) },
                    modifier = Modifier.selectable(selected = s.sensitivity == value, onClick = { vm.setSensitivity(value) }),
                )
            }
            SwitchRow(R.string.spam_notifications, R.string.spam_notifications_desc, s.spamNotifications, vm::setSpamNotifications)
            HorizontalDivider()
            Header(R.string.settings_messages)
            SwitchRow(R.string.delivery_reports, R.string.delivery_reports_desc, s.deliveryReports, vm::setDeliveryReports)
            HorizontalDivider()
            Header(R.string.settings_learning)
            ClickRow(R.string.model_stats, R.string.model_stats_desc, onOpenStats)
            ClickRow(R.string.rescan, R.string.rescan_desc) {
                vm.rescan()
                scope.launch { snackbar.showSnackbar(resources.getString(R.string.rescan_started)) }
            }
            ClickRow(R.string.learning_entries, R.string.learning_entries_desc, onOpenLearning)
            HorizontalDivider()
            Header(R.string.settings_about)
            ListItem(
                headlineContent = { Text(stringResource(R.string.about_privacy)) },
                supportingContent = { Text(stringResource(R.string.about_version, version)) },
            )
        }
    }
}

@Composable
private fun Header(text: Int) {
    Text(
        stringResource(text), style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary,
        modifier = Modifier.padding(start = 16.dp, top = 20.dp, bottom = 4.dp),
    )
}

@Composable
private fun SwitchRow(title: Int, desc: Int, checked: Boolean, onChange: (Boolean) -> Unit) {
    ListItem(
        headlineContent = { Text(stringResource(title)) },
        supportingContent = { Text(stringResource(desc)) },
        trailingContent = { Switch(checked = checked, onCheckedChange = null) },
        modifier = Modifier.clickable { onChange(!checked) },
    )
}

@Composable
private fun ClickRow(title: Int, desc: Int, onClick: () -> Unit) {
    ListItem(
        headlineContent = { Text(stringResource(title)) },
        supportingContent = { Text(stringResource(desc)) },
        modifier = Modifier.clickable(onClick = onClick),
    )
}
