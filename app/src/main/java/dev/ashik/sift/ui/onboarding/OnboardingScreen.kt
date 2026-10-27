package dev.ashik.sift.ui.onboarding

import android.content.Intent
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Icon
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import dev.ashik.sift.R
import dev.ashik.sift.ui.components.rememberGraph

@Composable
fun OnboardingScreen() {
    val graph = rememberGraph()
    val context = LocalContext.current
    var denied by rememberSaveable { mutableStateOf(false) }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) {
        denied = !graph.role.refresh()
    }
    Surface(Modifier.fillMaxSize()) {
        Column(
            Modifier.safeDrawingPadding().verticalScroll(rememberScrollState()).padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Spacer(Modifier.height(24.dp))
            Icon(
                painterResource(R.drawable.ic_stat_message), contentDescription = null,
                tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(64.dp),
            )
            Text(stringResource(R.string.onboarding_title), style = MaterialTheme.typography.headlineMedium, textAlign = TextAlign.Center)
            Text(stringResource(R.string.onboarding_body), style = MaterialTheme.typography.bodyLarge, textAlign = TextAlign.Center)
            Card(Modifier.fillMaxWidth()) {
                Note(Icons.Default.Lock, stringResource(R.string.onboarding_privacy))
                Note(Icons.Default.Info, stringResource(R.string.onboarding_caveat_rcs))
                Note(Icons.Default.Warning, stringResource(R.string.onboarding_caveat_mms))
            }
            Button(onClick = { launcher.launch(graph.role.requestIntent()) }, modifier = Modifier.fillMaxWidth()) {
                Text(stringResource(R.string.onboarding_button))
            }
            if (denied) {
                Text(stringResource(R.string.onboarding_denied), color = MaterialTheme.colorScheme.error)
                OutlinedButton(
                    onClick = {
                        context.startActivity(Intent(Settings.ACTION_MANAGE_DEFAULT_APPS_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) { Text(stringResource(R.string.onboarding_open_settings)) }
            }
        }
    }
}

@Composable
private fun Note(icon: ImageVector, text: String) {
    ListItem(
        leadingContent = { Icon(icon, contentDescription = null) },
        headlineContent = { Text(text, style = MaterialTheme.typography.bodyMedium) },
    )
}
