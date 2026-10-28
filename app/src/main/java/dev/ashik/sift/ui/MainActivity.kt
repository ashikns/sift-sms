package dev.ashik.sift.ui

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.lifecycleScope
import androidx.navigation3.runtime.NavKey
import dev.ashik.sift.SiftApp
import dev.ashik.sift.data.db.Folder
import dev.ashik.sift.ui.navigation.ThreadKey
import dev.ashik.sift.ui.theme.SiftTheme
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    /** A destination requested by the launching intent (notification, sms: link), consumed by the UI. */
    private val pendingDestination = MutableStateFlow<NavKey?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        if (savedInstanceState == null) pendingDestination.value = destinationFor(intent)
        setContent {
            SiftTheme {
                SiftRoot(pendingDestination = pendingDestination, onDestinationConsumed = { pendingDestination.value = null })
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        pendingDestination.value = destinationFor(intent)
    }

    override fun onResume() {
        super.onResume()
        val graph = (application as SiftApp).graph
        lifecycleScope.launch { graph.onUiStart() }
    }

    private fun destinationFor(intent: Intent?): NavKey? {
        intent ?: return null
        val data = intent.data
        return when {
            data?.scheme == "sift" && data.host == "thread" -> data.lastPathSegment?.toLongOrNull()?.let {
                ThreadKey(it, folder = data.getQueryParameter("folder")?.toIntOrNull() ?: Folder.INBOX)
            }
            // Another app asking to text someone: Sift doesn't send, so this opens the conversation with them.
            (intent.action == Intent.ACTION_SENDTO || intent.action == Intent.ACTION_VIEW) &&
                data?.scheme in setOf("sms", "smsto", "mms", "mmsto") -> {
                val address = data?.schemeSpecificPart?.substringBefore('?')?.split(',', ';')?.firstOrNull()?.trim()
                if (address.isNullOrEmpty()) null else ThreadKey(0, address = address)
            }
            else -> null
        }
    }

    companion object {
        /** Opens the thread as the [folder] tab shows it. */
        fun threadIntent(context: Context, threadId: Long, folder: Int): Intent =
            Intent(Intent.ACTION_VIEW, Uri.parse("sift://thread/$threadId?folder=$folder"), context, MainActivity::class.java)
    }
}
