package dev.ashik.sift.ui.navigation

import androidx.navigation3.runtime.NavKey
import dev.ashik.sift.data.db.Folder
import kotlinx.serialization.Serializable

@Serializable
data object ConversationsKey : NavKey

/**
 * A conversation as one tab shows it (only its messages in [folder]): [threadId] is the provider thread id, or 0 to find
 * the conversation with [address] (an sms: link). Opened from a search, it scrolls to [messageId] (the message that
 * matched) and highlights [query] in the messages.
 */
@Serializable
data class ThreadKey(
    val threadId: Long,
    val folder: Int = Folder.INBOX,
    val address: String? = null,
    val messageId: Long? = null,
    val query: String? = null,
) : NavKey

@Serializable
data object SettingsKey : NavKey

@Serializable
data object ModelStatsKey : NavKey

@Serializable
data object LearningKey : NavKey
