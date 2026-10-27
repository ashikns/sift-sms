package dev.ashik.sift.ui.navigation

import androidx.navigation3.runtime.NavKey
import kotlinx.serialization.Serializable

@Serializable
data object ConversationsKey : NavKey

/**
 * A conversation. [threadId] is the provider thread id, or 0 for a conversation that doesn't exist yet
 * (composing to [address]); the screen resolves it after the first send.
 */
@Serializable
data class ThreadKey(val threadId: Long, val address: String? = null, val draft: String? = null) : NavKey

@Serializable
data class NewConversationKey(val body: String? = null) : NavKey

@Serializable
data object SettingsKey : NavKey

@Serializable
data object ModelStatsKey : NavKey

@Serializable
data object LearningKey : NavKey
