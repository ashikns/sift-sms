package dev.ashik.sift.ui.navigation

import androidx.navigation3.runtime.NavKey
import kotlinx.serialization.Serializable

@Serializable
data object ConversationsKey : NavKey

/** A conversation: [threadId] is the provider thread id, or 0 to find the conversation with [address] (an sms: link). */
@Serializable
data class ThreadKey(val threadId: Long, val address: String? = null) : NavKey

@Serializable
data object SettingsKey : NavKey

@Serializable
data object ModelStatsKey : NavKey

@Serializable
data object LearningKey : NavKey
