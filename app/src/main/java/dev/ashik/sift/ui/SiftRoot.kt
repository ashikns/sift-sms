package dev.ashik.sift.ui

import androidx.compose.animation.AnimatedContentTransitionScope
import androidx.compose.animation.AnimatedContentTransitionScope.SlideDirection
import androidx.compose.animation.ContentTransform
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.animation.togetherWith
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.unit.IntOffset
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.navigation3.rememberViewModelStoreNavEntryDecorator
import androidx.navigation3.runtime.NavKey
import androidx.navigation3.runtime.entryProvider
import androidx.navigation3.runtime.rememberNavBackStack
import androidx.navigation3.runtime.rememberSaveableStateHolderNavEntryDecorator
import androidx.navigation3.scene.Scene
import androidx.navigation3.ui.NavDisplay
import dev.ashik.sift.ui.components.rememberGraph
import dev.ashik.sift.ui.conversations.ConversationsScreen
import dev.ashik.sift.ui.navigation.ConversationsKey
import dev.ashik.sift.ui.navigation.ModelStatsKey
import dev.ashik.sift.ui.navigation.NewConversationKey
import dev.ashik.sift.ui.navigation.SettingsKey
import dev.ashik.sift.ui.navigation.ThreadKey
import dev.ashik.sift.ui.newconversation.NewConversationScreen
import dev.ashik.sift.ui.onboarding.OnboardingScreen
import dev.ashik.sift.ui.settings.SettingsScreen
import dev.ashik.sift.ui.stats.ModelStatsScreen
import dev.ashik.sift.ui.thread.ThreadScreen
import kotlinx.coroutines.flow.StateFlow

/** Role gate, then the Navigation 3 back stack. */
@Composable
fun SiftRoot(pendingDestination: StateFlow<NavKey?>, onDestinationConsumed: () -> Unit) {
    val graph = rememberGraph()
    val roleHeld by graph.role.held.collectAsStateWithLifecycle()
    if (!roleHeld) {
        OnboardingScreen()
        return
    }
    val backStack = rememberNavBackStack(ConversationsKey)
    val destination by pendingDestination.collectAsStateWithLifecycle()
    LaunchedEffect(destination) {
        val key = destination ?: return@LaunchedEffect
        while (backStack.size > 1) backStack.removeAt(backStack.lastIndex)
        backStack.add(key)
        onDestinationConsumed()
    }
    NavDisplay(
        backStack = backStack,
        onBack = { if (backStack.size > 1) backStack.removeAt(backStack.lastIndex) },
        entryDecorators = listOf(
            rememberSaveableStateHolderNavEntryDecorator(),
            rememberViewModelStoreNavEntryDecorator(),
        ),
        transitionSpec = { slideForward() },
        popTransitionSpec = { slideBack() },
        predictivePopTransitionSpec = { slideBack() },
        entryProvider = entryProvider {
            entry<ConversationsKey> {
                ConversationsScreen(
                    onOpenThread = { backStack.add(ThreadKey(it)) },
                    onNewMessage = { backStack.add(NewConversationKey()) },
                    onOpenSettings = { backStack.add(SettingsKey) },
                )
            }
            entry<ThreadKey> { key ->
                ThreadScreen(key = key, onBack = { backStack.removeAt(backStack.lastIndex) })
            }
            entry<NewConversationKey> { key ->
                NewConversationScreen(
                    initialBody = key.body,
                    onBack = { backStack.removeAt(backStack.lastIndex) },
                    onPick = { threadId, address ->
                        backStack.removeAt(backStack.lastIndex)
                        backStack.add(ThreadKey(threadId, address, key.body))
                    },
                )
            }
            entry<SettingsKey> {
                SettingsScreen(
                    onBack = { backStack.removeAt(backStack.lastIndex) },
                    onOpenStats = { backStack.add(ModelStatsKey) },
                )
            }
            entry<ModelStatsKey> {
                ModelStatsScreen(onBack = { backStack.removeAt(backStack.lastIndex) })
            }
        },
    )
}

/** Page slides: a screen opens by sliding in from the end edge over the one below it, and slides back out on back. */
private val pageSlide = tween<IntOffset>(durationMillis = 400, easing = FastOutSlowInEasing)

private fun <T : Any> AnimatedContentTransitionScope<Scene<T>>.slideForward(): ContentTransform =
    slideIntoContainer(SlideDirection.Start, pageSlide) togetherWith slideOutOfContainer(SlideDirection.Start, pageSlide) { it / 3 }

/** Also drives the back gesture, so the screen follows the finger off toward the end edge. */
private fun <T : Any> AnimatedContentTransitionScope<Scene<T>>.slideBack(): ContentTransform =
    slideIntoContainer(SlideDirection.End, pageSlide) { it / 3 } togetherWith slideOutOfContainer(SlideDirection.End, pageSlide)
