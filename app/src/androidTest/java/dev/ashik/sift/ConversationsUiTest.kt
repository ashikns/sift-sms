package dev.ashik.sift

import androidx.compose.material3.SnackbarHostState
import androidx.compose.runtime.remember
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.longClick
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTouchInput
import androidx.test.ext.junit.runners.AndroidJUnit4
import dev.ashik.sift.data.db.ConversationEntity
import dev.ashik.sift.data.db.ConversationRow
import dev.ashik.sift.data.db.LabelCode
import dev.ashik.sift.data.db.MsgType
import dev.ashik.sift.data.db.ThreadMessageRow
import dev.ashik.sift.spamcore.UserLabel
import dev.ashik.sift.ui.conversations.ConversationsContent
import dev.ashik.sift.ui.theme.SiftTheme
import dev.ashik.sift.ui.thread.MessageBubble
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ConversationsUiTest {
    @get:Rule
    val compose = createComposeRule()

    private fun row(thread: Long, snippet: String, folder: Int = 0, label: Int = LabelCode.HAM, name: String? = null) = ConversationRow(
        ConversationEntity(thread, "+1650555000$thread", "+1650555000$thread", thread, 1_800_000_000_000L - thread, snippet, MsgType.INBOX,
            thread, 0, 1, folder, label),
        displayName = name, photoUri = null,
    )

    private var marked: Pair<Set<Long>, UserLabel>? = null

    private fun show(inbox: List<ConversationRow>, spam: List<ConversationRow>) = compose.setContent {
        SiftTheme {
            ConversationsContent(
                inbox = inbox, spam = spam, scan = null, snackbar = remember { SnackbarHostState() },
                onOpenThread = {}, onOpenSettings = {},
                onMark = { ids, label -> marked = ids to label }, onMarkRead = {}, onDelete = {},
            )
        }
    }

    @Test
    fun longPressSelectsAndMarksSpam() {
        show(listOf(row(1, "Dinner at 7?", name = "Alex"), row(2, "You won a prize")), emptyList())
        compose.onNodeWithTag("conversation_2").performTouchInput { longClick() }
        compose.onNodeWithText("1 selected").assertIsDisplayed()
        compose.onNodeWithContentDescription("Spam").performClick()
        compose.waitForIdle()
        assertEquals(setOf(2L) to UserLabel.SPAM, marked)
    }

    @Test
    fun spamTabOffersNotSpam() {
        show(listOf(row(1, "Dinner at 7?")), listOf(row(3, "Claim your gift card", folder = 1, label = LabelCode.SPAM)))
        compose.onNodeWithTag("tab_spam").performClick()
        compose.onNodeWithText("Claim your gift card").assertIsDisplayed()
        compose.onNodeWithTag("conversation_3").performTouchInput { longClick() }
        compose.onNodeWithContentDescription("Not spam").performClick()
        compose.waitForIdle()
        assertEquals(setOf(3L) to UserLabel.HAM, marked)
    }

    @Test
    fun suspiciousConversationsGetABadge() {
        show(listOf(row(4, "Your package is waiting", label = LabelCode.SUSPICIOUS)), emptyList())
        compose.onNodeWithText("Possible spam").assertIsDisplayed()
    }

    @Test
    fun messageTextIsPlainAndTapOpensTheMenu() {
        val m = ThreadMessageRow(
            9, 3, "+16505550003", "+16505550003", "PHONE", "Pay your toll: https://ezpass-tolls.top/pay", 0, 0, MsgType.INBOX, -1, 0,
            true, LabelCode.SPAM, "ENGINE", 0.97f, "MODEL", null, null, null, 1, 0,
        )
        var why = false
        compose.setContent { SiftTheme { MessageBubble(m, onWhy = { why = true }, onMark = {}, onDelete = {}) } }
        // No hyperlinks: the URL is plain text with no link annotation to open.
        val text = compose.onNodeWithTag("body_9", useUnmergedTree = true).fetchSemanticsNode().config[SemanticsProperties.Text].single()
        assertTrue(text.getLinkAnnotations(0, text.length).isEmpty())
        compose.onNodeWithTag("chip_9").performClick()
        compose.waitForIdle()
        assertEquals(true, why)
        // The text is selectable, which must not swallow a plain tap: that still opens the message's menu.
        compose.onNodeWithTag("body_9", useUnmergedTree = true).performClick()
        compose.onNodeWithText("Copy text").assertIsDisplayed()
    }
}
