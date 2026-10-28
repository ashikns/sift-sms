package dev.ashik.sift

import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.assert
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.hasText
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
import dev.ashik.sift.data.db.Folder
import dev.ashik.sift.data.db.LabelCode
import dev.ashik.sift.data.db.MsgType
import dev.ashik.sift.data.db.ThreadMessageRow
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

    private fun row(
        thread: Long, snippet: String, folder: Int = Folder.INBOX, label: Int = LabelCode.HAM, name: String? = null, unread: Int = 0,
    ) = ConversationRow(
        ConversationEntity(thread, folder, "+1650555000$thread", "+1650555000$thread", thread, 1_800_000_000_000L - thread, snippet,
            MsgType.INBOX, unread, 1, label),
        displayName = name, photoUri = null,
    )

    private fun show(
        inbox: List<ConversationRow>, spam: List<ConversationRow>, onOpenThread: (Long, Int) -> Unit = { _, _ -> },
    ) = compose.setContent {
        SiftTheme {
            ConversationsContent(
                inbox = inbox, spam = spam, scan = null,
                onOpenThread = onOpenThread, onOpenSettings = {},
                onDelete = { _, _ -> },
            )
        }
    }

    @Test
    fun longPressSelects() {
        show(listOf(row(1, "Dinner at 7?", name = "Alex"), row(2, "You won a prize")), emptyList())
        compose.onNodeWithTag("conversation_2").performTouchInput { longClick() }
        compose.onNodeWithText("1 selected").assertIsDisplayed()
        compose.onNodeWithContentDescription("Delete").assertIsDisplayed()
    }

    @Test
    fun suspiciousConversationsGetABadge() {
        show(listOf(row(4, "Your package is waiting", label = LabelCode.SUSPICIOUS)), emptyList())
        compose.onNodeWithText("Possible spam").assertIsDisplayed()
    }

    @Test
    fun unreadFilterShowsOnlyUnreadConversationsInBothTabs() {
        show(
            listOf(row(1, "Your OTP is 4821", unread = 1), row(2, "Your bill is paid")),
            listOf(row(1, "50% off loans", folder = Folder.SPAM), row(3, "You won a prize", folder = Folder.SPAM, unread = 2)),
        )
        compose.onNodeWithTag("conversation_2").assertIsDisplayed()
        compose.onNodeWithTag("filter_unread").performClick()
        compose.onNodeWithTag("conversation_1").assertIsDisplayed()
        compose.onNodeWithTag("conversation_2").assertDoesNotExist()
        compose.onNodeWithTag("tab_spam").performClick()
        compose.onNodeWithTag("conversation_3").assertIsDisplayed()
        compose.onNodeWithTag("conversation_1").assertDoesNotExist()
        compose.onNodeWithTag("filter_unread").performClick()
        compose.onNodeWithTag("conversation_1").assertIsDisplayed()
    }

    @Test
    fun bothTabBadgesCountUnreadConversationsAndOpenTheirOwnView() {
        val opened = mutableListOf<Pair<Long, Int>>()
        show(
            listOf(row(1, "Your OTP is 4821", unread = 1), row(2, "Your bill is paid")),
            listOf(row(1, "50% off loans", folder = Folder.SPAM), row(3, "You won a prize", folder = Folder.SPAM, unread = 2),
                row(4, "Cheap flights", folder = Folder.SPAM, unread = 1)),
            onOpenThread = { thread, folder -> opened += thread to folder },
        )
        compose.onNodeWithTag("tab_inbox").assert(hasText("1"))
        compose.onNodeWithTag("tab_spam").assert(hasText("2")) // 2 unread of 3 spam conversations
        compose.onNodeWithTag("conversation_1").performClick()
        compose.onNodeWithTag("tab_spam").performClick()
        compose.onNodeWithTag("conversation_1").performClick()
        assertEquals(listOf(1L to Folder.INBOX, 1L to Folder.SPAM), opened)
    }

    @Test
    fun messageTextIsPlainAndTapOpensTheMenu() {
        val m = ThreadMessageRow(
            9, 3, "+16505550003", "Pay your toll: https://ezpass-tolls.top/pay", 0, MsgType.INBOX, -1,
            true, LabelCode.SPAM, "ENGINE", 0.97f, "MODEL", null, null, 1, 0,
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
