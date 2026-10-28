package dev.ashik.sift

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.assert
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.getBoundsInRoot
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.longClick
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import androidx.compose.ui.test.performTouchInput
import androidx.test.ext.junit.runners.AndroidJUnit4
import dev.ashik.sift.data.db.ConversationEntity
import dev.ashik.sift.data.db.ConversationRow
import dev.ashik.sift.data.db.Folder
import dev.ashik.sift.data.db.LabelCode
import dev.ashik.sift.data.db.MsgType
import dev.ashik.sift.data.db.SearchableMessage
import dev.ashik.sift.data.db.ThreadMessageRow
import dev.ashik.sift.ui.conversations.ConversationsContent
import dev.ashik.sift.ui.conversations.SearchIndex
import dev.ashik.sift.ui.conversations.SearchResults
import dev.ashik.sift.ui.conversations.SearchText
import dev.ashik.sift.ui.navigation.ThreadKey
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
        address: String = "+1650555000$thread",
    ) = ConversationRow(
        ConversationEntity(thread, folder, address, address, thread, 1_800_000_000_000L - thread, snippet,
            MsgType.INBOX, unread, 1, label),
        displayName = name, photoUri = null,
    )

    /** The conversation list, searching [messages] like the view model does. */
    private fun show(
        inbox: List<ConversationRow>, spam: List<ConversationRow>, messages: List<SearchableMessage> = emptyList(),
        onOpenThread: (ThreadKey) -> Unit = {},
    ) = compose.setContent {
        val index = remember { SearchIndex(messages) }
        var search by remember { mutableStateOf<SearchResults?>(null) }
        SiftTheme {
            ConversationsContent(
                inbox = inbox, spam = spam, scan = null, search = search,
                onOpenThread = onOpenThread, onOpenSettings = {},
                onSearch = { q -> search = SearchText.key(q).takeIf { it.isNotEmpty() }?.let { index.search(inbox, spam, it) } },
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
        val opened = mutableListOf<ThreadKey>()
        show(
            listOf(row(1, "Your OTP is 4821", unread = 1), row(2, "Your bill is paid")),
            listOf(row(1, "50% off loans", folder = Folder.SPAM), row(3, "You won a prize", folder = Folder.SPAM, unread = 2),
                row(4, "Cheap flights", folder = Folder.SPAM, unread = 1)),
            onOpenThread = { opened += it },
        )
        compose.onNodeWithTag("tab_inbox").assert(hasText("1"))
        compose.onNodeWithTag("tab_spam").assert(hasText("2")) // 2 unread of 3 spam conversations
        compose.onNodeWithTag("conversation_1").performClick()
        compose.onNodeWithTag("tab_spam").performClick()
        compose.onNodeWithTag("conversation_1").performClick()
        assertEquals(listOf(ThreadKey(1, Folder.INBOX), ThreadKey(1, Folder.SPAM)), opened)
    }

    @Test
    fun searchSitsLeftOfTheUnreadFilter() {
        show(listOf(row(1, "Dinner at 7?")), emptyList())
        val search = compose.onNodeWithTag("search").getBoundsInRoot()
        val filter = compose.onNodeWithTag("filter_unread").getBoundsInRoot()
        assertTrue(search.right <= filter.left)
    }

    private val searchInbox = listOf(
        row(1, "Rs 2,500 debited from a/c", name = "HDFCBK", unread = 1),
        row(2, "did the hdfc salary come?", name = "Amit"),
        row(3, "Dinner at 7?", name = "Alex"),
        row(5, "Call me back", address = "+919876543210"),
    )
    private val searchSpam = listOf(row(4, "Win a HDFC salary account bonus", folder = Folder.SPAM))
    private val searchMessages = listOf(
        SearchableMessage(11, 1, Folder.INBOX, "Rs 2,500 debited from a/c", 1_700_000_000_300L, MsgType.INBOX),
        SearchableMessage(12, 2, Folder.INBOX, "did the hdfc salary come?", 1_700_000_000_200L, MsgType.INBOX),
        SearchableMessage(13, 1, Folder.INBOX, "Your HDFC-Salary of Rs 85,000 is credited", 1_700_000_000_100L, MsgType.INBOX),
        SearchableMessage(14, 3, Folder.INBOX, "Dinner at 7?", 1_700_000_000_000L, MsgType.INBOX),
        SearchableMessage(15, 4, Folder.SPAM, "Win a HDFC salary account bonus", 1_699_000_000_000L, MsgType.INBOX),
    )

    @Test
    fun searchIgnoresSpacesAndPunctuationAndEachTabKeepsItsOwnMatches() {
        val opened = mutableListOf<ThreadKey>()
        show(searchInbox, searchSpam, searchMessages, onOpenThread = { opened += it })
        compose.onNodeWithTag("search").performClick()
        compose.onNodeWithTag("search_field").performTextInput("hdfc salary")
        compose.onNodeWithTag("conversation_2").assertIsDisplayed()
        compose.onNodeWithTag("conversation_3").assertDoesNotExist()
        compose.onNodeWithTag("conversation_5").assertDoesNotExist()
        // The row shows the message that matched, not the conversation's latest.
        compose.onNodeWithText("Your HDFC-Salary of Rs 85,000 is credited").assertIsDisplayed()
        compose.onNodeWithTag("conversation_1").performClick()
        assertEquals(listOf(ThreadKey(1, Folder.INBOX, messageId = 13, query = "hdfc salary")), opened)
        compose.onNodeWithTag("tab_spam").performClick()
        compose.onNodeWithTag("conversation_4").assertIsDisplayed()
        compose.onNodeWithTag("tab_inbox").performClick()
        compose.onNodeWithContentDescription("Close search").performClick()
        compose.onNodeWithTag("conversation_3").assertIsDisplayed()
        compose.onNodeWithText("Rs 2,500 debited from a/c").assertIsDisplayed()
    }

    @Test
    fun searchFindsANumberHoweverItsSpacedAndOpensItsConversation() {
        val opened = mutableListOf<ThreadKey>()
        show(searchInbox, searchSpam, searchMessages, onOpenThread = { opened += it })
        compose.onNodeWithTag("search").performClick()
        compose.onNodeWithTag("search_field").performTextInput("98765 43210")
        compose.onNodeWithTag("conversation_5").assertIsDisplayed()
        compose.onNodeWithTag("conversation_1").assertDoesNotExist()
        compose.onNodeWithTag("conversation_5").performClick()
        assertEquals(listOf(ThreadKey(5, Folder.INBOX, query = "98765 43210")), opened)
        compose.onNodeWithTag("search_clear").performClick()
        compose.onNodeWithTag("conversation_1").assertIsDisplayed()
    }

    @Test
    fun theUnreadFilterNarrowsSearchResults() {
        show(searchInbox, searchSpam, searchMessages)
        compose.onNodeWithTag("search").performClick()
        compose.onNodeWithTag("search_field").performTextInput("hdfc")
        compose.onNodeWithTag("conversation_2").assertIsDisplayed()
        compose.onNodeWithTag("filter_unread").performClick()
        compose.onNodeWithTag("conversation_1").assertIsDisplayed()
        compose.onNodeWithTag("conversation_2").assertDoesNotExist()
        compose.onNodeWithTag("search_field").performTextInput("zzz")
        compose.onNodeWithText("No matches in this tab").assertIsDisplayed()
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

    @Test
    fun aSearchIsHighlightedInTheMessage() {
        val m = ThreadMessageRow(
            9, 3, "+16505550003", "Pay your toll: https://ezpass-tolls.top/pay", 0, MsgType.INBOX, -1,
            true, LabelCode.SPAM, "ENGINE", 0.97f, "MODEL", null, null, 1, 0,
        )
        compose.setContent { SiftTheme { MessageBubble(m, onWhy = {}, onMark = {}, onDelete = {}, query = "ezpass tolls") } }
        val text = compose.onNodeWithTag("body_9", useUnmergedTree = true).fetchSemanticsNode().config[SemanticsProperties.Text].single()
        assertEquals(listOf("ezpass-tolls"), text.spanStyles.map { text.text.substring(it.start, it.end) })
        assertTrue(text.getLinkAnnotations(0, text.length).isEmpty())
    }
}
