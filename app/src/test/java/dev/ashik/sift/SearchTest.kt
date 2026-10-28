package dev.ashik.sift

import dev.ashik.sift.data.db.ConversationEntity
import dev.ashik.sift.data.db.ConversationRow
import dev.ashik.sift.data.db.Folder
import dev.ashik.sift.data.db.LabelCode
import dev.ashik.sift.data.db.MsgType
import dev.ashik.sift.data.db.SearchableMessage
import dev.ashik.sift.ui.conversations.SearchIndex
import dev.ashik.sift.ui.conversations.SearchText
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class SearchTest {
    @Test
    fun `keys ignore case, spaces, punctuation and symbols`() {
        assertEquals("hdfcsalary", SearchText.key("HDFC-Salary"))
        assertEquals("hdfcsalary", SearchText.key(" hdfc  salary "))
        assertEquals("919876543210", SearchText.key("+91 98765-43210"))
        assertEquals("rs500", SearchText.key("Rs.500 ₹ 🎉❤️"))
        assertEquals("", SearchText.key(" .-+ "))
    }

    @Test
    fun `keys keep the vowel signs of Indic scripts`() {
        assertEquals("नमस्ते", SearchText.key("नमस्ते!"))
        assertTrue(SearchText.key("आपका खाता").contains(SearchText.key("खाता")))
    }

    @Test
    fun `ranges map matches back to the text across punctuation`() {
        val text = "Your HDFC-Salary of Rs 85,000 is in. hdfc salary!"
        assertEquals(listOf(5..15, 37..47), SearchText.ranges(text, "hdfc salary"))
        assertEquals(listOf(23..28), SearchText.ranges(text, "85000"))
        assertEquals(emptyList<IntRange>(), SearchText.ranges(text, "loan"))
        assertEquals(emptyList<IntRange>(), SearchText.ranges(text, " - "))
        assertEquals(emptyList<IntRange>(), SearchText.ranges(text, null))
    }

    @Test
    fun `an excerpt starts at a word shortly before a deep match`() {
        assertEquals("Your PNR is AB-1234", SearchText.excerpt("Your PNR is AB-1234", "ab1234"))
        val long = "Dear customer, thank you for booking with us. Your ticket for train 12627 is confirmed: PNR AB-1234."
        assertEquals("…12627 is confirmed: PNR AB-1234.", SearchText.excerpt(long, "ab1234"))
        assertEquals(long, SearchText.excerpt(long, "nothing"))
    }

    private fun row(thread: Long, folder: Int = Folder.INBOX, address: String = "+1650555000$thread", name: String? = null) =
        ConversationRow(
            ConversationEntity(thread, folder, address, address, thread, 1_000L * thread, "latest", MsgType.INBOX, 0, 1, LabelCode.HAM),
            displayName = name, photoUri = null,
        )

    private fun msg(id: Long, thread: Long, body: String, date: Long, folder: Int = Folder.INBOX) =
        SearchableMessage(id, thread, folder, body, date, MsgType.INBOX)

    @Test
    fun `conversations match by sender first, else by their newest matching message in the tab`() {
        val index = SearchIndex(
            listOf(
                msg(14, 4, "Win a HDFC salary account", 900, Folder.SPAM),
                msg(13, 1, "Your HDFC-Salary of Rs 85,000 is credited", 800),
                msg(12, 2, "did the hdfc salary come?", 700),
                msg(11, 1, "Older HDFC salary credit", 100),
                msg(10, 3, "Dinner at 7?", 50),
            ),
        )
        val inbox = listOf(row(1, name = "HDFCBK"), row(2, name = "Amit"), row(3, name = "Alex"), row(5, name = "HDFC Salary Desk"))
        val spam = listOf(row(4, Folder.SPAM), row(1, Folder.SPAM, name = "HDFCBK"))
        val found = index.search(inbox, spam, SearchText.key("hdfc salary"))
        // Thread 5 matched by name (its latest date, 5000), the others by text (their match's date).
        assertEquals(listOf(5L to null, 1L to 13L, 2L to 12L), found.inbox.map { it.row.conversation.threadId to it.match?.id })
        assertEquals(listOf(4L to 14L), found.spam.map { it.row.conversation.threadId to it.match?.id })
        assertEquals(listOf(1L to null), index.search(inbox, spam, SearchText.key("hdfcbk")).spam.map { it.row.conversation.threadId to it.match?.id })
    }

    @Test
    fun `a number matches however it's spaced`() {
        val index = SearchIndex(emptyList())
        val inbox = listOf(row(1, address = "+919876543210"), row(2))
        assertEquals(listOf(1L), index.search(inbox, emptyList(), SearchText.key("98765 43210")).inbox.map { it.row.conversation.threadId })
    }
}
