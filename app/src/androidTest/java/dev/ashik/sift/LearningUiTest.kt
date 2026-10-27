package dev.ashik.sift

import androidx.compose.material3.SnackbarHostState
import androidx.compose.runtime.remember
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithContentDescription
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.test.ext.junit.runners.AndroidJUnit4
import dev.ashik.sift.data.db.LearningEntryRow
import dev.ashik.sift.spamcore.UserLabel
import dev.ashik.sift.ui.learning.LearningContent
import dev.ashik.sift.ui.theme.SiftTheme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class LearningUiTest {
    @get:Rule
    val compose = createComposeRule()

    private fun entry(id: Long, body: String, label: UserLabel, name: String? = null) =
        LearningEntryRow(id, body, label.ordinal, 1_800_000_000_000L - id, "650555000$id", name)

    private var deleted: LearningEntryRow? = null
    private var deletedAll = false

    private fun show(entries: List<LearningEntryRow>) = compose.setContent {
        SiftTheme {
            LearningContent(
                entries = entries, snackbar = remember { SnackbarHostState() }, onBack = {},
                onDelete = { deleted = it }, onDeleteAll = { deletedAll = true },
            )
        }
    }

    @Test
    fun listsEntriesAndDeletesOne() {
        show(listOf(entry(1, "Claim your gift card", UserLabel.SPAM), entry(2, "Your code is 123456", UserLabel.HAM, name = "Bank")))
        compose.onNodeWithText("2 entries: 1 spam, 1 not spam").assertIsDisplayed()
        compose.onNodeWithText("Claim your gift card").assertIsDisplayed()
        compose.onNodeWithText("Bank", substring = true).assertIsDisplayed()
        compose.onAllNodesWithContentDescription("Delete entry")[1].performClick()
        compose.waitForIdle()
        assertEquals(2L, deleted?.messageId)
    }

    @Test
    fun deleteAllAsksFirst() {
        show(listOf(entry(1, "Claim your gift card", UserLabel.SPAM)))
        compose.onNodeWithContentDescription("More options").performClick()
        compose.onNodeWithText("Delete all").performClick()
        compose.onNodeWithText("Delete all entries?").assertIsDisplayed()
        assertFalse(deletedAll)
        compose.onNodeWithText("Delete all").performClick()
        compose.waitForIdle()
        assertTrue(deletedAll)
    }

    @Test
    fun emptyStateExplains() {
        show(emptyList())
        compose.onNodeWithText("No entries yet", substring = true).assertIsDisplayed()
        compose.onNodeWithContentDescription("More options").assertDoesNotExist()
    }
}
