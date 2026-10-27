package dev.ashik.sift

import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import dev.ashik.sift.data.db.AppDatabase
import dev.ashik.sift.data.db.ClassState
import dev.ashik.sift.data.db.ClassificationEntity
import dev.ashik.sift.data.db.ContactCacheEntity
import dev.ashik.sift.data.db.Folder
import dev.ashik.sift.data.db.LabelCode
import dev.ashik.sift.data.db.MessageEntity
import dev.ashik.sift.data.db.MsgType
import dev.ashik.sift.data.db.Source
import dev.ashik.sift.data.db.ThreadFolder
import dev.ashik.sift.data.db.TrainingExampleEntity
import dev.ashik.sift.data.repo.ConversationProjector
import dev.ashik.sift.spamcore.UserLabel
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class DatabaseTest {
    private lateinit var db: AppDatabase
    private lateinit var projector: ConversationProjector

    @Before
    fun setUp() {
        db = AppDatabase.build(ApplicationProvider.getApplicationContext(), name = null)
        projector = ConversationProjector(db)
    }

    @After
    fun tearDown() = db.close()

    private fun msg(id: Long, thread: Long, type: Int, date: Long, read: Boolean = true, sender: String = "+16505550001", kind: String = "PHONE") =
        MessageEntity(id, thread, sender, sender, kind, "m$id", date, date, type, -1, read, read, 1, 0, id, 0)

    @Test
    fun projectorPlacesConversationsByLatestIncomingMessage() = runBlocking {
        db.messages().upsertAll(
            listOf(
                msg(1, 10, MsgType.INBOX, 1000, read = false),
                msg(2, 10, MsgType.SENT, 2000),
                msg(3, 20, MsgType.INBOX, 1500, sender = "+16505550002"),
                msg(4, 20, MsgType.INBOX, 2500, read = false, sender = "+16505550002"),
                msg(5, 30, MsgType.INBOX, 900, sender = "AMAZON", kind = "ALPHANUMERIC"),
            ),
        )
        db.classifications().upsertAll(
            listOf(
                ClassificationEntity(3, ClassState.DONE, effectiveLabel = LabelCode.HAM, effectiveSource = Source.ENGINE),
                ClassificationEntity(4, ClassState.DONE, effectiveLabel = LabelCode.SPAM, effectiveSource = Source.ENGINE),
                ClassificationEntity(5, ClassState.DONE, effectiveLabel = LabelCode.SUSPICIOUS, effectiveSource = Source.ENGINE),
            ),
        )
        projector.rebuildAll()

        val c10 = db.conversations().get(10)!!
        assertEquals(Folder.INBOX, c10.folder)
        assertEquals(2L, c10.latestMessageId)
        assertEquals(1L, c10.latestIncomingId)
        assertEquals(MsgType.SENT, c10.latestType)
        assertEquals(1, c10.unreadCount)
        assertEquals(2, c10.messageCount)
        assertTrue(c10.canReply)

        val c20 = db.conversations().get(20)!!
        assertEquals(Folder.SPAM, c20.folder)
        assertEquals(LabelCode.SPAM, c20.latestLabel)

        val c30 = db.conversations().get(30)!!
        assertEquals(Folder.INBOX, c30.folder)
        assertEquals(LabelCode.SUSPICIOUS, c30.latestLabel)
        assertFalse(c30.canReply)

        // The user's "not spam" wins over the engine and moves the conversation back.
        db.classifications().setUserLabel(4, UserLabel.HAM.ordinal, 3000, LabelCode.HAM, Source.USER)
        projector.rebuild(listOf(20L))
        assertEquals(Folder.INBOX, db.conversations().get(20)!!.folder)
        assertEquals(listOf(10L, 20L, 30L), db.conversations().observeFolder(Folder.INBOX).first().map { it.conversation.threadId }.sorted())
    }

    @Test
    fun threadQueryJoinsVerdictsNewestFirst() = runBlocking {
        db.messages().upsertAll(listOf(msg(1, 10, MsgType.INBOX, 1000), msg(2, 10, MsgType.SENT, 2000), msg(3, 10, MsgType.INBOX, 3000)))
        db.classifications().upsert(ClassificationEntity(3, ClassState.DONE, modelScore = 0.97f, reason = "MODEL", effectiveLabel = LabelCode.SPAM))
        val rows = db.messages().observeThread(10, 50).first()
        assertEquals(listOf(3L, 2L, 1L), rows.map { it.id })
        assertEquals(LabelCode.SPAM, rows[0].effectiveLabel)
        assertEquals(0.97f, rows[0].modelScore!!, 1e-6f)
        assertEquals(LabelCode.HAM, rows[2].effectiveLabel) // no verdict yet: not spam
    }

    @Test
    fun rescanCoversEveryUnlabeledIncomingMessageAndResumes() = runBlocking {
        db.messages().upsertAll(
            listOf(
                msg(1, 10, MsgType.INBOX, 1000), // decades old, never classified
                msg(2, 10, MsgType.INBOX, 2000), // classified before the rescan started
                msg(3, 10, MsgType.INBOX, 3000), // classified again after it started
                msg(4, 10, MsgType.INBOX, 4000), // pending: ClassifyPendingWorker's
                msg(5, 10, MsgType.INBOX, 5000), // the user's label decides
                msg(6, 10, MsgType.SENT, 6000),
            ),
        )
        db.classifications().upsertAll(
            listOf(
                ClassificationEntity(2, ClassState.DONE, effectiveSource = Source.ENGINE, classifiedAt = 100),
                ClassificationEntity(3, ClassState.DONE, effectiveSource = Source.ENGINE, classifiedAt = 600),
                ClassificationEntity(4, ClassState.PENDING),
                ClassificationEntity(
                    5, ClassState.DONE, userLabel = UserLabel.SPAM.ordinal, effectiveLabel = LabelCode.SPAM,
                    effectiveSource = Source.USER, classifiedAt = 100,
                ),
            ),
        )
        assertEquals(3, db.messages().rescanCount())
        assertEquals(listOf(2L, 1L), db.messages().rescanRemainingIds(startedAt = 500))
        assertEquals(listOf(3L, 2L, 1L), db.messages().rescanRemainingIds(startedAt = 1000))
        projector.rebuildAll()
        assertEquals(listOf(ThreadFolder(10, Folder.SPAM)), db.conversations().folders())
    }

    @Test
    fun learningEntriesNewestFirstWithContactNames() = runBlocking {
        db.trainingExamples().upsertAll(
            listOf(
                TrainingExampleEntity(1, "Claim your gift card", "+16505550001", "PHONE", false, true, UserLabel.SPAM.ordinal, 1000, null),
                TrainingExampleEntity(2, "Your code is 123456", "BANK", "ALPHANUMERIC", false, false, UserLabel.HAM.ordinal, 2000, ByteArray(1536)),
            ),
        )
        db.contacts().upsertAll(listOf(ContactCacheEntity("+16505550001", "Alex", null, true, 0)))
        db.messages().upsertAll(listOf(msg(1, 10, MsgType.INBOX, 1000, sender = "6505550001")))
        val rows = db.trainingExamples().observeEntries().first()
        assertEquals(listOf(2L, 1L), rows.map { it.messageId })
        assertEquals(listOf(null, "Alex"), rows.map { it.displayName })
        assertEquals(listOf(UserLabel.HAM.ordinal, UserLabel.SPAM.ordinal), rows.map { it.label })
        assertEquals(listOf("BANK", "6505550001"), rows.map { it.address }) // the message's own address while it exists
    }

    @Test
    fun senderHistoryQueries() = runBlocking {
        db.messages().upsertAll(listOf(msg(1, 10, MsgType.INBOX, 1000), msg(2, 10, MsgType.INBOX, 2000), msg(3, 10, MsgType.SENT, 3000)))
        assertEquals(0, db.messages().earlierIncomingCount("+16505550001", 1, 1000))
        assertEquals(1, db.messages().earlierIncomingCount("+16505550001", 2, 2000))
        assertEquals(1, db.messages().outgoingCount("+16505550001"))
        assertEquals(10L, db.messages().threadForSender("+16505550001"))
    }
}
