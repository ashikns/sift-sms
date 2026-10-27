package dev.ashik.sift

import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.ashik.sift.data.db.MsgStatus
import dev.ashik.sift.data.db.MsgType
import dev.ashik.sift.telephony.SmsProviderGateway
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith

/** Exercises the real system SMS provider. Rows are dated 2020 so the mirror drops them as soon as they're gone. */
@RunWith(AndroidJUnit4::class)
class ProviderGatewayTest {
    companion object {
        @BeforeClass
        @JvmStatic
        fun holdSmsRole() {
            InstrumentationRegistry.getInstrumentation().uiAutomation
                .executeShellCommand("cmd role add-role-holder android.app.role.SMS dev.ashik.sift").close()
            Thread.sleep(1_000)
        }
    }

    private val gateway = SmsProviderGateway(ApplicationProvider.getApplicationContext())
    private val old = 1_600_000_000_000L

    @Test
    fun incomingInsertQueryReadDelete() {
        val id = gateway.insertIncoming("+16505559876", "gateway test", old, old, 1, 0, false, null)
        assertTrue(id > 0)
        val row = gateway.query(id)!!
        assertEquals(MsgType.INBOX, row.type)
        assertFalse(row.read)
        assertTrue("provider assigns the thread", row.threadId > 0)
        assertEquals(id, gateway.findDuplicateIncoming("+16505559876", old, "gateway test"))
        assertTrue(gateway.markThreadRead(row.threadId) >= 1)
        assertTrue(gateway.query(id)!!.read)
        assertEquals(1, gateway.deleteMessage(id))
        assertNull(gateway.query(id))
    }

    @Test
    fun outgoingLifecycle() {
        val id = gateway.insertOutgoing("+16505559877", "outgoing test", old, 1, deliveryReports = true)
        assertEquals(MsgType.OUTBOX, gateway.query(id)!!.type)
        assertEquals(MsgStatus.PENDING, gateway.query(id)!!.status)
        gateway.updateType(id, MsgType.SENT)
        gateway.updateStatus(id, MsgStatus.COMPLETE)
        val row = gateway.query(id)!!
        assertEquals(MsgType.SENT, row.type)
        assertEquals(MsgStatus.COMPLETE, row.status)
        assertTrue(gateway.keysAfter(id - 1).any { it.id == id && it.body == null })
        assertEquals(1, gateway.deleteMessage(id))
    }
}
