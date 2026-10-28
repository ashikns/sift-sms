package dev.ashik.sift

import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
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
        assertEquals(1, gateway.markRead(listOf(id)))
        assertTrue(gateway.query(id)!!.read)
        assertEquals(1, gateway.markUnread(listOf(id)))
        assertFalse(gateway.query(id)!!.read)
        assertEquals(0, gateway.markUnread(listOf(id)))
        assertEquals(1, gateway.deleteMessages(listOf(id)))
        assertNull(gateway.query(id))
    }
}
