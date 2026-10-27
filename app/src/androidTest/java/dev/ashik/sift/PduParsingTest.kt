package dev.ashik.sift

import android.content.Intent
import android.provider.Telephony
import androidx.test.ext.junit.runners.AndroidJUnit4
import dev.ashik.sift.incoming.IncomingSmsParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream

/** Builds 3GPP SMS-DELIVER PDUs by hand (no SMSC address) so parsing is tested against the real framework. */
object PduBuilder {
    private const val GSM_BASIC =
        "@£\$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞ\u001BÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?" +
            "¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà"

    fun septets(text: String): IntArray = text.map { c ->
        GSM_BASIC.indexOf(c).also { require(it >= 0) { "not in the GSM default alphabet: $c" } }
    }.toIntArray()

    fun pack(septets: IntArray, fillBits: Int): ByteArray {
        val out = ByteArray((fillBits + septets.size * 7 + 7) / 8)
        var bit = fillBits
        for (s in septets) {
            for (b in 0 until 7) {
                if ((s shr b) and 1 == 1) out[bit / 8] = (out[bit / 8].toInt() or (1 shl (bit % 8))).toByte()
                bit++
            }
        }
        return out
    }

    private fun bcd(digits: String): ByteArray = ByteArray((digits.length + 1) / 2) { i ->
        val lo = digits[2 * i] - '0'
        val hi = if (2 * i + 1 < digits.length) digits[2 * i + 1] - '0' else 0xF
        ((hi shl 4) or lo).toByte()
    }

    /** [concat] = (reference, total, sequence) adds a concatenation header. */
    fun deliver(from: String, text: String, ucs2: Boolean = false, alphanumericSender: Boolean = false, concat: Triple<Int, Int, Int>? = null): ByteArray {
        val out = ByteArrayOutputStream()
        out.write(0x00) // no SMSC address
        out.write(if (concat != null) 0x44 else 0x04) // SMS-DELIVER, more-messages-to-send = no, UDHI
        if (alphanumericSender) {
            val s = septets(from)
            out.write((s.size * 7 + 3) / 4)
            out.write(0xD0)
            out.write(pack(s, 0))
        } else {
            val digits = from.removePrefix("+")
            out.write(digits.length)
            out.write(if (from.startsWith("+")) 0x91 else 0x81)
            out.write(bcd(digits))
        }
        out.write(0x00) // TP-PID
        out.write(if (ucs2) 0x08 else 0x00) // TP-DCS
        out.write(byteArrayOf(0x62, 0x01, 0x72, 0x21, 0x43, 0x65, 0x00)) // 2026-10-27 12:34:56 GMT
        val udh = concat?.let { (ref, total, seq) -> byteArrayOf(0x05, 0x00, 0x03, ref.toByte(), total.toByte(), seq.toByte()) }
        if (ucs2) {
            val ud = (udh ?: ByteArray(0)) + text.toByteArray(Charsets.UTF_16BE)
            out.write(ud.size)
            out.write(ud)
        } else {
            val s = septets(text)
            if (udh == null) {
                out.write(s.size)
                out.write(pack(s, 0))
            } else {
                val fill = (7 - (udh.size * 8) % 7) % 7
                out.write((udh.size * 8 + fill) / 7 + s.size)
                out.write(udh)
                out.write(pack(s, fill))
            }
        }
        return out.toByteArray()
    }

    fun intent(vararg pdus: ByteArray): Intent = Intent(Telephony.Sms.Intents.SMS_DELIVER_ACTION)
        .putExtra("pdus", arrayOf<Any>(*pdus))
        .putExtra("format", "3gpp")
        .putExtra("subscription", 1)
}

@RunWith(AndroidJUnit4::class)
class PduParsingTest {
    @Test
    fun gsm7bit() {
        val sms = IncomingSmsParser.parse(PduBuilder.intent(PduBuilder.deliver("+16505550123", "Hello from the test, 100% sure?")))!!
        assertEquals("+16505550123", sms.address)
        assertEquals("Hello from the test, 100% sure?", sms.body)
        assertEquals(1, sms.subId)
        assertFalse(sms.isClass0)
    }

    @Test
    fun ucs2() {
        val text = "Café ☕ at 5? Ünïcödé ok"
        val sms = IncomingSmsParser.parse(PduBuilder.intent(PduBuilder.deliver("+16505550123", text, ucs2 = true)))!!
        assertEquals(text, sms.body)
    }

    @Test
    fun alphanumericSender() {
        val sms = IncomingSmsParser.parse(PduBuilder.intent(PduBuilder.deliver("AMAZON", "Your order shipped", alphanumericSender = true)))!!
        assertEquals("AMAZON", sms.address)
        assertEquals("Your order shipped", sms.body)
    }

    @Test
    fun concatenatedPartsAreJoined() {
        val one = PduBuilder.deliver("6505550123", "Part one of the message, ", concat = Triple(7, 2, 1))
        val two = PduBuilder.deliver("6505550123", "and part two.", concat = Triple(7, 2, 2))
        val sms = IncomingSmsParser.parse(PduBuilder.intent(one, two))!!
        assertEquals("Part one of the message, and part two.", sms.body)
        assertEquals("6505550123", sms.address)
    }
}
