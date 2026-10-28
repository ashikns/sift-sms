package dev.ashik.sift

import dev.ashik.sift.spamcore.SenderKind
import dev.ashik.sift.telephony.AddressNormalizer
import dev.ashik.sift.telephony.CountryCodes
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class AddressNormalizerTest {
    /** Stand-in for PhoneNumberUtils.formatNumberToE164 (Android-only): handles US 10-digit numbers. */
    private val fakeE164: (String, String) -> String? = { n, iso ->
        val d = n.filter { it.isDigit() }
        when {
            n.startsWith("+") -> "+$d"
            iso == "US" && d.length == 10 -> "+1$d"
            else -> null
        }
    }

    private fun classify(raw: String?, iso: String = "US") = AddressNormalizer.classify(raw, iso, fakeE164)

    @Test
    fun `us numbers normalize to e164 regardless of punctuation`() {
        val a = classify("(650) 555-0001")
        val b = classify("+1 650 555 0001")
        assertEquals("+16505550001", a.key)
        assertEquals(a.key, b.key)
        assertEquals(SenderKind.PHONE, a.kind)
        assertFalse(a.international)
    }

    @Test
    fun `short codes keep their digits`() {
        val s = classify("72975")
        assertEquals(SenderKind.SHORT_CODE, s.kind)
        assertEquals("72975", s.key)
    }

    @Test
    fun `alphanumeric and email senders keep a normalized key`() {
        val alpha = classify("  Amazon  Alerts ")
        assertEquals(SenderKind.ALPHANUMERIC, alpha.kind)
        assertEquals("AMAZON ALERTS", alpha.key)
        // A DLT header keeps its route prefix and category suffix; spamcore reads the sid and category from the key.
        val dlt = classify("cp-Blnkit-P")
        assertEquals(SenderKind.ALPHANUMERIC, dlt.kind)
        assertEquals("CP-BLNKIT-P", dlt.key)
        val email = classify("Bob@Example.COM")
        assertEquals(SenderKind.EMAIL, email.kind)
        assertEquals("bob@example.com", email.key)
    }

    @Test
    fun `foreign numbers are international for a US sim`() {
        val uk = classify("+447700900123")
        assertEquals(SenderKind.PHONE, uk.kind)
        assertTrue(uk.international)
        assertFalse(classify("+447700900123", iso = "GB").international)
    }

    @Test
    fun `fallbacks when the formatter gives up`() {
        assertEquals("+16505550001", classify("16505550001").key)
        assertEquals("+447700900123", classify("00447700900123", iso = "FR").key)
        assertEquals("5550001", classify("5550001").key)
        assertEquals(SenderKind.UNKNOWN, classify("").kind)
        assertEquals(SenderKind.UNKNOWN, classify(null).kind)
    }

    @Test
    fun `country calling codes`() {
        assertEquals("1", CountryCodes.callingCode("us"))
        assertEquals("44", CountryCodes.callingCode("GB"))
        assertEquals("91", CountryCodes.callingCode("IN"))
        assertEquals(null, CountryCodes.callingCode("ZZ"))
    }
}
