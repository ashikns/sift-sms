package dev.ashik.sift.telephony

import android.content.Context
import android.telephony.PhoneNumberUtils
import android.telephony.TelephonyManager
import dev.ashik.sift.spamcore.SenderKind
import java.util.Locale

/**
 * A sender's normalized identity. Per-sender state (verdicts, contacts, trust) is keyed by [key], never by
 * thread id, because the provider can recreate a thread with a new id.
 */
data class SenderKey(
    val key: String,
    val kind: SenderKind,
    /** A +1 (North American Numbering Plan) number. */
    val nanp: Boolean,
    /** The number's country calling code differs from the SIM's. */
    val international: Boolean,
) {
    val canReply: Boolean get() = kind == SenderKind.PHONE || kind == SenderKind.SHORT_CODE
}

class AddressNormalizer(private val countryIso: () -> String) {

    fun normalize(address: String?): SenderKey =
        classify(address, countryIso()) { number, iso -> PhoneNumberUtils.formatNumberToE164(number, iso) }

    companion object {
        fun forContext(context: Context): AddressNormalizer {
            val tm = context.getSystemService(TelephonyManager::class.java)
            return AddressNormalizer {
                listOfNotNull(tm?.simCountryIso, tm?.networkCountryIso, Locale.getDefault().country)
                    .firstOrNull { it.length == 2 }?.uppercase(Locale.ROOT) ?: "US"
            }
        }

        /** Pure classification; [e164] formats a number for a region or returns null when it can't. */
        fun classify(raw: String?, iso: String, e164: (String, String) -> String?): SenderKey {
            val a = raw?.trim().orEmpty()
            if (a.isEmpty()) return SenderKey("unknown", SenderKind.UNKNOWN, nanp = false, international = false)
            if ('@' in a) return SenderKey(a.lowercase(Locale.ROOT), SenderKind.EMAIL, nanp = false, international = false)
            val digits = a.filter { it in '0'..'9' }
            if (a.any { it.isLetter() }) {
                val key = a.uppercase(Locale.ROOT).split(' ', '\t').filter { it.isNotEmpty() }.joinToString(" ")
                return SenderKey(key, SenderKind.ALPHANUMERIC, nanp = false, international = false)
            }
            if (digits.isEmpty()) return SenderKey(a, SenderKind.UNKNOWN, nanp = false, international = false)
            val plus = a.startsWith("+")
            if (!plus && digits.length in 3..6) {
                return SenderKey(digits, SenderKind.SHORT_CODE, nanp = false, international = false)
            }
            val region = iso.uppercase(Locale.ROOT)
            val home = CountryCodes.callingCode(region)
            val key = runCatching { e164(a, region) }.getOrNull()?.takeIf { it.startsWith("+") }
                ?: fallbackE164(digits, plus, home)
            val nanp = key.startsWith("+1") && key.length == 12
            val international = home != null && key.startsWith("+") && !key.startsWith("+$home")
            return SenderKey(key, SenderKind.PHONE, nanp, international)
        }

        private fun fallbackE164(digits: String, plus: Boolean, home: String?): String = when {
            plus -> "+$digits"
            digits.startsWith("00") && digits.length > 6 -> "+" + digits.drop(2)
            digits.startsWith("011") && digits.length > 7 -> "+" + digits.drop(3)
            home == "1" && digits.length == 10 -> "+1$digits"
            home == "1" && digits.length == 11 && digits.startsWith("1") -> "+$digits"
            else -> digits
        }
    }
}

/** ISO 3166 alpha-2 region → ITU calling code (no libphonenumber on the public SDK surface). */
object CountryCodes {
    private val table: Map<String, String> by lazy {
        TABLE.trim().split(Regex("\\s+")).associate { entry -> entry.substring(0, 2) to entry.substring(2) }
    }

    fun callingCode(iso: String): String? = table[iso.uppercase(Locale.ROOT)]

    private const val TABLE = """
        AD376 AE971 AF93 AG1 AI1 AL355 AM374 AO244 AR54 AS1 AT43 AU61 AW297 AZ994 BA387 BB1 BD880 BE32 BF226 BG359
        BH973 BI257 BJ229 BM1 BN673 BO591 BR55 BS1 BT975 BW267 BY375 BZ501 CA1 CD243 CF236 CG242 CH41 CI225 CK682
        CL56 CM237 CN86 CO57 CR506 CU53 CV238 CY357 CZ420 DE49 DJ253 DK45 DM1 DO1 DZ213 EC593 EE372 EG20 ER291 ES34
        ET251 FI358 FJ679 FM691 FO298 FR33 GA241 GB44 GD1 GE995 GF594 GH233 GI350 GL299 GM220 GN224 GP590 GQ240 GR30
        GT502 GU1 GW245 GY592 HK852 HN504 HR385 HT509 HU36 ID62 IE353 IL972 IN91 IQ964 IR98 IS354 IT39 JM1 JO962 JP81
        KE254 KG996 KH855 KI686 KM269 KN1 KP850 KR82 KW965 KY1 KZ7 LA856 LB961 LC1 LI423 LK94 LR231 LS266 LT370 LU352
        LV371 LY218 MA212 MC377 MD373 ME382 MG261 MH692 MK389 ML223 MM95 MN976 MO853 MP1 MQ596 MR222 MS1 MT356 MU230
        MV960 MW265 MX52 MY60 MZ258 NA264 NC687 NE227 NG234 NI505 NL31 NO47 NP977 NR674 NZ64 OM968 PA507 PE51 PF689
        PG675 PH63 PK92 PL48 PR1 PS970 PT351 PW680 PY595 QA974 RE262 RO40 RS381 RU7 RW250 SA966 SB677 SC248 SD249
        SE46 SG65 SI386 SK421 SL232 SM378 SN221 SO252 SR597 SS211 ST239 SV503 SX1 SY963 SZ268 TC1 TD235 TG228 TH66
        TJ992 TL670 TM993 TN216 TO676 TR90 TT1 TV688 TW886 TZ255 UA380 UG256 US1 UY598 UZ998 VA39 VC1 VE58 VG1 VI1
        VN84 VU678 WS685 XK383 YE967 YT262 ZA27 ZM260 ZW263
    """
}
