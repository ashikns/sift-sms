package dev.ashik.sift.spamcore.features

import dev.ashik.sift.spamcore.SenderKind
import dev.ashik.sift.spamcore.normalize.Normalized
import dev.ashik.sift.spamcore.normalize.TextNormalizer
import java.util.TreeMap
import java.util.TreeSet
import kotlin.math.min
import kotlin.math.sqrt

/** A sparse vector with strictly ascending indices. */
class SparseVector(val indices: IntArray, val values: FloatArray) {
    val norm: Double by lazy {
        var s = 0.0
        for (v in values) s += v.toDouble() * v
        sqrt(s)
    }

    val size: Int get() = indices.size

    companion object {
        val EMPTY = SparseVector(IntArray(0), FloatArray(0))

        /** Cosine accumulated in double (ml/SPEC.md near-duplicate similarity). */
        fun cosine(a: SparseVector, b: SparseVector): Double {
            if (a.norm == 0.0 || b.norm == 0.0) return 0.0
            var i = 0
            var j = 0
            var dot = 0.0
            while (i < a.indices.size && j < b.indices.size) {
                val ai = a.indices[i]
                val bj = b.indices[j]
                when {
                    ai == bj -> { dot += a.values[i].toDouble() * b.values[j]; i++; j++ }
                    ai < bj -> i++
                    else -> j++
                }
            }
            return dot / (a.norm * b.norm)
        }
    }
}

/** Sender facts for metadata features 21-26 (always false for offline corpus messages, even ones with a sender). */
data class SenderFeatures(
    val kind: SenderKind = SenderKind.UNKNOWN,
    val nanp: Boolean = false,
    val international: Boolean = false,
    val firstContact: Boolean = false,
) {
    companion object {
        val NONE = SenderFeatures()
    }
}

/** feat_v2 (ml/SPEC.md): mirrors ml/src/smsml/features.py. */
object Features {
    const val VERSION = "feat_v2"
    const val BITS = 18
    const val BUCKETS = 1 shl BITS
    const val MASK = BUCKETS - 1
    const val SEED = 0x53504D31
    const val N_META = 27
    val SENDER_RANGE = 21 until 27

    val META_NAMES = listOf(
        "len_log", "url_count", "url_shortener", "url_ip_host", "url_risky_tld", "url_punycode_or_at",
        "url_brand_lookalike", "phone_number", "email_address", "currency_amount", "optout_stop", "otp_pattern",
        "upper_ratio", "digit_ratio", "excl_count", "emoji_count", "invisible", "lookalike", "dropped_ratio",
        "non_gsm7", "url_official", "snd_short_code", "snd_long_code_nanp", "snd_alphanumeric", "snd_email",
        "snd_international", "snd_first_contact",
    )

    private val CURRENCY_TOKENS: Set<Char> = TextNormalizer.CURRENCY.map { it.toChar() }.toSet() + '$'
    private const val PHONE_CHARS = "0123456789+-.() "
    private const val PHONE_START = "0123456789+("
    private val OPTOUT_VERBS = listOf("reply", "text", "txt", "send", "opt")
    private val OTP_WORDS = listOf("code", "otp", "passcode", "verification", "pin", "0fa")
    private val LN_1001 = StrictMath.log(1001.0)

    fun wordTokens(ng: String): List<String> {
        val out = ArrayList<String>()
        val cur = StringBuilder()
        for (c in ng) {
            if (c in 'a'..'z' || c in '0'..'9') {
                cur.append(c)
                continue
            }
            if (cur.isNotEmpty()) {
                out.add(cur.toString())
                cur.setLength(0)
            }
            if (c in CURRENCY_TOKENS) out.add(c.toString())
        }
        if (cur.isNotEmpty()) out.add(cur.toString())
        return out
    }

    /**
     * sid (ml/SPEC.md): the identity of a normalized sender key ([dev.ashik.sift.spamcore.SenderInfo.key]), or null.
     * An Indian DLT header `AX-ICICIT-S` is its sender ID `ICICIT` (the route prefix varies per message; the category
     * suffix is not used); any other key is itself, with unpaired surrogates as U+FFFD.
     */
    fun senderId(key: String?): String? {
        if (key.isNullOrEmpty() || key == "unknown") return null
        val parts = key.split('-')
        val dlt = parts.size in 2..3 &&
            parts[0].length == 2 && parts[0].all { it in 'A'..'Z' } &&
            parts[1].length in 3..8 && parts[1].all { it in 'A'..'Z' || it in '0'..'9' } &&
            (parts.size == 2 || (parts[2].length == 1 && parts[2][0] in 'A'..'Z'))
        return when {
            dlt -> parts[1]
            key.none { it.isSurrogate() } -> key
            else -> buildString { key.codePoints().forEach { appendCodePoint(if (it in 0xD800..0xDFFF) 0xFFFD else it) } }
        }
    }

    fun wordStrings(n: Normalized): TreeSet<String> {
        val toks = wordTokens(n.ng)
        val words = TreeSet<String>()
        for (t in toks) words.add("w:$t")
        for (i in 0 until toks.size - 1) words.add("b:" + toks[i] + " " + toks[i + 1])
        return words
    }

    fun urlStrings(n: Normalized): TreeSet<String> {
        val urls = TreeSet<String>()
        for (u in n.urls) {
            urls.add("u:" + u.domain)
            urls.add("h:" + u.host)
            if (u.tld.isNotEmpty()) urls.add("t:" + u.tld)
        }
        return urls
    }

    /** (char, word, url) feature-string families, each deduplicated and sorted. */
    fun featureStrings(n: Normalized): List<Collection<String>> {
        val s = " " + n.ng + " "
        val chars = TreeSet<String>()
        for (k in 3..5) for (i in 0..s.length - k) chars.add("c:" + s.substring(i, i + k))
        return listOf(chars, wordStrings(n), urlStrings(n))
    }

    /** The SND family: "s:" + sid + "|" + each WORD string and each URL `u:` string; none without a sid. */
    fun sndStrings(sid: String?, n: Normalized): List<String> {
        if (sid == null) return emptyList()
        val prefix = "s:$sid|"
        return (wordStrings(n) + urlStrings(n).filter { it.startsWith("u:") }).map { prefix + it }
    }

    /** Returns (bucket index, sign). */
    fun hashFeature(feature: String): Pair<Int, Int> {
        val h = Murmur3.hash32(feature.toByteArray(Charsets.UTF_8), SEED)
        return (h and MASK.toLong()).toInt() to (if ((h ushr 31) == 0L) 1 else -1)
    }

    private fun familyVector(strings: Collection<String>, idf: FloatArray): TreeMap<Int, Double> {
        val v = TreeMap<Int, Double>()
        for (f in strings) {
            val (idx, sign) = hashFeature(f)
            v[idx] = (v[idx] ?: 0.0) + sign * idf[idx].toDouble()
        }
        v.entries.removeIf { it.value == 0.0 }
        var sq = 0.0
        for (x in v.values) sq += x * x
        val norm = sqrt(sq)
        if (norm > 0.0) for (e in v.entries) e.setValue(e.value / norm)
        return v
    }

    /** Sum of the three L2-normalized family vectors (ascending indices, float32 values). */
    fun sparse(n: Normalized, idf: FloatArray): SparseVector {
        val total = TreeMap<Int, Double>()
        for (family in featureStrings(n)) {
            for ((k, x) in familyVector(family, idf)) total[k] = (total[k] ?: 0.0) + x
        }
        total.entries.removeIf { it.value == 0.0 }
        val idx = IntArray(total.size)
        val vals = FloatArray(total.size)
        var i = 0
        for ((k, x) in total) {
            idx[i] = k
            vals[i] = x.toFloat()
            i++
        }
        return SparseVector(idx, vals)
    }

    /**
     * The SND vector (sender × content): ±1 per [sndStrings] string summed per bucket, zeros dropped, L2-normalized,
     * no idf. Separate from [sparse], which alone drives near-duplicate similarity.
     */
    fun snd(sid: String?, n: Normalized): SparseVector {
        val v = TreeMap<Int, Int>()
        for (f in sndStrings(sid, n)) {
            val (idx, sign) = hashFeature(f)
            v[idx] = (v[idx] ?: 0) + sign
        }
        v.entries.removeIf { it.value == 0 }
        var sq = 0.0
        for (x in v.values) sq += x.toDouble() * x
        val norm = sqrt(sq)
        val idx = IntArray(v.size)
        val vals = FloatArray(v.size)
        var i = 0
        for ((k, x) in v) {
            idx[i] = k
            vals[i] = (x / norm).toFloat()
            i++
        }
        return SparseVector(idx, vals)
    }

    fun phoneNumber(clean: String): Boolean {
        var i = 0
        val n = clean.length
        while (i < n) {
            if (clean[i] in PHONE_START) {
                var j = i
                var digits = 0
                while (j < n && clean[j] in PHONE_CHARS) {
                    if (clean[j] in '0'..'9') digits++
                    j++
                }
                if (digits in 10..15) return true
                i = j
            } else {
                i++
            }
        }
        return false
    }

    fun emailAddress(clean: String, urlTokens: Set<Int>): Boolean {
        if (clean.isEmpty()) return false
        clean.split(' ').forEachIndexed { idx, tok ->
            if (idx in urlTokens) return@forEachIndexed
            val a = tok.indexOf('@')
            if (a >= 1 && tok.indexOf('.', a + 1) != -1) return true
        }
        return false
    }

    fun currencyAmount(ng: String, toks: List<String>): Boolean {
        val n = ng.length
        for (i in 0 until n) {
            if (ng[i] !in CURRENCY_TOKENS) continue
            if (i + 1 < n && ng[i + 1] == '0') return true
            if (i + 2 < n && ng[i + 1] == ' ' && ng[i + 2] == '0') return true
            if (i >= 1 && ng[i - 1] == '0') return true
            if (i >= 2 && ng[i - 1] == ' ' && ng[i - 2] == '0') return true
        }
        for (i in 1 until toks.size) {
            if (toks[i] in setOf("usd", "dollars", "bucks") && toks[i - 1].all { it == '0' }) return true
        }
        return false
    }

    fun metadata(n: Normalized, sender: SenderFeatures = SenderFeatures.NONE): FloatArray {
        val toks = wordTokens(n.ng)
        val words = toks.toHashSet()
        val m = DoubleArray(N_META)
        fun b(x: Boolean) = if (x) 1.0 else 0.0
        m[0] = min(StrictMath.log(1.0 + n.cpCount) / LN_1001, 1.0)
        m[1] = min(n.urls.size, 3) / 3.0
        m[2] = b(n.urls.any { it.shortener })
        m[3] = b(n.urls.any { it.ipHost })
        m[4] = b(n.urls.any { it.riskyTld })
        m[5] = b(n.urls.any { it.punycode || it.atSign })
        m[6] = b(n.urls.any { it.brandLookalike })
        m[7] = b(phoneNumber(n.clean))
        m[8] = b(emailAddress(n.clean, n.urlTokens))
        m[9] = b(currencyAmount(n.ng, toks))
        m[10] = b("stop" in words && OPTOUT_VERBS.any { it in words })
        val hasCode = toks.any { it.length in 4..8 && it.all { c -> c == '0' } }
        val hasWord = OTP_WORDS.any { it in words } || (0 until toks.size - 1).any { toks[it] == "one" && toks[it + 1] == "time" }
        m[11] = b(hasCode && hasWord)
        var letters = 0
        var upper = 0
        var digits = 0
        var nonSpace = 0
        var excl = 0
        for (c in n.clean) {
            if (c in 'a'..'z') letters++
            if (c in 'A'..'Z') { letters++; upper++ }
            if (c in '0'..'9') digits++
            if (c != ' ') nonSpace++
            if (c == '!') excl++
        }
        m[12] = upper.toDouble() / maxOf(1, letters)
        m[13] = digits.toDouble() / maxOf(1, nonSpace)
        m[14] = min(excl, 5) / 5.0
        m[15] = min(n.emojiCount, 5) / 5.0
        m[16] = min(n.invisibleCount, 3) / 3.0
        m[17] = min(n.lookalikeCount, 4) / 4.0
        m[18] = min(n.droppedCount.toDouble() / maxOf(1, n.cpCount), 1.0)
        m[19] = b(n.nonGsm7)
        m[20] = b(n.urls.any { it.official })
        m[21] = b(sender.kind == SenderKind.SHORT_CODE)
        m[22] = b(sender.kind == SenderKind.PHONE && sender.nanp)
        m[23] = b(sender.kind == SenderKind.ALPHANUMERIC)
        m[24] = b(sender.kind == SenderKind.EMAIL)
        m[25] = b(sender.international)
        m[26] = b(sender.firstContact)
        return FloatArray(N_META) { m[it].toFloat() }
    }
}
