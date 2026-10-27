package dev.ashik.sift.spamcore.normalize

import java.text.Normalizer

data class Url(
    val host: String,
    val tld: String,
    val domain: String,
    val ipHost: Boolean,
    val punycode: Boolean,
    val atSign: Boolean,
    val shortener: Boolean,
    val riskyTld: Boolean,
    val brandLookalike: Boolean,
    val official: Boolean,
)

data class Normalized(
    val clean: String,
    val enc: String,
    val ng: String,
    val urls: List<Url>,
    /** Indices of `clean`'s space-separated tokens that are URLs. */
    val urlTokens: Set<Int>,
    val cpCount: Int,
    val emojiCount: Int,
    val invisibleCount: Int,
    val lookalikeCount: Int,
    val droppedCount: Int,
    val nonGsm7: Boolean,
)

/**
 * norm_v1 (ml/SPEC.md): mirrors ml/src/smsml/normalize.py exactly. Character classes are explicit code-point
 * lists; there are no regular expressions (Android's regex engine is ICU-based and would diverge from the JVM).
 */
class TextNormalizer(private val res: Resources) {

    fun normalize(raw: String): Normalized {
        // 1-2. code points and raw stats
        val cps = raw.codePoints().map { if (it in 0xD800..0xDFFF) 0xFFFD else it }.limit(MAX_CPS.toLong()).toArray()
        val nonGsm7 = cps.any { it !in res.gsm7 }
        val emojiCount = cps.count { inRanges(it, EMOJI) }
        // 3. invisible characters
        val kept = IntArrayBuilder(cps.size)
        var invisibleCount = 0
        for (cp in cps) {
            if (inRanges(cp, INVISIBLE)) {
                if (!inRanges(cp, INVISIBLE_UNCOUNTED)) invisibleCount++
            } else {
                kept.add(cp)
            }
        }
        // 4. lookalike census (before NFKC)
        var lookalikeCount = 0
        for (i in 0 until kept.size) {
            val cp = kept[i]
            if (cp < 0x80) continue
            val ch = String(Character.toChars(cp))
            val f = Normalizer.normalize(ch, Normalizer.Form.NFKC)
            if (f != ch && isAsciiAlnum(f)) lookalikeCount++
            else if (cp in res.confusables) lookalikeCount++
        }
        // 5. NFKC
        var text = Normalizer.normalize(kept.toCodePointString(), Normalizer.Form.NFKC)
        // 6. confusables
        text = mapCodePoints(text) { cp -> res.confusables[cp] }
        // 7. strip marks
        text = Normalizer.normalize(text, Normalizer.Form.NFD)
        text = buildString { text.codePoints().forEach { if (!inRanges(it, MARKS)) appendCodePoint(it) } }
        text = Normalizer.normalize(text, Normalizer.Form.NFC)
        // 8. punctuation map
        text = mapCodePoints(text) { cp -> PUNCT[cp] }
        // 9. restrict
        var dropped = 0
        val restricted = StringBuilder(text.length)
        text.codePoints().forEach { cp ->
            when {
                cp <= 0x1F || cp in 0x7F..0x9F -> restricted.append(' ')
                inRanges(cp, EMOJI) -> restricted.append(' ')
                cp in 0x20..0x7E || cp in CURRENCY -> restricted.appendCodePoint(cp)
                else -> {
                    restricted.append(' ')
                    if (cp != 0xFFFD) dropped++ // replacement chars come from decoding damage, not from the sender
                }
            }
        }
        // 10. collapse + trim
        val clean = restricted.split(' ').filter { it.isNotEmpty() }.joinToString(" ")
        // 11-12. URL scan, enc
        val tokens = if (clean.isEmpty()) emptyList() else clean.split(' ')
        val urls = ArrayList<Url>()
        val urlTokens = HashSet<Int>()
        val encTokens = ArrayList<String>(tokens.size)
        tokens.forEachIndexed { idx, tok ->
            var i = 0
            var j = tok.length
            while (i < j && tok[i] in LEAD_STRIP) i++
            while (j > i && tok[j - 1] in TRAIL_STRIP) j--
            val core = tok.substring(i, j)
            val url = if (core.isNotEmpty()) parseUrl(core) else null
            if (url != null) {
                urls.add(url)
                urlTokens.add(idx)
                encTokens.add(tok.substring(0, i) + url.host + tok.substring(j))
            } else {
                encTokens.add(tok)
            }
        }
        val enc = encTokens.joinToString(" ")
        // 13. n-gram view
        val ng = buildString(enc.length) {
            for (c in enc) append(
                when (c) {
                    in '0'..'9' -> '0'
                    in 'A'..'Z' -> c + 32
                    else -> c
                },
            )
        }
        return Normalized(
            clean = clean, enc = enc, ng = ng, urls = urls, urlTokens = urlTokens, cpCount = cps.size,
            emojiCount = emojiCount, invisibleCount = invisibleCount, lookalikeCount = lookalikeCount,
            droppedCount = dropped, nonGsm7 = nonGsm7,
        )
    }

    fun parseUrl(core: String): Url? {
        val low = asciiLower(core)
        val rest: String
        val schemed: Boolean
        when {
            low.startsWith("https://") -> { rest = low.substring(8); schemed = true }
            low.startsWith("http://") -> { rest = low.substring(7); schemed = true }
            else -> { rest = low; schemed = low.startsWith("www.") }
        }
        val end = authorityEnd(rest)
        var authority = rest.substring(0, end)
        var atSign = false
        if ('@' in authority) {
            if (!schemed) return null // a bare "name@host" is an email address, not a URL
            authority = authority.substring(authority.lastIndexOf('@') + 1)
            atSign = true
        }
        val colon = authority.indexOf(':')
        var host = if (colon == -1) authority else authority.substring(0, colon)
        val labels = host.split('.')
        if (labels.size < 2 || labels.any { it.isEmpty() }) return null
        val ip = isIpv4(host)
        if (!schemed && !ip) {
            if (labels.any { lab -> lab.any { it !in LABEL_CHARS } }) return null
            val tld = labels.last()
            if (!isLetters(tld) || tld !in res.tlds || isDigits(labels[0])) return null
            if (tld in res.tldStoplist) {
                val second = labels[labels.size - 2]
                val evidence = '/' in rest.substring(end) || labels.size >= 3 || '-' in second ||
                    second.any { it in '0'..'9' }
                if (!evidence) return null
            }
        }
        if (host.startsWith("www.")) host = host.substring(4)
        val hl = host.split('.')
        val tld: String
        val domain: String
        if (ip) {
            tld = ""
            domain = host
        } else {
            tld = hl.last()
            domain = if (hl.size >= 3 && (hl[hl.size - 2] + "." + hl.last()) in res.psl) {
                hl.subList(hl.size - 3, hl.size).joinToString(".")
            } else if (hl.size >= 2) {
                hl.subList(hl.size - 2, hl.size).joinToString(".")
            } else {
                host
            }
        }
        val words = host.replace('-', '.').split('.').toSet()
        val lookalike = res.brands.any { b ->
            ((b.substring && host.contains(b.keyword)) || b.keyword in words) && domain !in b.officials
        }
        return Url(
            host = host,
            tld = tld,
            domain = domain,
            ipHost = ip,
            punycode = hl.any { it.startsWith("xn--") },
            atSign = atSign,
            shortener = domain in res.shorteners,
            riskyTld = tld.isNotEmpty() && tld in res.riskyTlds,
            brandLookalike = lookalike,
            official = domain in res.officialDomains,
        )
    }

    private inline fun mapCodePoints(text: String, crossinline f: (Int) -> String?): String = buildString(text.length) {
        text.codePoints().forEach { cp -> val m = f(cp); if (m != null) append(m) else appendCodePoint(cp) }
    }

    private class IntArrayBuilder(capacity: Int) {
        private var data = IntArray(maxOf(capacity, 4))
        var size = 0
            private set

        fun add(v: Int) {
            if (size == data.size) data = data.copyOf(size * 2)
            data[size++] = v
        }

        operator fun get(i: Int) = data[i]

        fun toCodePointString(): String = String(data, 0, size)
    }

    companion object {
        const val MAX_CPS = 2000

        private val EMOJI = intArrayOf(0x1F000, 0x1FAFF, 0x2600, 0x27BF, 0x2300, 0x23FF, 0x2B00, 0x2BFF)
        private val INVISIBLE = intArrayOf(
            0x00AD, 0x00AD, 0x034F, 0x034F, 0x061C, 0x061C, 0x115F, 0x1160, 0x17B4, 0x17B5, 0x180B, 0x180F,
            0x200B, 0x200F, 0x202A, 0x202E, 0x2060, 0x206F, 0x3164, 0x3164, 0xFE00, 0xFE0F, 0xFEFF, 0xFEFF,
            0xFFA0, 0xFFA0, 0x1BCA0, 0x1BCA3, 0xE0000, 0xE007F, 0xE0100, 0xE01EF,
        )
        private val INVISIBLE_UNCOUNTED = intArrayOf(0x200D, 0x200D, 0xFE0E, 0xFE0F, 0xE0020, 0xE007F)
        private val MARKS = intArrayOf(0x0300, 0x036F, 0x1AB0, 0x1AFF, 0x1DC0, 0x1DFF, 0x20D0, 0x20FF, 0xFE20, 0xFE2F)
        private val PUNCT: Map<Int, String> = buildMap {
            for (cp in intArrayOf(0x2018, 0x2019, 0x201A, 0x201B, 0x2032, 0x02BC)) put(cp, "'")
            for (cp in intArrayOf(0x201C, 0x201D, 0x201E, 0x201F, 0x2033)) put(cp, "\"")
            for (cp in intArrayOf(0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212)) put(cp, "-")
            for (cp in intArrayOf(0x2022, 0x2043)) put(cp, "*")
        }
        val CURRENCY: Set<Int> = setOf(0x00A2, 0x00A3, 0x00A5, 0x20AC, 0x20B9)

        private const val LEAD_STRIP = "([{<\"'"
        private const val TRAIL_STRIP = ".,;:!?)]}>\"'"
        private const val LABEL_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-"

        /** [ranges] holds inclusive (lo, hi) pairs. */
        private fun inRanges(cp: Int, ranges: IntArray): Boolean {
            var i = 0
            while (i < ranges.size) {
                if (cp >= ranges[i] && cp <= ranges[i + 1]) return true
                i += 2
            }
            return false
        }

        private fun isAsciiAlnum(s: String): Boolean =
            s.isNotEmpty() && s.all { it in '0'..'9' || it in 'A'..'Z' || it in 'a'..'z' }

        fun asciiLower(s: String): String = buildString(s.length) { for (c in s) append(if (c in 'A'..'Z') c + 32 else c) }

        private fun isDigits(s: String) = s.isNotEmpty() && s.all { it in '0'..'9' }

        private fun isLetters(s: String) = s.isNotEmpty() && s.all { it in 'a'..'z' }

        private fun authorityEnd(s: String): Int {
            var end = s.length
            for (ch in "/?#") {
                val k = s.indexOf(ch)
                if (k != -1 && k < end) end = k
            }
            return end
        }

        private fun isIpv4(host: String): Boolean {
            val parts = host.split('.')
            if (parts.size != 4) return false
            return parts.all { it.length in 1..3 && isDigits(it) && it.toInt() <= 255 }
        }
    }
}
